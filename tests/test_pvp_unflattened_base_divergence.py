"""Eval's base reconstruction silently diverges from the trainer when an adapter's
`base_model_name_or_path` is NOT flattened to the foundation.

Parity between the trained base and the eval-served base hinges entirely on one
invariant: every uploaded adapter declares the *foundation* as its base
(trainer/utils/hf_upload.py:patch_model_metadata via _resolve_base_model). Under
that invariant the trainer's chain-walking merge (trainer_downloader.py
_detect_and_merge_lora, which follows base_model_name_or_path up to 10 hops and
merges every intermediate adapter) collapses to a single merge onto the
foundation — which is exactly what eval's single-element base_chain does.

But the invariant is best-effort and breaks in realistic ways:
  * patch_model_metadata swallows exceptions (hf_upload.py:74-76); a transient
    network failure inside _resolve_base_model leaves base_model_name_or_path
    pointing at the *previous adapter* instead of the foundation.
  * _resolve_base_model / _detect_and_merge_lora both cap the walk at 10 hops, so
    a lineage deeper than ~10 rounds resolves to an intermediate adapter.

When that happens the two sides DIVERGE, and they diverge differently:

  TRAINER  (_detect_and_merge_lora): chain-walks R2 -> R1 -> foundation and merges
           BOTH R1 and R2 onto the foundation.  base = foundation (+) R1 (+) R2.

  EVAL     (materialize_base_model): never chain-walks. It reads R2's declared
           base (R1, an *adapter* repo), tries to load it as a full base model,
           and merges only R2.  It drops R1's delta and uses an adapter repo as
           the merge root (a load that crashes in production).

This test pins the divergence: it asserts eval reconstructs the same base the
trainer would (foundation as merge root, both adapters applied). It FAILS on the
current code, demonstrating eval has no defense against an unflattened lineage.
"""

import json
import os

import validator.evaluation.pvp.materialize as mat


def _write_adapter_config(local_dir: str, declared_base: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, "adapter_config.json"), "w") as f:
        json.dump({"base_model_name_or_path": declared_base}, f)


def test_materialize_reconstructs_full_lineage_when_config_unflattened(monkeypatch):
    """A 2-round lineage whose top adapter is NOT flattened to the foundation.

    Lineage (what the miner actually trained on, per the trainer):
        foundation  --R1-->  M1  --R2-->  trained base for round 3
    R2's adapter_config declares its base as R1 (an adapter repo), NOT the
    foundation — i.e. flattening did not run / could not resolve.
    """
    foundation = "org/foundation"
    R1 = "org/miner-round1"          # an adapter repo
    R2 = "org/miner-round2"          # an adapter repo whose config points at R1

    # Map every repo to a declared base, mirroring what _declared_base would read
    # off each repo's adapter_config.json. R2 -> R1 (unflattened), R1 -> foundation.
    declared_base_of = {R2: R1, R1: foundation}

    downloaded_as_base: list[str] = []
    merged_adapters: list[str] = []

    def fake_download_lora(repo, local_dir, *a, **k):
        # The adapter repo R2 is what eval is handed in the base_chain; write the
        # config it would carry on HF so _declared_base resolves to R1.
        _write_adapter_config(local_dir, declared_base_of[repo])
        # Stash which logical repo this dir represents for the merge to read back.
        with open(os.path.join(local_dir, "_repo.txt"), "w") as f:
            f.write(repo)
        return local_dir

    def fake_download_model(repo, *a, **k):
        # Whatever _declared_base returns gets loaded here as a *full base model*.
        downloaded_as_base.append(repo)
        return f"/base/{repo.replace('/', '_')}"

    def fake_merge(base_path, lora_dir, output_dir="/tmp/merged_model", device=None):
        with open(os.path.join(lora_dir, "_repo.txt")) as f:
            merged_adapters.append(f.read())
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    monkeypatch.setattr(mat, "_download_lora_with_retry", fake_download_lora)
    monkeypatch.setattr(mat, "_download_model_with_retry", fake_download_model)
    monkeypatch.setattr(mat, "_merge_base_and_lora", fake_merge)

    # scoring._get_continuation_base_chains always produces a SINGLE-element chain:
    # [starting_repo] == [R2]. This is the only input eval gets.
    mat.materialize_base_model(foundation, [R2], label="a")

    # --- The parity requirement the trainer satisfies and eval must too ---

    # 1. The merge root must be the real foundation (a loadable full model), never
    #    an adapter repo. On current code eval loads R1 (an adapter) as a base.
    assert downloaded_as_base == [foundation], (
        f"eval loaded {downloaded_as_base} as the base model; the trainer chain-walks "
        f"to {foundation!r}. Loading an adapter repo as a full base crashes in prod."
    )

    # 2. Both R1 and R2 deltas must be applied — the trainer merges the whole chain.
    #    Eval drops R1, so the served base is missing the round-1 contribution.
    assert merged_adapters == [R1, R2], (
        f"eval merged {merged_adapters}; the trainer merges {[R1, R2]} (foundation->R1->R2). "
        f"Dropping R1 serves the round-2 adapter on a base it was never trained against."
    )
