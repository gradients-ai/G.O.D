"""Eval must reconstruct the same base the trainer trained on, across a multi-round lineage.

Each uploaded adapter declares its immediate parent (trainer/containers/uploader.py
patch_model_metadata), so a round-3 submission's lineage is
    foundation --R1--> M1 --R2--> M2 --R3-->
Both sides have to walk that whole chain or they serve a model on weights it never saw:

  TRAINER  (trainer/containers/downloader.py _detect_and_merge_lora): follows
           base_model_name_or_path up to 10 hops and merges every adapter it finds.

  EVAL     (validator/evaluation/pvp/materialize.py): scoring hands it a single-element
           base_chain -- [R2] for a round-3 miner -- and _resolve_chain expands that to the
           full lineage before merging.

This pins the parity. It is the regression test for the flattening the uploader used to do:
when every adapter declared the foundation instead of its parent, both sides collapsed to one
merge and every round before the last was silently dropped.
"""

import json
import os

import validator.evaluation.pvp.materialize as mat


FOUNDATION = "org/foundation"
R1 = "org/miner-round1"
R2 = "org/miner-round2"
DECLARED_BASE_OF = {R2: R1, R1: FOUNDATION}


def _install_fakes(monkeypatch, tmp_path):
    """Stand in for HF: repo -> adapter_config.json, plus download/merge recorders."""
    downloaded_as_base: list[str] = []
    merged_adapters: list[str] = []

    def fake_hf_hub_download(repo, filename, *a, **k):
        if repo not in DECLARED_BASE_OF:
            raise OSError(f"{repo} has no {filename}")
        cfg = tmp_path / f"{repo.replace('/', '_')}_{filename}"
        cfg.write_text(json.dumps({"base_model_name_or_path": DECLARED_BASE_OF[repo]}))
        return str(cfg)

    def fake_download_lora(repo, local_dir, *a, **k):
        os.makedirs(local_dir, exist_ok=True)
        # Stash which logical repo this dir represents so the merge can read it back.
        with open(os.path.join(local_dir, "_repo.txt"), "w") as f:
            f.write(repo)
        return local_dir

    def fake_download_model(repo, *a, **k):
        downloaded_as_base.append(repo)
        return f"/base/{repo.replace('/', '_')}"

    def fake_merge(base_path, lora_dir, output_dir="/tmp/merged_model", device=None):
        with open(os.path.join(lora_dir, "_repo.txt")) as f:
            merged_adapters.append(f.read())
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    monkeypatch.setattr(mat, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(mat, "_download_lora_with_retry", fake_download_lora)
    monkeypatch.setattr(mat, "_download_model_with_retry", fake_download_model)
    monkeypatch.setattr(mat, "_merge_base_and_lora", fake_merge)
    return downloaded_as_base, merged_adapters


def test_materialize_reconstructs_full_lineage(monkeypatch, tmp_path):
    """Round 3's base is foundation + R1 + R2 -- exactly what the trainer merged."""
    downloaded_as_base, merged_adapters = _install_fakes(monkeypatch, tmp_path)

    # scoring._get_continuation_base_chains always produces a SINGLE-element chain: [R2].
    # This is the only input eval gets; _resolve_chain has to recover R1 from it.
    mat.materialize_base_model(FOUNDATION, [R2], label="a")

    # 1. The merge root must be the real foundation (a loadable full model), never an adapter
    #    repo -- loading an adapter as a base crashes in prod.
    assert downloaded_as_base == [FOUNDATION], (
        f"eval loaded {downloaded_as_base} as the base model; the trainer chain-walks to {FOUNDATION!r}."
    )

    # 2. Both deltas must be applied, bottom-to-top, matching the trainer's merge order.
    assert merged_adapters == [R1, R2], (
        f"eval merged {merged_adapters}; the trainer merges {[R1, R2]} (foundation->R1->R2). "
        f"Dropping R1 serves the round-2 adapter on a base it was never trained against."
    )


def test_materialize_returns_foundation_untouched_for_full_finetune(monkeypatch, tmp_path):
    """A full-weight starting model has no lineage to walk."""
    downloaded_as_base, merged_adapters = _install_fakes(monkeypatch, tmp_path)

    result = mat.materialize_base_model(FOUNDATION, [], label="b")

    assert result == FOUNDATION
    assert downloaded_as_base == []
    assert merged_adapters == []
