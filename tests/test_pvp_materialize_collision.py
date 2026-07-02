"""Regression test exposing the /tmp path collision in materialize_base_model.

A PvP pair prepares BOTH miners in the same process (see pvp.__main__._prepare_model
called for model_a then model_b). When both are continuation miners, each calls
materialize_base_model with a single-element base_chain, so the enumerate index is
always 0 and both write to the SAME /tmp/base_chain_merged_0 directory. Preparing
model_b overwrites the base that model_a's PreparedModel.sglang_model_path already
points at, so model_a ends up served on model_b's reconstructed base.

This test fails on the current code and should pass once the merge output path is
made unique per miner/spec.
"""

import os

import validator.evaluation.pvp.materialize as mat


def test_two_continuation_miners_do_not_collide_on_tmp(monkeypatch):
    def fake_download_model(repo):
        return f"/tmp/foundation_{repo.replace('/', '_')}"

    def fake_download_lora(repo, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "marker.txt"), "w") as f:
            f.write(repo)
        return local_dir

    def fake_merge(base_path, lora_dir, output_dir="/tmp/merged_model", device=None):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(lora_dir, "marker.txt")) as f:
            adapter = f.read()
        with open(os.path.join(output_dir, "produced_by.txt"), "w") as f:
            f.write(adapter)
        return output_dir

    monkeypatch.setattr(mat, "_download_model_with_retry", fake_download_model)
    monkeypatch.setattr(mat, "_download_lora_with_retry", fake_download_lora)
    monkeypatch.setattr(mat, "_merge_base_and_lora", fake_merge)

    foundation = "org/foundation"
    # _prepare_model passes the model's label ("a"/"b") so scratch dirs stay distinct.
    path_a = mat.materialize_base_model(foundation, ["org/minerA-round1"], label="a")
    path_b = mat.materialize_base_model(foundation, ["org/minerB-round1"], label="b")

    # Distinct miners must get distinct served bases.
    assert path_a != path_b, "both continuation miners materialize to the same /tmp dir"

    # model_a must still be served the base built from ITS OWN adapter, not model_b's.
    with open(os.path.join(path_a, "produced_by.txt")) as f:
        produced_by_a = f.read()
    assert produced_by_a == "org/minerA-round1", (
        f"model_a's served base was overwritten by model_b (produced_by={produced_by_a})"
    )
