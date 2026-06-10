"""Shared SGLang launch-command base for the PvP eval container and model-prep.

Both contexts serve a model deterministically and append their own extras
(tool-call parser resolution and extra CLI flags differ per context), but the
base command — model path, host/port, parallelism, dtype, determinism — must
stay identical or the two paths drift apart flag by flag.
"""

import os


def build_base_command(model_path: str, port: int | str, seed: int) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    return (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
