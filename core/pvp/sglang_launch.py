"""Shared SGLang launch-command base for the PvP eval container and model-prep.

Both contexts serve a model deterministically and append their own extras
(tool-call parser resolution and extra CLI flags differ per context), but the
base command — model path, host/port, parallelism, dtype, determinism — must
stay identical or the two paths drift apart flag by flag.
"""

import os
import shlex


TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"


def ensure_remote_code_disabled(command: str) -> str:
    """Reject SGLang commands that opt into executing model-repository code."""
    try:
        args = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Invalid SGLang command: {exc}") from exc

    if any(arg.split("=", maxsplit=1)[0].replace("_", "-") == TRUST_REMOTE_CODE_FLAG for arg in args):
        raise ValueError(f"SGLang {TRUST_REMOTE_CODE_FLAG} is forbidden for miner model evaluation")
    return command


def build_base_command(model_path: str, port: int | str, seed: int) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    return (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed} "
        "--log-level warning --decode-log-interval 10000"
    )
