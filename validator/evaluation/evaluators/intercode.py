"""InterCode-Bash NL2Bash evaluator that runs inside a Basilica deployment.

Mirrors the contract of validator/evaluation/evaluators/environment.py:
  - Reads MODELS / ORIGINAL_MODEL / EVAL_SEED / ENV_EVAL_TEMPERATURE /
    DATASET_TYPE / SGLANG_* env vars set by docker_evaluation.py.
  - Launches SGLang locally for the candidate model and waits for /v1/models.
  - Writes the standard {repo: {"is_finetune": True, "eval_loss": avg}} payload
    to /aplp/evaluation_results.json so the Basilica result wrapper can return
    it unchanged.

The differences from eval_environment.py:
  - There is no external env-server. The InterCode filesystem layouts for the
    four NL2Bash variants are baked into the image at /intercode_fs/fs{1..4}.tar,
    and the dataset JSONs at /intercode_data/nl2bash_fs_{1..4}.json. We restore
    the relevant snapshot before each task instead of POSTing to an env-server.
  - Inference goes through SGLang's OpenAI-compatible endpoint via the `openai`
    client (replacing the Chutes-API path from scripts/intercode_eval.py).
  - Tasks run sequentially: the managed paths (/testbed, /system, /workspace,
    /backup) are global to the deployment, so parallel tasks would corrupt
    each other's state.

The original ReAct examples are adapted to the same tool-calling contract used
by PvP: every turn must call exactly one action tool instead of emitting
free-form `Action: execute[...]` text for a regex parser.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from huggingface_hub import snapshot_download

import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from core.models.dataset_models import EnvironmentDatasetType
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import FunctionSchema
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from core.pvp.chat import chat_completion
from core.pvp.sglang_parsers import tool_call_parser_for
from core.tokenizer_utils import ensure_chat_template
from core.tokenizer_utils import read_chat_template
from validator.evaluation.model_checks import check_for_lora
from validator.evaluation.model_checks import check_lora_has_added_tokens
from validator.evaluation.pvp.materialize import materialize_base_model
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EVAL_LOG_LEVEL = "INFO"
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_BASE_SEED = vcst.ENV_EVAL_DEFAULT_SEED
DEFAULT_TEMPERATURE = vcst.ENV_EVAL_TEMPERATURE

DEFAULT_SGLANG_TENSOR_PARALLEL_SIZE = "1"
DEFAULT_SGLANG_DTYPE = "float16"
DEFAULT_SGLANG_PORT = "30000"
DEFAULT_SGLANG_BASE_URL = "http://127.0.0.1:30000"
DEFAULT_SGLANG_HEALTH_PATH = "/v1/models"
DEFAULT_SGLANG_HEALTH_TIMEOUT_SECONDS = 1800
DEFAULT_SGLANG_EXTRA_CLI = vcst.SGLANG_ENV_EVAL_EXTRA_CLI
DEFAULT_FLASHINFER_WORKSPACE_MIN_BYTES = vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES
DEFAULT_INTERCODE_LOG_SGLANG = False

DEFAULT_INTERCODE_DATA_ROOT = Path("/intercode_data")
DEFAULT_INTERCODE_FS_ROOT = Path("/intercode_fs")

DEFAULT_ACTION_TIMEOUT_SECONDS = 30
DEFAULT_OBS_TRUNCATE_CHARS = 350
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOKENS_PER_CALL = 512
DEFAULT_PER_TASK_TIMEOUT_SECONDS = vcst.ENV_EVAL_TASK_TIMEOUT
DEFAULT_SESSION_TIMEOUT_SECONDS = vcst.ENV_EVAL_SESSION_TIMEOUT

DEFAULT_SCORING_MODE = "continuous"
VALID_SCORING_MODES = {"continuous", "binary"}

INTERCODE_EXECUTE_TOOL_NAME = "execute_bash"
INTERCODE_SUBMIT_TOOL_NAME = "submit"

ALL_MANAGED_PATHS = ("/testbed", "/system", "/workspace", "/backup")
PATHS_PER_FS: dict[int, tuple[str, ...]] = {
    1: ("/testbed",),
    2: ("/system",),
    3: ("/workspace", "/backup"),
    4: (),  # filesystem-agnostic
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# SGLang is chatty (per-batch progress lines) and floods Basilica logs. Default
# off; flip INTERCODE_LOG_SGLANG=1 to re-enable when debugging the inference server.
LOG_SGLANG_STDOUT = _env_bool("INTERCODE_LOG_SGLANG", DEFAULT_INTERCODE_LOG_SGLANG)
SCORING_MODE = os.getenv("INTERCODE_SCORING_MODE", DEFAULT_SCORING_MODE).strip().lower()
assert SCORING_MODE in VALID_SCORING_MODES, f"invalid INTERCODE_SCORING_MODE={SCORING_MODE!r}"


# ─────────────────────────────────────────────────────────────────────────────
# LoRA + SGLang setup
#
# This block is intentionally copy-pasted (with light edits) from
# validator/evaluation/evaluators/environment.py so that the two evaluators can
# evolve independently. If you change SGLang flags here, sync the env eval.
# ─────────────────────────────────────────────────────────────────────────────

def _download_model_with_retry(repo_id: str, max_retries: int = DEFAULT_DOWNLOAD_RETRIES) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "eval_setup download base model (attempt %s/%s): %s",
                attempt, max_retries, repo_id,
            )
            start = time.time()
            path = snapshot_download(repo_id, local_files_only=False)
            logger.info("eval_setup base model snapshot_download done in %.1fs -> %s", time.time() - start, path)
            return path
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                raise


def _download_lora_with_retry(repo_id: str, local_dir: str, max_retries: int = DEFAULT_DOWNLOAD_RETRIES) -> str:
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("eval_setup download LoRA (attempt %s/%s): %s -> %s", attempt, max_retries, repo_id, local_dir)
            start = time.time()
            snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            logger.info("eval_setup LoRA snapshot_download done in %.1fs", time.time() - start)
            return local_dir
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                raise


def _merge_base_and_lora(base_model_path: str, lora_dir: str, output_dir: str = "/tmp/merged_model") -> str:
    needs_install = (
        importlib.util.find_spec("peft") is None
        or importlib.util.find_spec("accelerate") is None
    )
    if needs_install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "peft", "accelerate"],
            check=True,
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    logger.info("eval_setup merge: start base=%s lora=%s out=%s", base_model_path, lora_dir, output_dir)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )

    base_vocab_size = base.get_input_embeddings().weight.shape[0]
    target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= base_vocab_size else base_tokenizer
    target_vocab_size = len(target_tokenizer)
    if target_vocab_size > base_vocab_size:
        base.resize_token_embeddings(target_vocab_size)

    model = PeftModel.from_pretrained(base, lora_dir)
    merged = model.merge_and_unload(safe_merge=False)
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    # Carry the adapter's chat template onto the saved tokenizer (base selection would drop it).
    ensure_chat_template(target_tokenizer, read_chat_template(lora_dir), base_tokenizer.chat_template)
    target_tokenizer.save_pretrained(output_dir)
    return output_dir


def _configure_logging() -> None:
    level_name = os.getenv("EVAL_LOG_LEVEL", DEFAULT_EVAL_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.addHandler(handler)
    logger.setLevel(level)
    # Silence the per-request "HTTP Request: POST .../v1/chat/completions" lines
    # emitted by the openai client's underlying httpx transport. At ReAct loop
    # rates these are pure noise — keep WARNING+ so real connection errors still surface.
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_environment_name() -> env_cst.EnvironmentName:
    dataset_type_raw = os.getenv("DATASET_TYPE", "{}")
    env_name = os.getenv("ENVIRONMENT_NAME")
    if not env_name:
        try:
            dataset_type = EnvironmentDatasetType.model_validate_json(dataset_type_raw)
            environment_names = dataset_type.environment_names or []
            env_name = environment_names[0] if environment_names else None
        except Exception:
            env_name = None
    env_name = getattr(env_name, "value", env_name)
    if not env_name:
        raise ValueError("Missing environment name. Set ENVIRONMENT_NAME or DATASET_TYPE.")
    if env_name != env_cst.EnvironmentName.INTERCODE.value:
        raise ValueError(f"eval_intercode invoked with environment_name={env_name!r}; expected 'intercode'")
    return env_cst.EnvironmentName.INTERCODE


def _build_sglang_command(model_path: str, seed: int, *, parser_model_id: str | None = None) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", DEFAULT_SGLANG_TENSOR_PARALLEL_SIZE)
    dtype = os.getenv("SGLANG_DTYPE", DEFAULT_SGLANG_DTYPE)
    port = os.getenv("SGLANG_PORT", DEFAULT_SGLANG_PORT)
    base = (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    parser = tool_call_parser_for(parser_model_id or model_path)
    if parser:
        base = f"{base} --tool-call-parser {parser}"
    extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or DEFAULT_SGLANG_EXTRA_CLI).strip()
    return f"{base} {extra}" if extra else base


def _start_process(command: str, name: str, *, capture_stdout: bool = True) -> subprocess.Popen:
    logger.info("Starting %s: %s", name, command)
    stdout = subprocess.PIPE if capture_stdout else subprocess.DEVNULL
    stderr = subprocess.STDOUT if capture_stdout else subprocess.DEVNULL
    return subprocess.Popen(
        command,
        shell=True,
        stdout=stdout,
        stderr=stderr,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )


def _stop_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("Stopping %s (pid=%s)", name, proc.pid)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("Failed to stop %s cleanly: %s", name, exc)


async def _wait_for_health(
    url: str,
    path: str,
    timeout_seconds: int,
    *,
    service_name: str = "service",
    log_interval_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_seconds
    started = time.time()
    last_log = started
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}{path}", timeout=aiohttp.ClientTimeout(total=8)) as response:
                    if response.status == 200:
                        logger.info("eval_setup %s healthy after %.1fs", service_name, time.time() - started)
                        return
            except Exception:
                pass
            now = time.time()
            if now - last_log >= log_interval_s:
                logger.info(
                    "eval_setup still waiting for %s: GET %s%s (elapsed=%.0fs / timeout=%ss)",
                    service_name, url, path, now - started, timeout_seconds,
                )
                last_log = now
            await asyncio.sleep(2)
    raise TimeoutError(f"{service_name} at {url}{path} did not become healthy within {timeout_seconds}s")


async def _stream_logs(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.stdout is None:
        return
    while True:
        if proc.poll() is not None and proc.stdout.closed:
            return
        line = await asyncio.to_thread(proc.stdout.readline)
        if not line:
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.2)
            continue
        logger.info("[%s] %s", name, line.rstrip())


# ─────────────────────────────────────────────────────────────────────────────
# NL2Bash dataset mapping
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(data_root: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for fs in (1, 2, 3, 4):
        path = data_root / f"nl2bash_fs_{fs}.json"
        out[fs] = json.loads(path.read_text())
    return out


def _compute_fs_ranges(data: dict[int, list[dict]]) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    cursor = 1
    for fs in (1, 2, 3, 4):
        n = len(data[fs])
        ranges.append((fs, cursor, cursor + n - 1))
        cursor += n
    return ranges


def _map_task_id(global_id: int, ranges: list[tuple[int, int, int]]) -> tuple[int, int]:
    for fs, start, end in ranges:
        if start <= global_id <= end:
            return fs, global_id - start
    total = ranges[-1][2]
    raise ValueError(f"task id {global_id} out of range; valid range is 1..{total}")


@dataclass(frozen=True)
class InterCodeAssets:
    data: dict[int, list[dict]]
    ranges: list[tuple[int, int, int]]
    snapshot_root: Path

    @property
    def total_tasks(self) -> int:
        return self.ranges[-1][2] if self.ranges else 0


def load_intercode_assets(
    data_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
) -> InterCodeAssets:
    data_path = (
        Path(data_root)
        if data_root is not None
        else Path(os.getenv("INTERCODE_DATA_ROOT", str(DEFAULT_INTERCODE_DATA_ROOT)))
    )
    snapshot_path = (
        Path(snapshot_root)
        if snapshot_root is not None
        else Path(os.getenv("INTERCODE_FS_ROOT", str(DEFAULT_INTERCODE_FS_ROOT)))
    )
    if not data_path.exists():
        raise RuntimeError(f"NL2Bash data not found at {data_path}; image may be misbuilt")
    if not snapshot_path.exists():
        raise RuntimeError(f"InterCode fs snapshots not found at {snapshot_path}; image may be misbuilt")

    data = _load_data(data_path)
    ranges = _compute_fs_ranges(data)
    return InterCodeAssets(data=data, ranges=ranges, snapshot_root=snapshot_path)


# ─────────────────────────────────────────────────────────────────────────────
# LocalBashEnv — docker-free replacement for intercode.envs.BashEnv
#
# The reward formula matches princeton-nlp/intercode's BashEnv.get_reward():
#   reward = 0.01
#          + 0.33 * (1 - erf(|diff_miss| + |diff_extra|))   # filesystem diff
#          + 0.33 * (same_changes / |diff_same|)            # content match
#          + 0.33 * tfidf_cosine(agent_obs, eval_obs)       # answer similarity
# diff_miss / diff_extra / diff_same are computed by comparing the per-file
# (md5, size) state of the agent's filesystem vs the gold's filesystem against
# the pristine snapshot, mimicking what `git status --short` produces upstream.
# ─────────────────────────────────────────────────────────────────────────────

# Scoring mode for LocalBashEnv._get_reward().
#   "continuous" — upstream InterCode formula, reward ∈ [0.01, 1.0]:
#                    0.01
#                  + 0.33 * (1 - erf(|diff_miss| + |diff_extra|))
#                  + 0.33 * (same_changes / |diff_same|)
#                  + 0.33 * tfidf_cosine(agent_obs, gold_obs)
#   "binary"     — 1.0 iff all three parts pass (no missing/extra fs diffs,
#                  every common change is byte-identical, agent_obs matches
#                  gold_obs exactly after whitespace normalization); else 0.0.
# Override at deploy time without rebuilding the image via the
# INTERCODE_SCORING_MODE env var.


class LocalBashEnv:
    """In-process, docker-free analogue of intercode.envs.BashEnv for NL2Bash."""

    def __init__(self, fs_version: int, entries: list[dict], snapshot_root: Path):
        self.fs_version = fs_version
        self.entries = entries
        self.managed_paths = PATHS_PER_FS[fs_version]
        self.snapshot_tar = snapshot_root / f"fs{fs_version}.tar"
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self.action_executed = False
        self.query: str | None = None
        self.gold: str | None = None
        self._snapshot_state: dict[str, tuple] | None = None
        self._agent_state: dict[str, tuple] | None = None
        self._eval_state: dict[str, tuple] | None = None

    def reset(self, index: int) -> str:
        record = self.entries[index]
        self.query = record["query"]
        self.gold = record.get("gold", "") or ""
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self._restore_fs()
        self._snapshot_state = self._capture_state()
        return self.query

    def _restore_fs(self) -> None:
        # Wipe ALL managed paths (not just this variant's) so leftovers from a
        # previous task can't leak into the current one — important for fs_4
        # which has no managed paths of its own.
        for p in ALL_MANAGED_PATHS:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        if not self.managed_paths or not self.snapshot_tar.exists():
            return
        try:
            subprocess.run(
                ["tar", "-xpf", str(self.snapshot_tar), "-C", "/"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"failed to restore fs_{self.fs_version} snapshot: "
                f"{exc.stderr.decode('utf-8', errors='replace')}"
            )

    def _capture_state(self) -> dict[str, tuple]:
        state: dict[str, tuple] = {}
        for root_path in self.managed_paths:
            if not os.path.exists(root_path):
                continue
            for cur, dirs, files in os.walk(root_path):
                for name in dirs:
                    full = os.path.join(cur, name)
                    try:
                        st = os.lstat(full)
                        state[full] = ("<DIR>", st.st_mode)
                    except OSError:
                        state[full] = ("<ERR>", 0)
                for name in files:
                    full = os.path.join(cur, name)
                    try:
                        if os.path.islink(full):
                            state[full] = ("<LINK>", os.readlink(full))
                        else:
                            h = hashlib.md5()
                            with open(full, "rb") as fh:
                                for chunk in iter(lambda: fh.read(65536), b""):
                                    h.update(chunk)
                            st = os.lstat(full)
                            state[full] = (h.hexdigest(), st.st_size)
                    except OSError:
                        state[full] = ("<ERR>", 0)
        return state

    @staticmethod
    def _simplify_path(current: str, changed: str) -> str:
        """Resolve a `cd` argument against the current workdir — matches BashEnv."""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""
        path: list[str] = []
        for seg in (current + "/" + changed).split("/"):
            if seg == "..":
                if path:
                    path.pop()
            elif seg and seg != ".":
                path.append(seg)
        return "/" + "/".join(path)

    def _exec_action(self, action: str) -> None:
        is_cd = action.startswith("cd")
        new_path: str | None = None
        if is_cd and "cd " in action:
            cd_arg = action[action.index("cd ") + 3:].strip()
            new_path = self._simplify_path(self.workdir, cd_arg)
            action = f"cd {new_path}"
        try:
            res = subprocess.run(
                ["/bin/bash", "-c", action],
                cwd="/" if is_cd else (self.workdir or "/"),
                capture_output=True,
                timeout=DEFAULT_ACTION_TIMEOUT_SECONDS,
            )
            stdout = res.stdout.decode("utf-8", errors="replace")
            stderr = res.stderr.decode("utf-8", errors="replace")
            self.observation = stdout + (stderr if not stdout else "")
            self.action_executed = res.returncode == 0
            if is_cd and self.action_executed and new_path is not None:
                self.workdir = new_path
        except subprocess.TimeoutExpired:
            self.observation = "Command timed out"
            self.action_executed = False
        except Exception:
            self.observation = "Malformed command"
            self.action_executed = False

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if action.startswith("submit"):
            reward, info = self._get_reward()
            info["action_executed"] = True
            return self.observation, reward, True, info
        self._exec_action(action)
        return self.observation, 0.0, False, {"action_executed": self.action_executed}

    def _get_reward(self) -> tuple[float, dict]:
        # Snapshot end-state of the agent's filesystem before running gold.
        self._agent_state = self._capture_state()

        # Run the gold command in a freshly-restored filesystem.
        self._restore_fs()
        gold_obs = ""
        corrupt_gold = False
        if self.gold:
            try:
                res = subprocess.run(
                    ["/bin/bash", "-c", self.gold],
                    cwd="/",
                    capture_output=True,
                    timeout=DEFAULT_ACTION_TIMEOUT_SECONDS,
                )
                gold_obs = (
                    res.stdout.decode("utf-8", errors="replace")
                    + res.stderr.decode("utf-8", errors="replace")
                )
            except Exception:
                corrupt_gold = True
        self.observation_eval = gold_obs
        self._eval_state = self._capture_state()

        snapshot = self._snapshot_state or {}
        agent_changed = self._changed_paths(snapshot, self._agent_state or {})
        eval_changed = self._changed_paths(snapshot, self._eval_state or {})

        diff_miss = eval_changed - agent_changed
        diff_extra = agent_changed - eval_changed
        diff_same = agent_changed & eval_changed

        common_changes_total = len(diff_same)
        common_changes_correct = sum(
            1
            for path in diff_same
            if (self._agent_state or {}).get(path) == (self._eval_state or {}).get(path)
        )
        agent_obs = self.observation or ""
        gold_obs = self.observation_eval or ""

        # Part 1: filesystem-state diff size, smoothed via erf (matches upstream).
        p1 = round(0.33 * (1 - math.erf(len(diff_miss) + len(diff_extra))), 2)

        # Part 2: of the paths both agent and gold modified, what fraction match?
        if common_changes_total:
            p2 = round(0.33 * (common_changes_correct / common_changes_total), 2)
        else:
            p2 = 0.33

        # Part 3: TF-IDF cosine on agent vs gold stdout; falls back to exact match.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([agent_obs, gold_obs])
            sim = float((tfidf * tfidf.T).toarray()[0][1])
        except Exception:
            sim = 1.0 if agent_obs == gold_obs else 0.0
        p3 = round(0.33 * sim, 2)

        continuous_reward = 0.01 + p1 + p2 + p3

        # Binary pass criteria for each component:
        #   - fs diff:    no missing or extra changes vs gold
        #   - content:    every commonly-changed path is byte-identical
        #   - answer:     stdout matches gold after whitespace normalization
        fs_pass = (len(diff_miss) == 0) and (len(diff_extra) == 0)
        content_pass = (common_changes_total == 0) or (common_changes_correct == common_changes_total)
        answer_pass = " ".join(agent_obs.split()) == " ".join(gold_obs.split())
        all_pass = fs_pass and content_pass and answer_pass
        binary_reward = 1.0 if all_pass else 0.0

        reward = binary_reward if SCORING_MODE == "binary" else continuous_reward
        info = {
            "scoring_mode": SCORING_MODE,
            "file_diff": p1,
            "file_changes": p2,
            "answer_similarity": p3,
            "continuous_reward": continuous_reward,
            "binary_reward": binary_reward,
            "fs_pass": fs_pass,
            "content_pass": content_pass,
            "answer_pass": answer_pass,
            "diff_miss": list(diff_miss),
            "diff_extra": list(diff_extra),
            "corrupt_gold": corrupt_gold,
        }
        return reward, info

    @staticmethod
    def _changed_paths(before: dict[str, tuple], after: dict[str, tuple]) -> set[str]:
        keys = set(before) | set(after)
        return {k for k in keys if before.get(k) != after.get(k)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool-calling prompt + loop
# ─────────────────────────────────────────────────────────────────────────────

INTERCODE_TOOL_SYSTEM_PROMPT = """Interact with a Bourne Shell system using Bash commands to answer a question.
You may reason briefly in normal assistant text, but every turn must call exactly one tool:
(1) execute_bash, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Do not write actions as plain text. Use the tools for every action.

Examples:

Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Turn 1 tool: execute_bash(command="xxd -p -l 16 /testbed/textfile7.txt")
Observation 1: /bin/bash: line 1: xxd: command not found
Turn 2 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt")
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Turn 3 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' '")
Observation 3: 546578742066696c6520372077697468
Turn 4 tool: submit()

Question: print disk usage in human readable format of files or folders in /workspace
Turn 1 tool: execute_bash(command="du /workspace")
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Turn 2 tool: execute_bash(command="du -h /workspace")
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Turn 3 tool: execute_bash(command="du -sh /workspace")
Observation 3: 100K\t/workspace
Turn 4 tool: submit()

Question: Count all the lines of all php files in the /testbed directory recursively
Turn 1 tool: execute_bash(command="find /testbed -name \"*.php\" | xargs wc -l")
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Turn 2 tool: execute_bash(command="find /testbed -name \"*.php\" -type f -exec cat {} + | wc -l")
Observation 2: 2
Turn 3 tool: submit()
"""


def _function_tool(name: str, description: str, parameters: dict) -> ToolSchema:
    return ToolSchema(function=FunctionSchema(name=name, description=description, parameters=parameters))


def build_intercode_action_tools() -> list[ToolSchema]:
    return [
        _function_tool(
            INTERCODE_EXECUTE_TOOL_NAME,
            "Execute one Bash command in the InterCode Bourne Shell environment.",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Bash command to execute.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        _function_tool(
            INTERCODE_SUBMIT_TOOL_NAME,
            "Submit the previous observation as the final answer and end the task.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
    ]


def _format_tool_history(history: list[str]) -> str:
    if not history:
        return "No commands have been executed yet."
    return "\n".join(history)


def _build_tool_messages(query: str, history: list[str], turn: int) -> list[ChatMessage]:
    user_prompt = (
        f"Question: {query}\n\n"
        f"Previous steps:\n{_format_tool_history(history)}\n\n"
        f"Turn {turn}: call exactly one tool now."
    )
    return [
        ChatMessage(role=ChatRole.SYSTEM, content=INTERCODE_TOOL_SYSTEM_PROMPT),
        ChatMessage(role=ChatRole.USER, content=user_prompt),
    ]


def _intercode_chat(
    client,
    model_name: str,
    temperature: float,
    messages: list[ChatMessage],
    tools: list[ToolSchema],
    max_tokens: int,
) -> ChatResult:
    config = ChatCompletionConfig(
        inference_model=model_name,
        base_url="",
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=0,
    )
    return chat_completion(client, config, messages, tools)


def _first_action_tool_call(result: ChatResult) -> ToolCall | None:
    for call in result.tool_calls or []:
        if call.name in {INTERCODE_EXECUTE_TOOL_NAME, INTERCODE_SUBMIT_TOOL_NAME}:
            return call
    return None


def _tool_call_for_history(call: ToolCall | None) -> str:
    if call is None:
        return "no valid tool call"
    if call.name == INTERCODE_SUBMIT_TOOL_NAME:
        return "submit()"
    command = call.arguments.get("command", "")
    return f"{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(command)})"


def _apply_intercode_tool_call(env: LocalBashEnv, call: ToolCall | None) -> tuple[str, float, bool]:
    if call is None:
        return (
            "Error executing query: no valid action tool call received; call execute_bash or submit.",
            0.0,
            False,
        )
    if call.name == INTERCODE_SUBMIT_TOOL_NAME:
        observation, reward, done, _info = env.step("submit")
        return observation, float(reward or 0.0), done

    command = call.arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return (
            "Error executing query: execute_bash requires a non-empty string command.",
            0.0,
            False,
        )
    observation, reward, done, _info = env.step(command)
    return observation, float(reward or 0.0), done


def _run_tool_episode(
    env: LocalBashEnv,
    query: str,
    client,
    model_name: str,
    temperature: float,
    max_turns: int,
    max_tokens_per_call: int,
) -> float:
    history: list[str] = []
    tools = build_intercode_action_tools()
    reward = 0.0
    done = False
    for turn in range(1, max_turns + 1):
        result = _intercode_chat(
            client,
            model_name,
            temperature,
            _build_tool_messages(query, history, turn),
            tools,
            max_tokens_per_call,
        )
        call = _first_action_tool_call(result)
        observation, reward, done_step = _apply_intercode_tool_call(env, call)

        if isinstance(observation, str) and len(observation) > DEFAULT_OBS_TRUNCATE_CHARS:
            observation = observation[:DEFAULT_OBS_TRUNCATE_CHARS]

        if result.content:
            history.append(f"Thought {turn}: {result.content}")
        history.append(f"Tool {turn}: {_tool_call_for_history(call)}")
        history.append(f"Observation {turn}: {observation}")

        if done_step:
            done = True
            break

    if not done:
        _, reward, _, _ = env.step("submit")

    return float(reward) if reward is not None else 0.0


async def run_intercode_task(
    task_id: int,
    assets: InterCodeAssets,
    client,
    model_name: str,
    temperature: float,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens_per_call: int = DEFAULT_MAX_TOKENS_PER_CALL,
    per_task_timeout: int = DEFAULT_PER_TASK_TIMEOUT_SECONDS,
    progress_label: str | None = None,
) -> float:
    fs_version, local_idx = _map_task_id(task_id, assets.ranges)
    env = LocalBashEnv(fs_version, assets.data[fs_version], assets.snapshot_root)
    query = env.reset(local_idx)
    label = f" {progress_label}" if progress_label else ""
    logger.info(
        "eval_progress%s task_global=%s fs=%s local=%s query=%r",
        label, task_id, fs_version, local_idx, query[:120],
    )
    return await asyncio.wait_for(
        asyncio.to_thread(
            _run_tool_episode,
            env, query, client, model_name, temperature,
            max_turns, max_tokens_per_call,
        ),
        timeout=per_task_timeout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

async def _run() -> None:
    sglang_proc: subprocess.Popen | None = None
    sglang_log_task: asyncio.Task | None = None
    try:
        logger.info(
            "eval_intercode: start pid=%s EVAL_LOG_LEVEL=%s",
            os.getpid(),
            os.getenv("EVAL_LOG_LEVEL", DEFAULT_EVAL_LOG_LEVEL),
        )

        models_raw = os.getenv("MODELS", "")
        model_repo = models_raw.split(",")[0].strip()
        if not model_repo:
            raise ValueError("MODELS is required and must contain a single repo")

        original_model = os.getenv("ORIGINAL_MODEL", model_repo)
        base_chain_raw = os.getenv("BASE_CHAIN", "")
        base_chain = json.loads(base_chain_raw) if base_chain_raw.strip() else []
        base_seed = int(os.getenv("EVAL_SEED", str(DEFAULT_BASE_SEED)))
        temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

        env_name = _parse_environment_name()
        env_config = env_cst.ENVIRONMENT_CONFIGS[env_name]
        task_id_min = env_config.task_id_min
        task_id_max = env_config.task_id_max
        _num_seeds_env = os.getenv("ENV_EVAL_NUM_SEEDS")
        if _num_seeds_env is not None and _num_seeds_env.strip() != "":
            num_seeds = int(_num_seeds_env)
        else:
            num_seeds = env_config.num_seeds

        seed_generator = random.Random(base_seed)
        task_ids_to_test = seed_generator.sample(range(task_id_min, task_id_max + 1), num_seeds)
        logger.info(
            "eval_setup config: env=%s num_seeds=%s task_id_range=(%s,%s) model_repo=%s original_model=%s "
            "eval_seed=%s temperature=%s base_chain=%s",
            env_name, num_seeds, task_id_min, task_id_max, model_repo, original_model, base_seed, temperature, base_chain,
        )

        # LoRA detection (matches eval_environment.py).
        is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
        should_merge_lora = False
        if is_lora:
            should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, model_repo, False)
        logger.info("eval_setup LoRA: is_lora=%s merge=%s base_chain=%s", is_lora, should_merge_lora, base_chain)

        inference_model_name = model_repo
        model_path_for_sglang = model_repo
        sglang_command = os.getenv("SGLANG_START_CMD")
        if not sglang_command:
            if is_lora and not should_merge_lora:
                model_path_for_sglang = await asyncio.to_thread(
                    materialize_base_model, original_model, base_chain, "cand"
                )
                lora_dir = "/lora/trained_lora"
                await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_dir)
                for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
                    try:
                        os.remove(model_file)
                    except Exception as exc:
                        logger.warning("Failed to remove %s: %s", model_file, exc)
                index_file = os.path.join(lora_dir, "model.safetensors.index.json")
                if os.path.exists(index_file):
                    try:
                        os.remove(index_file)
                    except Exception as exc:
                        logger.warning("Failed to remove index file: %s", exc)
                inference_model_name = f"{original_model}:trained_lora"
                sglang_command = (
                    _build_sglang_command(model_path_for_sglang, base_seed, parser_model_id=original_model)
                    + " --enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton"
                )
            elif is_lora and should_merge_lora:
                base_path = await asyncio.to_thread(
                    materialize_base_model, original_model, base_chain, "cand"
                )
                lora_temp_dir = "/tmp/lora/trained_lora"
                await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_temp_dir)
                model_path_for_sglang = await asyncio.to_thread(_merge_base_and_lora, base_path, lora_temp_dir)
                inference_model_name = model_repo
                sglang_command = _build_sglang_command(model_path_for_sglang, base_seed, parser_model_id=original_model)
            else:
                model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, model_repo)
                inference_model_name = model_repo
                sglang_command = _build_sglang_command(model_path_for_sglang, base_seed, parser_model_id=model_repo)

        sglang_health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", str(DEFAULT_SGLANG_HEALTH_TIMEOUT_SECONDS)))
        _min_ws = DEFAULT_FLASHINFER_WORKSPACE_MIN_BYTES
        try:
            _cur_ws = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
        except ValueError:
            _cur_ws = 0
        if _cur_ws < _min_ws:
            os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(_min_ws)

        logger.info("eval_setup SGLang command: %s", sglang_command)
        sglang_proc = _start_process(sglang_command, "sglang", capture_stdout=LOG_SGLANG_STDOUT)
        if LOG_SGLANG_STDOUT:
            sglang_log_task = asyncio.create_task(_stream_logs(sglang_proc, "sglang"))

        sglang_base_url = os.getenv("SGLANG_BASE_URL", DEFAULT_SGLANG_BASE_URL)
        await _wait_for_health(
            sglang_base_url,
            os.getenv("SGLANG_HEALTH_PATH", DEFAULT_SGLANG_HEALTH_PATH),
            sglang_health_timeout,
            service_name="SGLang",
        )

        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key="dummy", base_url=f"{sglang_base_url}/v1")

        # Load NL2Bash datasets + per-fs mapping.
        assets = load_intercode_assets()
        logger.info("eval_setup intercode: %s total tasks across fs_1..fs_4 (%s)", assets.total_tasks, assets.ranges)

        # Run tasks sequentially — managed paths are global per-deployment.
        rewards: list[float] = []
        max_turns = int(os.getenv("INTERCODE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
        max_tokens_per_call = int(os.getenv("INTERCODE_MAX_TOKENS_PER_CALL", str(DEFAULT_MAX_TOKENS_PER_CALL)))
        per_task_timeout = DEFAULT_PER_TASK_TIMEOUT_SECONDS
        session_deadline = time.monotonic() + DEFAULT_SESSION_TIMEOUT_SECONDS

        for idx, task_id in enumerate(task_ids_to_test):
            if time.monotonic() >= session_deadline:
                logger.warning(
                    "eval_progress: session timeout reached after %s/%s tasks; stopping early",
                    idx, len(task_ids_to_test),
                )
                break
            fs_version, _ = _map_task_id(task_id, assets.ranges)

            start_t = time.time()
            try:
                reward = await run_intercode_task(
                    task_id,
                    assets,
                    client,
                    inference_model_name,
                    temperature,
                    max_turns=max_turns,
                    max_tokens_per_call=max_tokens_per_call,
                    per_task_timeout=per_task_timeout,
                    progress_label=f"{idx + 1}/{len(task_ids_to_test)}",
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "eval_progress %s/%s task_global=%s timed out after %ss; score=0.0",
                    idx + 1, len(task_ids_to_test), task_id, per_task_timeout,
                )
                reward = 0.0
            except Exception as exc:
                logger.warning(
                    "eval_progress %s/%s task_global=%s failed: %s; score=0.0",
                    idx + 1, len(task_ids_to_test), task_id, exc,
                    exc_info=True,
                )
                reward = 0.0

            rewards.append(float(reward))
            logger.info(
                "eval_progress %s/%s done task_global=%s fs=%s reward=%.4f elapsed=%.1fs",
                idx + 1, len(task_ids_to_test), task_id, fs_version, reward, time.time() - start_t,
            )

        if rewards:
            avg = sum(rewards) / len(rewards)
        else:
            avg = 0.0
            logger.warning("eval_intercode: no completed tasks; writing avg=0.0")

        output = {model_repo: {"is_finetune": True, "eval_loss": avg}}
        result_path = Path(CONTAINER_EVAL_RESULTS_PATH)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output), encoding="utf-8")
        logger.info(
            "eval_intercode: wrote %s tasks=%s avg=%.6f",
            result_path, len(rewards), avg,
        )
    finally:
        _stop_process(sglang_proc, "sglang")
        if sglang_log_task:
            sglang_log_task.cancel()


def main() -> int:
    _configure_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("eval_intercode failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
