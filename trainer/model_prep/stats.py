"""
Comprehensive stats collection: dataset analysis, weight structure, and training dynamics.
Per-type stats for instruct, DPO, GRPO, and chat tasks.
"""

import math
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.models.model_prep_models import BaselineStats
from core.models.model_prep_models import DpoBaselineStats
from core.models.model_prep_models import DpoDatasetStats
from core.models.model_prep_models import DpoTrainingDynamics
from core.models.model_prep_models import GrpoBaselineStats
from core.models.model_prep_models import GrpoDatasetStats
from core.models.model_prep_models import GrpoTrainingDynamics
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import LayerGroupWeightStats
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import ThroughputStats
from core.models.model_prep_models import WeightStats
from core.models.task_models import TaskType


BPB_REFERENCE_MODEL = "gpt2"

# Token-length stats are computed on a random sample and scaled to the full
# record count; model forward passes use the same sample.
LENGTH_SAMPLE_SIZE = 2_000
BPB_SAMPLE_SIZE = 200
LENGTH_SAMPLE_SEED = 42

# Throughput probe settings.
THROUGHPUT_TARGET_BATCH_TOKENS = 8_192
THROUGHPUT_WARMUP_STEPS = 1
THROUGHPUT_TIMED_STEPS = 3
THROUGHPUT_MIN_SEQ_LEN = 256
THROUGHPUT_MAX_SEQ_LEN = 2_048


# --- Text extraction per task type ---

def _extract_instruct_texts(records: list[dict]) -> list[tuple[str, str]]:
    """Returns list of (prompt, completion) tuples."""
    results = []
    for r in records:
        prompt_parts = []
        if r.get("system"):
            prompt_parts.append(str(r["system"]))
        if r.get("instruct"):
            prompt_parts.append(str(r["instruct"]))
        if r.get("input"):
            prompt_parts.append(str(r["input"]))
        prompt = " ".join(prompt_parts) or " ".join(str(v) for v in r.values())
        completion = str(r.get("output", ""))
        results.append((prompt, completion))
    return results


def _extract_dpo_texts(records: list[dict]) -> list[tuple[str, str, str]]:
    """Returns list of (prompt, chosen, rejected) tuples."""
    results = []
    for r in records:
        prompt = str(r.get("prompt", ""))
        chosen = str(r.get("chosen", ""))
        rejected = str(r.get("rejected", ""))
        results.append((prompt, chosen, rejected))
    return results


def _extract_grpo_texts(records: list[dict]) -> list[str]:
    """Returns list of prompt strings."""
    return [str(r.get("prompt", " ".join(str(v) for v in r.values()))) for r in records]


def _extract_chat_texts(records: list[dict]) -> list[tuple[str, str]]:
    """Returns list of (prompt, completion) tuples from conversation turns."""
    results = []
    for r in records:
        convos = r.get("conversations", [])
        if isinstance(convos, str):
            results.append((convos, ""))
            continue
        user_parts = []
        assistant_parts = []
        for turn in convos:
            role = str(turn.get("from", turn.get("role", "")))
            content = str(turn.get("value", turn.get("content", "")))
            if role in ("user", "human"):
                user_parts.append(content)
            elif role in ("assistant", "gpt", "bot"):
                assistant_parts.append(content)
        results.append((" ".join(user_parts), " ".join(assistant_parts)))
    return results


# --- Tokenization helper ---

class SimpleTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.encodings = []
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# --- Seq length distribution helper ---

def _make_seq_dist(lengths: list[int]) -> SeqLengthDistribution:
    arr = np.array(lengths) if lengths else np.array([0])
    return SeqLengthDistribution(
        mean=float(np.mean(arr)),
        p50=int(np.percentile(arr, 50)),
        p95=int(np.percentile(arr, 95)),
        p99=int(np.percentile(arr, 99)),
        max=int(np.max(arr)),
    )


def _token_lengths(texts: list[str], tokenizer) -> list[int]:
    return [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]


def _sample_for_lengths(records: list[dict]) -> tuple[list[dict], float]:
    """Random sample for token-length stats plus the sample-to-full scale factor."""
    if len(records) <= LENGTH_SAMPLE_SIZE:
        return records, 1.0
    sample = random.Random(LENGTH_SAMPLE_SEED).sample(records, LENGTH_SAMPLE_SIZE)
    return sample, len(records) / LENGTH_SAMPLE_SIZE


def measure_training_throughput(model, device, seq_len: int, vocab_size: int) -> ThroughputStats | None:
    """Time fwd+bwd steps on synthetic batches to estimate training tokens/sec."""
    seq_len = int(min(max(seq_len, THROUGHPUT_MIN_SEQ_LEN), THROUGHPUT_MAX_SEQ_LEN))
    micro_batch = max(1, THROUGHPUT_TARGET_BATCH_TOKENS // seq_len)
    was_training = model.training
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def _step(batch_size: int) -> None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)

    result = None
    try:
        while micro_batch >= 1:
            try:
                for _ in range(THROUGHPUT_WARMUP_STEPS):
                    _step(micro_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(THROUGHPUT_TIMED_STEPS):
                    _step(micro_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                tokens_per_sec = micro_batch * seq_len * THROUGHPUT_TIMED_STEPS / max(elapsed, 1e-6)
                result = ThroughputStats(
                    tokens_per_sec=tokens_per_sec,
                    seq_len=seq_len,
                    micro_batch_size=micro_batch,
                    n_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                )
                print(
                    f"[stats] Throughput probe: {tokens_per_sec:.0f} tok/s "
                    f"(seq_len={seq_len}, micro_batch={micro_batch})",
                    flush=True,
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if micro_batch == 1:
                    print("[stats] Throughput probe OOM at micro_batch=1, skipping", flush=True)
                    break
                micro_batch //= 2
                print(f"[stats] Throughput probe OOM, retrying with micro_batch={micro_batch}", flush=True)
    except Exception as exc:
        print(f"[stats] Throughput probe failed: {exc}", flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if not was_training:
            model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _count_unique_tokens(texts: list[str], tokenizer) -> int:
    unique: set[int] = set()
    for t in texts:
        unique.update(tokenizer(t, truncation=False)["input_ids"])
    return len(unique)


# --- Layer type classification ---

LAYER_TYPE_PATTERNS = {
    "attention_qkv": [
        r"\.q_proj\.", r"\.k_proj\.", r"\.v_proj\.", r"\.c_attn\.",
        r"\.query\.", r"\.key\.", r"\.value\.", r"\.qkv\.", r"\.Wqkv\.",
        r"\.query_key_value\.",
    ],
    "attention_output": [
        r"\.o_proj\.", r"\.out_proj\.", r"\.attn\.c_proj\.",
        r"self_attn.*\.dense\.", r"self_attention.*\.dense\.",
    ],
    "ffn_up": [
        r"\.up_proj\.", r"\.gate_proj\.", r"\.c_fc\.", r"\.fc1\.",
        r"\.wi\.", r"\.dense_h_to_4h\.", r"\.gate\.",
        r"\.w1\.", r"\.w3\.",
    ],
    "ffn_down": [
        r"\.down_proj\.", r"\.mlp\.c_proj\.", r"\.fc2\.",
        r"\.wo\.", r"\.dense_4h_to_h\.", r"\.w2\.",
    ],
    "embedding": [
        r"embed_tokens\.", r"\.wte\.", r"word_embeddings\.", r"embed_in\.",
    ],
    "unembedding": [
        r"lm_head\.", r"embed_out\.",
    ],
    "layer_norm": [
        r"layernorm", r"layer_norm", r"\.ln_", r"\.norm\.", r"rmsnorm",
        r"\.ln_f\.", r"final_layer_norm",
    ],
}


def classify_layer(name: str) -> str:
    name_lower = name.lower()
    for group, patterns in LAYER_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return group
    return "other"


# --- Shared computations ---

def _compute_near_duplicate_rate(texts: list[str], num_perm: int = 128, threshold: float = 0.5) -> float:
    try:
        from datasketch import MinHash
        from datasketch import MinHashLSH
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = []
        for i, text in enumerate(texts):
            m = MinHash(num_perm=num_perm)
            for word in text.lower().split():
                m.update(word.encode("utf-8"))
            minhashes.append(m)
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass
        dup_count = sum(1 for i, m in enumerate(minhashes) if len(lsh.query(m)) > 1)
        return dup_count / max(len(texts), 1)
    except ImportError:
        print("Warning: datasketch not installed, skipping near-duplicate detection", flush=True)
        return float("nan")


def _compute_bits_per_byte(texts: list[str]) -> float:
    """Compute bits-per-byte using GPT-2 reference model.

    Always runs on CPU to avoid competing for GPU VRAM with the main model.
    GPT-2 is small enough that CPU inference is fine here.
    """
    t0 = time.time()
    texts = [t for t in texts if t.strip()]
    if not texts:
        return 0.0
    ref_model = AutoModelForCausalLM.from_pretrained(BPB_REFERENCE_MODEL)
    ref_tokenizer = AutoTokenizer.from_pretrained(BPB_REFERENCE_MODEL)
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model.eval()
    total_loss_nats = 0.0
    total_bytes = 0
    with torch.no_grad():
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            enc = ref_tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            outputs = ref_model(**enc, labels=enc["input_ids"])
            n_predicted_tokens = enc["input_ids"].shape[1] - 1
            total_loss_nats += outputs.loss.item() * max(n_predicted_tokens, 1)
    del ref_model
    result = (total_loss_nats / math.log(2)) / max(total_bytes, 1)
    print(f"[stats] BPB done in {time.time() - t0:.1f}s ({len(texts)} texts, {total_bytes} bytes, bpb={result:.4f})", flush=True)
    return result


def compute_weight_stats(model) -> WeightStats:
    t0 = time.time()
    group_names: dict[str, list[str]] = defaultdict(list)
    for name, _ in model.named_parameters():
        group_names[classify_layer(name)].append(name)
    by_group = {}
    for group, names in group_names.items():
        sum_sq = 0.0
        group_max_abs = 0.0
        numel = 0
        for n in names:
            w = model.get_parameter(n).data.detach().cpu().float()
            sum_sq += (w ** 2).sum().item()
            group_max_abs = max(group_max_abs, w.abs().max().item())
            numel += w.numel()
            del w
        by_group[group] = LayerGroupWeightStats(
            weight_rms=float(math.sqrt(sum_sq / max(numel, 1))),
            weight_norm=float(math.sqrt(sum_sq)),
            max_abs=float(group_max_abs),
        )
    print(f"[stats] Weight stats done in {time.time() - t0:.1f}s ({len(by_group)} groups)", flush=True)
    return WeightStats(by_group=by_group)


def _get_model_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        return torch.device("cuda:0")
    return next(model.parameters()).device


def _compute_base_training_dynamics(
    model, tokenizer, texts: list[str], device, max_length: int = 512,
) -> dict:
    """Compute shared training dynamics (loss, activations, entropy) with forward passes only."""
    t_start = time.time()
    dataset = SimpleTextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[stats] Training dynamics: {len(dataset)} samples, device={device}", flush=True)

    # Forward hooks for activation RMS — registered before eval loop so we
    # collect across all batches in eval mode (no dropout/batchnorm noise).
    activation_rms_accum: dict[str, list[float]] = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                activation_rms_accum[name].append(torch.sqrt(torch.mean(out.float() ** 2)).item())
        return hook

    for name, module in model.named_modules():
        if not list(module.children()) and any(p.requires_grad for p in module.parameters(recurse=False)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Init loss + entropy (eval mode — also collects activation RMS via hooks)
    model.eval()
    batch_losses: list[float] = []
    batch_entropies: list[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            batch_losses.append(outputs.loss.cpu().item())
            logits = outputs.logits.float()
            probs = F.softmax(logits, dim=-1)
            per_position_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            mask_f = attention_mask.to(device=logits.device).float()
            entropy = (per_position_entropy * mask_f).sum().item() / max(mask_f.sum().item(), 1.0)
            del logits, probs, per_position_entropy
            if not math.isnan(entropy) and not math.isinf(entropy):
                batch_entropies.append(entropy)

    loss_arr = np.array(batch_losses) if batch_losses else np.array([0.0])
    entropy_arr = np.array(batch_entropies) if batch_entropies else np.array([0.0])
    init_loss = float(np.mean(loss_arr))
    init_loss_std = float(np.std(loss_arr))
    output_entropy = float(np.mean(entropy_arr))
    output_entropy_std = float(np.std(entropy_arr))

    for h in hooks:
        h.remove()
    activation_rms = {n: float(np.mean(v)) for n, v in activation_rms_accum.items()}
    print(
        f"[stats] Eval loop done in {time.time() - t_start:.1f}s "
        f"(loss={init_loss:.4f}, {len(batch_losses)} batches)",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[stats] Training dynamics total: {time.time() - t_start:.1f}s", flush=True)
    return {
        "init_loss": init_loss,
        "init_loss_std": init_loss_std,
        "activation_rms": activation_rms,
        "output_entropy": output_entropy if not math.isnan(output_entropy) else 0.0,
        "output_entropy_std": output_entropy_std if not math.isnan(output_entropy_std) else 0.0,
    }


def _tokenize_prompt_completion(
    tokenizer, prompt: str, completion: str, max_length: int = 512,
) -> tuple[torch.Tensor, int]:
    """Tokenize prompt and completion separately, concatenate IDs.

    Returns (input_ids [1, seq_len], prompt_token_count).
    This avoids BPE boundary artifacts from concatenating strings before tokenizing.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
    combined = (prompt_ids + completion_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(combined))
    return torch.tensor([combined], dtype=torch.long), prompt_len


def _compute_masked_loss(
    model, tokenizer, prompt_texts: list[str], completion_texts: list[str], device, max_length: int = 512
) -> float:
    """Compute loss masked to completion tokens only."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for prompt, completion in zip(prompt_texts, completion_texts):
            if not completion.strip():
                continue
            input_ids, prompt_len = _tokenize_prompt_completion(tokenizer, prompt, completion, max_length)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)

            labels = input_ids.clone()
            labels[0, :prompt_len] = -100  # mask prompt tokens

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs.loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                n += 1
    return total_loss / max(n, 1)


def _compute_log_probs(
    model, tokenizer, prompts: list[str], completions: list[str], device, max_length: int = 512
) -> list[float]:
    """Compute mean log-prob of completions given prompts."""
    model.eval()
    log_probs = []
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            if not completion.strip():
                continue
            input_ids, prompt_len = _tokenize_prompt_completion(tokenizer, prompt, completion, max_length)
            input_ids = input_ids.to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab)

            # Get log-probs for completion tokens
            completion_logits = logits[prompt_len - 1:-1]  # shifted
            completion_targets = input_ids[0, prompt_len:]
            if completion_targets.shape[0] == 0:
                continue

            log_p = F.log_softmax(completion_logits, dim=-1)
            token_log_probs = log_p.gather(1, completion_targets.unsqueeze(1)).squeeze(1)
            log_probs.append(token_log_probs.mean().item())
    return log_probs


# --- Per-type compute functions ---

def _compute_instruct_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
    text_extractor=None,
) -> InstructBaselineStats:
    t_total = time.time()
    extractor = text_extractor or _extract_instruct_texts

    # Token-length stats on a random sample, scaled to the full record count.
    length_records, length_scale = _sample_for_lengths(records)
    length_texts = extractor(length_records)
    length_prompts, length_completions = zip(*length_texts) if length_texts else ([], [])
    prompt_lengths = _token_lengths(list(length_prompts), tokenizer)
    completion_lengths = _token_lengths(list(length_completions), tokenizer)

    # Random subset for model forward passes (length_records is already seeded).
    texts = length_texts[:max_samples]
    prompts, completions = zip(*texts) if texts else ([], [])
    all_texts = [p + " " + c for p, c in texts]
    print(f"[stats] Instruct: {len(records)} records, {len(texts)} samples for forward passes", flush=True)

    t0 = time.time()
    unique_tokens = _count_unique_tokens(all_texts, tokenizer)
    vocab_size = len(tokenizer)
    dataset_stats = InstructDatasetStats(
        total_tokens=int((sum(prompt_lengths) + sum(completion_lengths)) * length_scale),
        num_records=len(records),
        seq_length_distribution=_make_seq_dist([p + c for p, c in zip(prompt_lengths, completion_lengths)]),
        near_duplicate_rate=_compute_near_duplicate_rate(all_texts),
        bits_per_byte=_compute_bits_per_byte(list(completions)[:BPB_SAMPLE_SIZE]),
        vocab_size=vocab_size,
        unique_tokens_in_data=unique_tokens,
        vocab_coverage_ratio=unique_tokens / max(vocab_size, 1),
        prompt_tokens=int(sum(prompt_lengths) * length_scale),
        completion_tokens=int(sum(completion_lengths) * length_scale),
        completion_length_distribution=_make_seq_dist(completion_lengths),
    )
    print(f"[stats] Dataset stats done in {time.time() - t0:.1f}s", flush=True)

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, all_texts, device)

    t0 = time.time()
    masked_loss = _compute_masked_loss(model, tokenizer, list(prompts), list(completions), device)
    print(f"[stats] Masked loss done in {time.time() - t0:.1f}s (loss={masked_loss:.4f})", flush=True)

    training = InstructTrainingDynamics(**base_dynamics, masked_completion_loss=masked_loss)

    t0 = time.time()
    weights = compute_weight_stats(model)

    print(f"[stats] Instruct stats total: {time.time() - t_total:.1f}s", flush=True)
    return InstructBaselineStats(
        dataset=dataset_stats,
        weights=weights,
        training=training,
    )


def _compute_dpo_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
) -> DpoBaselineStats:
    # Token-length stats on a random sample, scaled to the full record count.
    length_records, length_scale = _sample_for_lengths(records)
    length_texts = _extract_dpo_texts(length_records)
    all_p, all_c, all_r = zip(*length_texts) if length_texts else ([], [], [])
    prompt_lengths = _token_lengths(list(all_p), tokenizer)
    chosen_lengths = _token_lengths(list(all_c), tokenizer)
    rejected_lengths = _token_lengths(list(all_r), tokenizer)

    ratios = [c / r if r > 0 else 1.0 for c, r in zip(chosen_lengths, rejected_lengths)]

    # Random subset for model forward passes (length sample is already seeded).
    texts = length_texts[:max_samples]
    prompts, chosens, rejecteds = zip(*texts) if texts else ([], [], [])
    all_texts = [p + " " + c for p, c in zip(prompts, chosens)]

    all_dpo_texts = list(prompts) + list(chosens) + list(rejecteds)
    unique_tokens = _count_unique_tokens(all_dpo_texts, tokenizer)
    vocab_size = len(tokenizer)
    dataset_stats = DpoDatasetStats(
        total_tokens=int((sum(prompt_lengths) + sum(chosen_lengths) + sum(rejected_lengths)) * length_scale),
        num_records=len(records),
        seq_length_distribution=_make_seq_dist([p + c for p, c in zip(prompt_lengths, chosen_lengths)]),
        near_duplicate_rate=_compute_near_duplicate_rate(list(prompts)),
        bits_per_byte=_compute_bits_per_byte(list(prompts)[:BPB_SAMPLE_SIZE]),
        vocab_size=vocab_size,
        unique_tokens_in_data=unique_tokens,
        vocab_coverage_ratio=unique_tokens / max(vocab_size, 1),
        prompt_tokens=int(sum(prompt_lengths) * length_scale),
        chosen_tokens=int(sum(chosen_lengths) * length_scale),
        rejected_tokens=int(sum(rejected_lengths) * length_scale),
        chosen_length_distribution=_make_seq_dist(chosen_lengths),
        rejected_length_distribution=_make_seq_dist(rejected_lengths),
        chosen_rejected_length_ratio=float(np.median(ratios)),
    )

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, all_texts, device)

    chosen_log_probs = _compute_log_probs(model, tokenizer, list(prompts), list(chosens), device)
    rejected_log_probs = _compute_log_probs(model, tokenizer, list(prompts), list(rejecteds), device)

    mean_chosen = float(np.mean(chosen_log_probs)) if chosen_log_probs else 0.0
    mean_rejected = float(np.mean(rejected_log_probs)) if rejected_log_probs else 0.0

    training = DpoTrainingDynamics(
        **base_dynamics,
        ref_log_prob_chosen=mean_chosen,
        ref_log_prob_rejected=mean_rejected,
        implicit_reward_gap=mean_chosen - mean_rejected,
    )

    return DpoBaselineStats(
        dataset=dataset_stats,
        weights=compute_weight_stats(model),
        training=training,
    )


def _compute_grpo_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
    reward_functions=None,
) -> GrpoBaselineStats:
    # Token-length stats on a random sample, scaled to the full record count.
    length_records, length_scale = _sample_for_lengths(records)
    length_prompts = _extract_grpo_texts(length_records)
    prompt_lengths = _token_lengths(length_prompts, tokenizer)

    # Random subset for model forward passes (length sample is already seeded).
    prompts = length_prompts[:max_samples]

    unique_tokens = _count_unique_tokens(prompts, tokenizer)
    vocab_size = len(tokenizer)
    dataset_stats = GrpoDatasetStats(
        total_tokens=int(sum(prompt_lengths) * length_scale),
        num_records=len(records),
        seq_length_distribution=_make_seq_dist(prompt_lengths),
        near_duplicate_rate=_compute_near_duplicate_rate(prompts),
        bits_per_byte=_compute_bits_per_byte(prompts[:BPB_SAMPLE_SIZE]),
        vocab_size=vocab_size,
        unique_tokens_in_data=unique_tokens,
        vocab_coverage_ratio=unique_tokens / max(vocab_size, 1),
        prompt_tokens=int(sum(prompt_lengths) * length_scale),
        prompt_length_distribution=_make_seq_dist(prompt_lengths),
    )

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, prompts, device)

    # Compute baseline reward scores
    reward_scores: dict[str, float] = {}
    if reward_functions:
        completions = _generate_completions(model, tokenizer, prompts[:10], device)
        for rf in reward_functions:
            if isinstance(rf, dict):
                func_code = rf.get("reward_func")
            else:
                func_code = rf.reward_func if hasattr(rf, "reward_func") else str(rf)
            # Extract function name from source for clean keys
            name_match = re.match(r"def\s+(\w+)", func_code.strip()) if func_code else None
            fallback_name = name_match.group(1) if name_match else func_code[:30]
            try:
                namespace: dict = {}
                exec(func_code, namespace)  # noqa: S102
                callables = [
                    (k, v) for k, v in namespace.items()
                    if callable(v) and k != "__builtins__"
                ]
                if not callables:
                    print(f"Warning: reward function code defined no callables: {func_code[:80]}", flush=True)
                    reward_scores[fallback_name] = 0.0
                    continue
                func_name, func = callables[0]
                scores = func(completions)
                reward_scores[func_name] = float(np.mean(scores))
            except Exception as exc:
                print(f"Warning: reward function {fallback_name} failed: {exc}", flush=True)
                reward_scores[fallback_name] = 0.0

    training = GrpoTrainingDynamics(**base_dynamics, baseline_reward_scores=reward_scores)

    return GrpoBaselineStats(
        dataset=dataset_stats,
        weights=compute_weight_stats(model),
        training=training,
    )


def _generate_completions(model, tokenizer, prompts: list[str], device, max_new_tokens: int = 50) -> list[str]:
    """Generate short completions from the base model."""
    model.eval()
    completions = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        generated = tokenizer.decode(output_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(generated)
    return completions


# --- Main entry point for text tasks ---

def compute_text_stats(
    model,
    tokenizer,
    data_records: list[dict],
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK,
    max_samples: int = LENGTH_SAMPLE_SIZE,
    reward_functions=None,
) -> BaselineStats:
    """Compute stats for text-based tasks (instruct, DPO, GRPO, chat)."""
    device = str(_get_model_device(model))

    if task_type == TaskType.CHATTASK:
        print("Computing chat stats...", flush=True)
        stats = _compute_instruct_stats(model, tokenizer, data_records, device, max_samples, text_extractor=_extract_chat_texts)
    elif task_type == TaskType.DPOTASK:
        print("Computing DPO stats...", flush=True)
        stats = _compute_dpo_stats(model, tokenizer, data_records, device, max_samples)
    elif task_type == TaskType.GRPOTASK:
        print("Computing GRPO stats...", flush=True)
        stats = _compute_grpo_stats(model, tokenizer, data_records, device, max_samples, reward_functions)
    else:
        print(f"Computing instruct stats (task_type={task_type})...", flush=True)
        stats = _compute_instruct_stats(model, tokenizer, data_records, device, max_samples)

    seq_dist = stats.dataset.seq_length_distribution
    packed_len = max(seq_dist.p95, 2 * seq_dist.p50)
    stats.throughput = measure_training_throughput(model, device, packed_len, model.config.vocab_size)
    return stats
