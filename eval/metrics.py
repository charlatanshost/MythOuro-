"""
Individual evaluation metrics for MythOuro.

Each public function follows the contract:
    metric(model, tokenizer, *, max_samples, n_loops, device, **kwargs) -> dict

Returned dicts always include:
    "name"          : str, the metric's identifier
    "samples"       : int, how many examples were actually evaluated
    "elapsed_s"     : float, wall-clock seconds
plus the metric-specific scalars (e.g. "ppl", "accuracy", "ece", ...).

All metrics:
    - run on CPU or GPU transparently
    - never crash on data-loading failure (return `"error": <str>` instead)
    - cap to `max_samples` so a smoke run completes in seconds

The model is assumed to be in eval mode; the harness handles `.eval()`.
"""

from __future__ import annotations

import math
import re
import time
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _safe_load_dataset(*args, **kwargs):
    """Best-effort wrapper around `datasets.load_dataset`.

    Returns the dataset on success, or `None` on any failure (network,
    HF auth, dataset config error). Callers should treat None as
    "metric skipped" rather than a hard error.
    """
    try:
        from datasets import load_dataset
        return load_dataset(*args, **kwargs)
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"eval.metrics: load_dataset failed ({exc})")
        return None


def _seq_logprobs(
    model,
    input_ids: torch.Tensor,
    n_loops: int,
) -> torch.Tensor:
    """
    Sum of log-probabilities the model assigns to `input_ids[:, 1:]`
    conditional on the prefix. Shape: (B,).

    Standard teacher-forced scoring used by PPL and ARC.
    """
    with torch.no_grad():
        logits, _ = model(input_ids, n_loops=n_loops)
    # Shift: predict token t+1 from logits[t]. We score positions 1..T-1.
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    # Gather the log-prob of the realised next token at each position.
    chosen = logprobs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    return chosen.sum(dim=1)                                  # (B,)


def _truncate_left(ids: list[int], max_len: int) -> list[int]:
    """Keep the LAST `max_len` tokens. Used so long prompts always
    leave room for the answer span."""
    return ids[-max_len:] if len(ids) > max_len else ids


# ---------------------------------------------------------------------------
# Perplexity (held-out FineWeb-Edu)
# ---------------------------------------------------------------------------


def perplexity(
    model,
    tokenizer,
    *,
    max_samples: int = 50,
    n_loops: Optional[int] = None,
    device: str = "cpu",
    seq_len: int = 512,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
) -> dict:
    """
    Standard token-level perplexity on a held-out streaming slice of
    FineWeb-Edu (configurable via `dataset_name` / `dataset_config`).

    Notes:
        - Streams the dataset to avoid local-disk download for a quick eval.
        - Packs consecutive documents into fixed-length `seq_len` chunks
          using the same scheme as `MixedDataset` in training_utils.
        - `n_loops` defaults to `model.cfg.max_loop_iters`.
    """
    name = "perplexity"
    t0 = time.perf_counter()
    n_loops = n_loops or model.cfg.max_loop_iters

    ds = _safe_load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    if ds is None:
        return {"name": name, "samples": 0, "elapsed_s": 0.0,
                "error": f"could not open {dataset_name}/{dataset_config}"}

    total_nll = 0.0
    total_tokens = 0
    n_chunks = 0

    buf: list[int] = []
    for sample in ds:
        if n_chunks >= max_samples:
            break
        text = sample.get("text") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text))
        # Drain the buffer into fixed-length chunks.
        while len(buf) >= seq_len + 1 and n_chunks < max_samples:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            ids = torch.tensor(
                [chunk[:-1]], dtype=torch.long, device=device,
            )
            tgt = torch.tensor(
                [chunk[1:]], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                logits, _ = model(ids, n_loops=n_loops)
            nll = F.cross_entropy(
                logits.view(-1, logits.shape[-1]).float(),
                tgt.view(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += tgt.numel()
            n_chunks += 1

    if total_tokens == 0:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "no tokens evaluated"}

    mean_nll = total_nll / total_tokens
    return {
        "name": name,
        "samples": n_chunks,
        "tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 50)),   # cap to avoid overflow on random init
        "elapsed_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# ARC-Challenge (cloze-style log-likelihood ranking)
# ---------------------------------------------------------------------------


def arc_challenge(
    model,
    tokenizer,
    *,
    max_samples: int = 100,
    n_loops: Optional[int] = None,
    device: str = "cpu",
    max_prompt_len: int = 256,
) -> dict:
    """
    ARC-Challenge accuracy via per-choice log-likelihood ranking.

    For each (question, [A, B, C, D]) tuple we compute
        score(c) = Σ_t log P(c_t | question, c_<t)
    and pick the argmax. Length-normalising the score by token count
    is a common variant; we keep raw sum to match the OG ARC eval.
    """
    name = "arc_challenge"
    t0 = time.perf_counter()
    n_loops = n_loops or model.cfg.max_loop_iters

    ds = _safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if ds is None:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "could not open allenai/ai2_arc"}

    correct = 0
    total = 0

    for sample in ds.select(range(min(max_samples, len(ds)))):
        question = sample.get("question") or ""
        choices = sample.get("choices") or {}
        texts = choices.get("text") or []
        labels = choices.get("label") or []
        gold = sample.get("answerKey")
        if not texts or not gold:
            continue

        q_ids = _truncate_left(tokenizer.encode(f"Question: {question}\nAnswer: "), max_prompt_len)
        best_idx, best_score = None, float("-inf")
        for i, choice_text in enumerate(texts):
            c_ids = tokenizer.encode(choice_text)
            full = q_ids + c_ids
            ids = torch.tensor([full], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids, n_loops=n_loops)
            # Score only the choice tokens (not the question prefix).
            shift = logits[:, len(q_ids) - 1 : len(full) - 1, :]
            logprobs = F.log_softmax(shift.float(), dim=-1)
            tgt = torch.tensor([c_ids], dtype=torch.long, device=device)
            score = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is not None and best_idx < len(labels) and labels[best_idx] == gold:
            correct += 1
        total += 1

    if total == 0:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "no samples evaluated"}

    return {
        "name": name,
        "samples": total,
        "accuracy": correct / total,
        "n_correct": correct,
        "random_baseline": 0.25,
        "elapsed_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# GSM8K (greedy generation + regex parse)
# ---------------------------------------------------------------------------


_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def gsm8k(
    model,
    tokenizer,
    *,
    max_samples: int = 50,
    n_loops: Optional[int] = None,
    device: str = "cpu",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """
    GSM8K final-answer accuracy via greedy generation + `#### N` regex.

    Notes:
        - `temperature=0.0` enables true argmax decoding (handled by the
          sampler's `top_k=1` shortcut).
        - We compare numeric value, not exact string, so "1234" and
          "1,234" both match if the parser strips commas (we don't —
          GSM8K gold answers don't use commas).
    """
    name = "gsm8k"
    t0 = time.perf_counter()
    n_loops = n_loops or model.cfg.max_loop_iters

    ds = _safe_load_dataset("openai/gsm8k", "main", split="test")
    if ds is None:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "could not open openai/gsm8k"}

    correct = 0
    total = 0

    for sample in ds.select(range(min(max_samples, len(ds)))):
        question = sample.get("question") or ""
        gold_text = sample.get("answer") or ""
        gold_match = _GSM8K_ANSWER_RE.search(gold_text)
        if not gold_match:
            continue
        gold = gold_match.group(1)

        prompt = f"Question: {question}\nAnswer: "
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device,
        )
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=max(temperature, 1e-3),    # generate() clamps
            top_k=1 if temperature <= 0 else 50,
        )
        gen_ids = out[0, ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids)
        pred_match = _GSM8K_ANSWER_RE.search(gen_text)
        pred = pred_match.group(1) if pred_match else None

        if pred is not None and pred == gold:
            correct += 1
        total += 1

    if total == 0:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "no samples evaluated"}

    return {
        "name": name,
        "samples": total,
        "accuracy": correct / total,
        "n_correct": correct,
        "elapsed_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Loop efficiency
# ---------------------------------------------------------------------------


def loop_efficiency(
    model,
    tokenizer,
    *,
    max_samples: int = 50,
    n_loops: Optional[int] = None,
    device: str = "cpu",
    seq_len: int = 256,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
) -> dict:
    """
    Average loop depth actually used / `n_loops` budgeted.

    Reads `model.recurrent.last_halt_step` after a forward pass — this
    tensor is populated by RecurrentBlock and gives the loop index at
    which each (B, T) position halted (or `n_loops` if it never halted).

    Below 1.0 means the model is exiting early on at least some tokens
    — the ACT gate is working. Equal to 1.0 means every position runs
    to max depth (gate has collapsed or model is uniformly uncertain).
    """
    name = "loop_efficiency"
    t0 = time.perf_counter()
    n_loops = n_loops or model.cfg.max_loop_iters

    ds = _safe_load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    if ds is None:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "could not open eval corpus"}

    total_positions = 0
    total_depth = 0
    halt_distribution = [0] * (n_loops + 1)
    n_chunks = 0

    buf: list[int] = []
    for sample in ds:
        if n_chunks >= max_samples:
            break
        text = sample.get("text") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text))
        while len(buf) >= seq_len + 1 and n_chunks < max_samples:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            ids = torch.tensor(
                [chunk[:-1]], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                model(ids, n_loops=n_loops)

            halt = getattr(model.recurrent, "last_halt_step", None)
            if halt is None:
                continue
            # Strip sink-token positions so we report depths only over
            # real content positions.
            sink_len = model.sink.n_tokens
            content_halt = halt[:, sink_len:].flatten()
            total_depth += int(content_halt.sum().item())
            total_positions += int(content_halt.numel())
            for v in content_halt.tolist():
                halt_distribution[min(v, n_loops)] += 1
            n_chunks += 1

    if total_positions == 0:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "no positions evaluated"}

    avg_depth = total_depth / total_positions
    return {
        "name": name,
        "samples": n_chunks,
        "positions": total_positions,
        "avg_halt_depth": avg_depth,
        "max_depth": n_loops,
        "efficiency": avg_depth / n_loops,         # 1.0 = always runs to max
        "halt_distribution": halt_distribution,
        "elapsed_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error (uncertainty head)
# ---------------------------------------------------------------------------


def expected_calibration_error(
    model,
    tokenizer,
    *,
    max_samples: int = 50,
    n_loops: Optional[int] = None,
    device: str = "cpu",
    seq_len: int = 256,
    n_bins: int = 10,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
) -> dict:
    """
    Expected Calibration Error of the UncertaintyHead.

    The UncertaintyHead is trained to predict P(model is wrong at this
    token). We collect (predicted_uncertainty, was_actually_wrong) pairs
    across a held-out stream, bin by predicted uncertainty, and compute
    a weighted mean absolute difference between bin-average confidence
    and bin-average error rate.

    ECE ∈ [0, 1]; lower is better. A perfectly calibrated head has 0.
    Untrained model typically lands near 0.4-0.5 because the head's
    sigmoid sits near 0.5 while accuracy is ≈ 1/vocab_size.
    """
    name = "expected_calibration_error"
    t0 = time.perf_counter()
    n_loops = n_loops or model.cfg.max_loop_iters

    ds = _safe_load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    if ds is None:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "could not open eval corpus"}

    preds: list[float] = []
    errors: list[float] = []
    n_chunks = 0

    buf: list[int] = []
    for sample in ds:
        if n_chunks >= max_samples:
            break
        text = sample.get("text") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text))
        while len(buf) >= seq_len + 1 and n_chunks < max_samples:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            ids = torch.tensor(
                [chunk[:-1]], dtype=torch.long, device=device,
            )
            tgt = torch.tensor(
                [chunk[1:]], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                logits, unc = model(ids, n_loops=n_loops)
            pred = logits.argmax(dim=-1)                    # (1, T)
            is_wrong = (pred != tgt).float()                # (1, T)
            preds.extend(unc.float().flatten().cpu().tolist())
            errors.extend(is_wrong.flatten().cpu().tolist())
            n_chunks += 1

    if not preds:
        return {"name": name, "samples": 0, "elapsed_s": time.perf_counter() - t0,
                "error": "no positions evaluated"}

    # Histogram by predicted uncertainty
    preds_t = torch.tensor(preds)
    errors_t = torch.tensor(errors)
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = preds_t.numel()
    bin_rows = []
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        in_bin = (preds_t >= lo) & (preds_t < hi if i < n_bins - 1 else preds_t <= hi)
        n_in = int(in_bin.sum().item())
        if n_in == 0:
            bin_rows.append({"lo": lo, "hi": hi, "count": 0,
                              "mean_pred": None, "mean_error": None})
            continue
        mean_pred = float(preds_t[in_bin].mean().item())
        mean_err  = float(errors_t[in_bin].mean().item())
        ece += (n_in / total) * abs(mean_pred - mean_err)
        bin_rows.append({
            "lo": lo, "hi": hi, "count": n_in,
            "mean_pred": mean_pred, "mean_error": mean_err,
        })

    return {
        "name": name,
        "samples": n_chunks,
        "positions": total,
        "ece": ece,
        "bins": bin_rows,
        "elapsed_s": time.perf_counter() - t0,
    }
