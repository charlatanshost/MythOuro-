"""OPT-B loader: alignment, manifest guards, dose reporting — on synthetic shards."""
import sys, os, json, pathlib, tempfile, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pytest as _pt
_pt.importorskip("mythouro.logit_cache", reason="OPT-B not applied (optim/apply.sh B)")
from mythouro.logit_cache import TeacherLogitCache

L, K, V, N = 8, 4, 100, 20

def build(tmp, temperature=2.0, seq_len=L):
    d = os.path.join(tmp, "shard_00000"); os.makedirs(d)
    # token row i is filled with value i -> makes misalignment detectable
    toks = np.tile(np.arange(N, dtype=np.int32)[:, None], (1, seq_len + 1))
    idx = np.tile(np.arange(N, dtype=np.uint16)[:, None, None], (1, seq_len, K))
    np.save(f"{d}/tokens.npy", toks)
    np.save(f"{d}/topk_idx.npy", idx)
    np.save(f"{d}/topk_val.npy", np.zeros((N, seq_len, K), np.float16))
    np.save(f"{d}/tail_lse.npy",
            np.tile(np.arange(N, dtype=np.float32)[:, None], (1, seq_len)))
    json.dump({"teacher_id": "x", "seq_len": seq_len, "top_k": K,
               "temperature": temperature, "vocab_size": V, "rows": N,
               "tokens": N * seq_len, "shards": 1, "mean_topk_mass": 0.99},
              open(os.path.join(tmp, "manifest.json"), "w"))
    return tmp

def test_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        c = TeacherLogitCache(tmp, seq_len=L, temperature=2.0, batch_size=4)
        it = iter(c)
        for _ in range(5):
            x, y, ti, tv, tl = next(it)
            row = x[:, 0]                       # token value == row id
            assert torch.equal(ti[:, 0, 0], row), "topk_idx misaligned"
            assert torch.equal(tl[:, 0].long(), row), "tail_lse misaligned"
            assert x.shape == (4, L) and y.shape == (4, L)
            assert ti.shape == (4, L, K)
        print("  ok: tokens, topk_idx and tail_lse stay aligned across batches")

def test_temperature_mismatch_is_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, temperature=2.0)
        try:
            TeacherLogitCache(tmp, seq_len=L, temperature=1.0, batch_size=4)
        except ValueError as e:
            assert "not rescalable" in str(e)
            print("  ok: temperature mismatch is a hard error")
            return
    raise AssertionError("temperature guard missed")

def test_seqlen_mismatch_is_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, seq_len=L)
        try:
            TeacherLogitCache(tmp, seq_len=L + 1, temperature=2.0, batch_size=4)
        except ValueError as e:
            assert "cannot be resized" in str(e)
            print("  ok: seq_len mismatch is a hard error")
            return
    raise AssertionError("seq_len guard missed")

def test_dose_reporting():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        c = TeacherLogitCache(tmp, seq_len=L, temperature=2.0, batch_size=4)
        assert abs(c.epochs_for(steps=10, micro_per_step=1) - 2.0) < 1e-9
        print("  ok: epochs_for() reports re-read dose (10 steps x 4 / 20 = 2.0)")

for f in (test_alignment, test_temperature_mismatch_is_fatal,
          test_seqlen_mismatch_is_fatal, test_dose_reporting):
    f()
print("\n  OPT-B loader: all tests pass")
