"""
OPT-B step 2 — read the precomputed teacher top-K cache during training.

Pairs with `tools/precompute_teacher_logits.py`. The cache stores the TOKENS
next to the logits, so a batch and its teacher answer cannot drift apart: this
loader always yields them together, and never re-tokenises anything.

The manifest is checked against the live training config on construction. A
temperature mismatch is a hard error, not a warning — `tail_lse` is
logsumexp(z/T) and is simply wrong at another T, which would corrupt the
distillation target silently for an entire multi-night run.
"""
from __future__ import annotations

import glob, json, os
import numpy as np
import torch
from loguru import logger


class TeacherLogitCache:
    def __init__(self, path: str, *, seq_len: int, temperature: float,
                 batch_size: int, device: str = "cpu", seed: int = 1234):
        man_path = os.path.join(path, "manifest.json")
        if not os.path.exists(man_path):
            raise FileNotFoundError(
                f"{man_path} missing — build it with "
                f"tools/precompute_teacher_logits.py"
            )
        self.man = json.load(open(man_path))
        if int(self.man["seq_len"]) != int(seq_len):
            raise ValueError(
                f"logit cache seq_len={self.man['seq_len']} but training "
                f"--seq-len={seq_len}. The cache windows ARE the batches; "
                "they cannot be resized."
            )
        if abs(float(self.man["temperature"]) - float(temperature)) > 1e-9:
            raise ValueError(
                f"logit cache was built at temperature "
                f"{self.man['temperature']} but training uses {temperature}. "
                "tail_lse = logsumexp(logits/T) is not rescalable — rebuild the "
                "cache at the training temperature, or the distillation target "
                "is silently wrong."
            )
        self.dirs = sorted(glob.glob(os.path.join(path, "shard_*")))
        if not self.dirs:
            raise FileNotFoundError(f"no shard_* directories under {path}")
        self.batch_size = batch_size
        self.device = device
        self.top_k = int(self.man["top_k"])
        self.rng = np.random.default_rng(seed)
        self._n = 0
        for d in self.dirs:
            self._n += int(np.load(os.path.join(d, "tokens.npy"),
                                   mmap_mode="r").shape[0])
        logger.info(
            f"TeacherLogitCache: {len(self.dirs)} shards, {self._n:,} rows, "
            f"K={self.top_k}, T={self.man['temperature']}, "
            f"seq_len={self.man['seq_len']}, "
            f"mean top-K mass {self.man.get('mean_topk_mass')}"
        )

    def __len__(self):
        return self._n

    def epochs_for(self, steps: int, micro_per_step: int) -> float:
        """Rows consumed / rows available — the dose number."""
        return steps * micro_per_step * self.batch_size / max(1, self._n)

    def __iter__(self):
        while True:
            for d in self.rng.permutation(len(self.dirs)):
                sd = self.dirs[d]
                tk = np.load(os.path.join(sd, "tokens.npy"), mmap_mode="r")
                ti = np.load(os.path.join(sd, "topk_idx.npy"), mmap_mode="r")
                tv = np.load(os.path.join(sd, "topk_val.npy"), mmap_mode="r")
                tl = np.load(os.path.join(sd, "tail_lse.npy"), mmap_mode="r")
                order = self.rng.permutation(tk.shape[0])
                for i in range(0, len(order) - self.batch_size + 1,
                               self.batch_size):
                    sel = np.sort(order[i: i + self.batch_size])
                    toks = torch.from_numpy(np.ascontiguousarray(tk[sel])).long()
                    yield (
                        toks[:, :-1].to(self.device),
                        toks[:, 1:].to(self.device),
                        torch.from_numpy(
                            np.ascontiguousarray(ti[sel])
                        ).long().to(self.device),
                        torch.from_numpy(
                            np.ascontiguousarray(tv[sel])
                        ).float().to(self.device),
                        torch.from_numpy(
                            np.ascontiguousarray(tl[sel])
                        ).float().to(self.device),
                    )
