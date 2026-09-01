#!/usr/bin/env python3
"""
Silence the experts added by growth, without retraining anything.

WHY. Both grown models doubled the L0 (no usable function) rate against the
278M base: 15% -> 30-31%, independently, across two lineages. The owner's
hypothesis for that: with topk=4 of 48, roughly half of every token's experts
are now the UNDERTRAINED new ones (|down| at 0.40 and 0.25 of parent amplitude),
so each token gets a weaker FFN output. Growth would then look harmful without
being fundamentally wrong -- it is simply not finished.

That is testable for the price of one eval. Setting `router_bias[E_src:]` to the
-100 sentinel makes those experts unselectable in `topk(logits + router_bias)`,
so the model becomes functionally 24-expert again while KEEPING whatever the
original 24 learned during the grown run.

  L0 falls back toward the base's 15%  -> the new experts ARE the degradation.
                                          Growth is unfinished, not wrong, and
                                          the question becomes whether they can
                                          ever be finished (see the twinning
                                          result: measured asymptotic).
  L0 stays near 30%                    -> the grown TRAINING degraded the
                                          original experts too; the new experts
                                          are not the whole story.
  L0 falls BELOW 15%                   -> the extra steps helped the originals
                                          and the new experts were masking it.

    python3 -u -m tools.mask_new_experts --src <ckpt> --dst <ckpt> [--n-src 24]
"""
from __future__ import annotations
import argparse, os
import torch
from loguru import logger


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-src", type=int, default=None,
                    help="how many experts are ORIGINAL (default: from "
                         "growth_metadata, else half)")
    ap.add_argument("--sentinel", type=float, default=-100.0)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    gm = (ck.get("extra") or {}).get("growth_metadata") or {}
    n_total = None
    n_masked = 0
    for k, v in ck["model"].items():
        if not k.endswith("router_bias"):
            continue
        n_total = v.shape[0]
        e_src = a.n_src or int(gm.get("source_n_experts") or n_total // 2)
        if e_src >= n_total:
            raise SystemExit(f"{k}: n_src {e_src} >= n_experts {n_total}")
        before = v.clone()
        v[e_src:] = a.sentinel
        n_masked = n_total - e_src
        logger.info(
            f"{k}: masked experts {e_src}..{n_total-1} "
            f"(was [{before[e_src:].min():+.2f}, {before[e_src:].max():+.2f}])"
        )
    if n_total is None:
        raise SystemExit("no router_bias found — not an MoE checkpoint")

    ck.setdefault("extra", {})["masked_new_experts"] = {
        "source_path": os.path.abspath(a.src),
        "n_experts": n_total, "n_masked": n_masked, "sentinel": a.sentinel,
    }
    # The sentinel must not be decayed away by the training loop, so drop the
    # growth metadata: this checkpoint is for EVALUATION, not resumption.
    ck["extra"].pop("growth_metadata", None)
    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    torch.save(ck, a.dst + ".tmp"); os.replace(a.dst + ".tmp", a.dst)
    logger.success(f"wrote {a.dst} — {n_masked} experts silenced. EVAL ONLY: "
                   "growth_metadata was removed so a resume cannot decay the "
                   "sentinel back off.")


if __name__ == "__main__":
    main()
