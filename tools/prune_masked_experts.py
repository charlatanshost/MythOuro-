#!/usr/bin/env python3
"""
Delete the sentinel-masked experts, turning a masked 48-expert model back into a
real 24-expert one. EXACTLY function-preserving.

WHY IT IS EXACT. `tools/mask_new_experts.py` sets `router_bias[24:] = -100`, so
`topk(logits + router_bias, k=4)` can never select experts 24-47 — the largest
content logit spread measured on these checkpoints is ~3.5, against a 100-point
sentinel. Those experts contribute exactly zero to every token. Removing them,
their router rows, and their bias entries therefore changes no output.

WHY BOTHER. The masked model is the best on record (2026-09-01: beats the 278M
base and the regressed mathcode leg on prose, with perfect rank separation), but
it carries 24 experts that:
  * two controlled experiments say will never differentiate (initialisation and
    load balancing both ruled out; the curve is asymptotic at cos ~0.89),
  * cost 34% more per checkpoint (4.5 GB vs 3.35 GB), and
  * make every future comparison against 24-expert history awkward.

Pruning gives a genuine `mythouro_distill_tiny`-shaped 278M model that keeps the
8,696 steps of gains.

    python3 -u -m tools.prune_masked_experts --src <masked.pt> --dst <out.pt>
"""
from __future__ import annotations
import argparse, os, copy
import torch
from loguru import logger

SENTINEL_FLOOR = -50.0     # anything below this is a masked expert


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["model"]
    bkeys = [k for k in sd if k.endswith("router_bias")]
    if not bkeys:
        raise SystemExit("no router_bias — not an MoE checkpoint")

    keep_n = None
    for bk in bkeys:
        b = sd[bk]
        keep = (b > SENTINEL_FLOOR).nonzero().flatten()
        if keep.numel() == b.numel():
            raise SystemExit(f"{bk}: nothing is masked — run mask_new_experts first")
        if keep_n is None:
            keep_n = keep.numel()
        elif keep.numel() != keep_n:
            raise SystemExit("layers disagree on how many experts survive")
        if not torch.equal(keep, torch.arange(keep.numel())):
            raise SystemExit(
                f"{bk}: surviving experts are not a prefix {keep.tolist()[:8]}... "
                "— this tool only handles the grow-then-mask layout"
            )
        prefix = bk[: -len("router_bias")]
        sd[bk] = b[:keep_n].clone()
        rw = f"{prefix}router.weight"
        sd[rw] = sd[rw][:keep_n].clone()
        logger.info(f"{prefix}: {b.numel()} -> {keep_n} experts")

    # drop the expert modules themselves
    dropped = 0
    for k in list(sd):
        if ".routed_experts." not in k:
            continue
        idx = int(k.split(".routed_experts.")[1].split(".")[0])
        if idx >= keep_n:
            del sd[k]; dropped += 1
    logger.info(f"removed {dropped} expert weight tensors")

    cfg = ck.get("cfg")
    if cfg is not None:
        cfg = copy.deepcopy(cfg); cfg.n_experts = keep_n; ck["cfg"] = cfg
    if ck.get("cfg_dict"):
        ck["cfg_dict"] = dict(ck["cfg_dict"]); ck["cfg_dict"]["n_experts"] = keep_n

    # Reset the step counter. The pruned model starts a NEW lineage, and
    # grow.py sets step=0 on promotion for the same reason. Leaving the source
    # step (e.g. 8696) while naming the file step_0000000.pt makes the trainer
    # resume at 8696, compare against --total-steps, and exit with "training
    # complete" after ZERO steps — reporting success. Cost a pilot window on
    # 2026-09-01.
    ck["step"] = 0
    ck["optimizer"] = {}          # shapes changed — a stale optimizer would break
    extra = dict(ck.get("extra") or {})
    extra.pop("growth_metadata", None)
    extra.pop("masked_new_experts", None)
    extra["pruned_from"] = {"source": os.path.abspath(a.src), "n_experts": keep_n}
    ck["extra"] = extra

    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    torch.save(ck, a.dst + ".tmp"); os.replace(a.dst + ".tmp", a.dst)
    logger.success(f"wrote {a.dst} — {keep_n} experts, optimizer cleared")


if __name__ == "__main__":
    main()
