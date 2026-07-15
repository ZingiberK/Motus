#!/usr/bin/env python3
"""Merge a PPO round payload back into a full deploy Motus checkpoint.

``ppo_update.py`` saves ``{action_expert, und_expert, ...}`` (the parameters it
trains). To roll into the next RL round, overlay those onto the base deploy
checkpoint and write a new deploy-format directory the RL server can load.

    python -m motus_rl.merge_round_ckpt \\
        --base deploy_ckpts/motus_finetune_50000 \\
        --ppo_payload rl_ckpts/round0_action.pt \\
        --out deploy_ckpts/motus_rl_round1
"""

import argparse
import shutil
from pathlib import Path

import torch


def _load_module(deploy_dir: Path) -> dict:
    f = deploy_dir / "mp_rank_00_model_states.pt"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found (expected deploy-format ckpt)")
    blob = torch.load(f, map_location="cpu")
    return blob["module"] if isinstance(blob, dict) and "module" in blob else blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base deploy ckpt dir")
    ap.add_argument("--ppo_payload", required=True, help="ppo_update out_ckpt .pt")
    ap.add_argument("--out", required=True, help="output deploy ckpt dir")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sd = _load_module(base)
    payload = torch.load(args.ppo_payload, map_location="cpu")

    n = 0
    for prefix in ("action_expert", "und_expert"):
        sub = payload.get(prefix, {})
        for k, v in sub.items():
            sd[f"{prefix}.{k}"] = v
            n += 1
    torch.save({"module": sd}, out / "mp_rank_00_model_states.pt")
    print(f"Merged {n} tensors from {args.ppo_payload} onto {args.base} -> {out}")

    cfg = base / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, out / "config.json")


if __name__ == "__main__":
    main()
