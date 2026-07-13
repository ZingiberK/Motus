"""Offline PPO update for WRM (one round) — PPO + supervised World Model.

Replaces the self-written Flow-GRPO learner (``grpo_update.py``) with the mature
RLinf/πRL-style recipe (see ``WRM_RL_PLAN_PPO.md``). Two objectives, one model,
gradient-isolated:

  * **Action (corrector)** — PPO on the Flow-SDE denoiser. Per-chunk advantage
    = (discounted return-to-go) − V(s)  [GAE λ=1, i.e. MC return + value baseline],
    reward = pure terminal success. ratio-clip + value-clip + multi-epoch. Only
    the action expert + und (shared) + value head receive these gradients; WAN is
    detached from the policy gradient (root cause of the two GRPO collapses).
  * **Video (WAN / WM)** — supervised flow-matching against the *observed* future
    frames returned by the simulator (now frequently failing futures). Identical
    to the SFT video pipeline; only WAN + und (shared) receive these gradients.

Per micro-sample we run two passes toggling ``requires_grad`` (video pass: WAN+und;
action pass: action+value+und), accumulate both grads, then one optimizer step.

Run (motus env, 1 GPU):
    cd /mnt/data14/yyg/Motus
    /mnt/data14/ccy/pip_packs/miniconda3/envs/motus/bin/python -m wrm_rl.ppo_update \
        --tasks stack_blocks_two --rl_root /mnt/data14/yyg/wrm_rl_runs/ppo_v1 --round 0 \
        --ckpt runs/wrm_und_zero_v1/wrm_und_step040000.pt \
        --out_ckpt /mnt/data14/yyg/wrm_rl_runs/ppo_v1/ckpts/round1.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.wrm_und import WRMUnd, WRMUndConfig  # noqa: E402
from wrm_rl.reward import episode_returns, paired_vla_returns  # noqa: E402

logger = logging.getLogger("ppo_update")


# ----------------------------------------------------------------- I/O helpers
def load_trace(path: str) -> dict:
    return torch.load(path, map_location="cpu")


_LANG_CACHE: dict[str, torch.Tensor] = {}


def load_lang(tr: dict) -> torch.Tensor:
    """Per-episode T5 embedding [1, 512, 4096] (tiny LRU-ish cache)."""
    p = tr["meta"]["lang_path"]
    t = _LANG_CACHE.get(p)
    if t is None:
        t = torch.load(p, map_location="cpu").float().unsqueeze(0)
        if len(_LANG_CACHE) > 64:
            _LANG_CACHE.clear()
        _LANG_CACHE[p] = t
    return t


def compute_logp(model, tr, lang, video_mode, use_ckpt):
    return model.action_logprob_from_trace(
        state=tr["state"].unsqueeze(0),
        a_vla=tr["a_vla_real"].unsqueeze(0),
        language_embeddings=lang,
        und_feats=tr["und_feats"].unsqueeze(0),
        action_latents=tr["action_latents"].float(),
        video_latents=tr["video_latents"].float(),
        timesteps=tr["timesteps"],
        eta=float(tr["eta"]),
        video_mode=tr.get("video_mode", video_mode),
        use_checkpoint=use_ckpt,
    )  # [1]


def compute_value(model, tr):
    return model.value(
        tr["state"].unsqueeze(0), tr["a_vla_real"].unsqueeze(0), tr["und_feats"].unsqueeze(0)
    )  # [1]


def compute_video_loss(model, tr, future_path, lang):
    fut = np.load(future_path)["frames"]                    # [T, H, W, 3] uint8
    vf = torch.from_numpy(fut).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0  # [1,T,C,H,W]
    ff = tr["first_frame"].float().unsqueeze(0)             # [1, C, H, W] in [0,1]
    if tr.get("vla_zero"):
        chunk = tr["a_vla_real"].shape[0]
        delta_exec = torch.zeros(1, chunk, tr["a_vla_real"].shape[-1], device=model.device)
    else:
        # executed residual (real space); action_latents[-1] is the final normalized latent
        delta_exec = model.unnormalize_delta(tr["action_latents"][-1].float().to(model.device))  # [1, chunk, 14]
    return model.video_supervised_loss(
        first_frame=ff, video_frames=vf,
        state=tr["state"].unsqueeze(0), a_vla=tr["a_vla_real"].unsqueeze(0),
        delta_exec=delta_exec, language_embeddings=lang, und_feats=tr["und_feats"].unsqueeze(0),
    )


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def build_model(ckpt_path: str, wan_dir: str, delta_stats, device="cuda") -> WRMUnd:
    cfg = WRMUndConfig(
        wan_checkpoint_path=wan_dir, wan_config_path=wan_dir,
        vae_path=str(Path(wan_dir) / "Wan2.2_VAE.pth"),
        num_layers=30, action_dim=14, action_state_dim=14, action_chunk_size=50,
        num_video_frames=8, video_height=384, video_width=320, batch_size=1,
        wan_finetune_mode="full", und_expert_hidden_size=512, vlm_adapter_input_dim=2048,
        delta_stats_path=delta_stats,
    )
    model = WRMUnd(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    trainable = ckpt.get("trainable_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(trainable, strict=False)
    logger.info("LOAD %s", json.dumps({
        "ckpt": ckpt_path, "tensors": len(trainable),
        "missing": len(missing), "unexpected": len(unexpected),
    }))
    return model


def build_samples(round_dir: Path, task: str, gamma: float, max_chunks_per_ep: int,
                  adv_mode: str = "paired_vla"):
    """Read episodes, build PPO + WM-only samples.

    Returns (ppo_samples, wm_samples, stats). PPO samples carry trace_path + ret/adv;
    WM-only samples (vla_zero arm) carry slim wm traces with delta_exec=0.
    """
    ep_files = sorted((round_dir / "episodes").glob(f"{task}_seed*_ep*.json"))
    vla_succ: dict[tuple, bool] = {}
    for f in ep_files:
        e = json.loads(f.read_text())
        if e.get("mode", "wrm") == "vla_zero":
            key = (task, int(e["seed"]), int(e.get("pair_id", e["ep"] // 2)))
            vla_succ[key] = bool(e["success"])

    ppo_samples: list[dict] = []
    wm_samples: list[dict] = []
    n_eps = n_succ = n_fut = n_wrm = n_vla = 0
    for f in ep_files:
        e = json.loads(f.read_text())
        mode = e.get("mode", "wrm")
        pair_id = int(e.get("pair_id", e["ep"] // 2))
        n_eps += 1
        n_succ += int(e["success"])
        fps = e.get("future_paths", [])
        if mode == "vla_zero":
            n_vla += 1
            wps = e.get("wm_trace_paths", e.get("trace_paths", []))
            idxs = [i for i in range(len(wps)) if wps[i]]
            if max_chunks_per_ep > 0 and len(idxs) > max_chunks_per_ep:
                idxs = sorted(random.sample(idxs, max_chunks_per_ep))
            for i in idxs:
                fp = fps[i] if i < len(fps) else ""
                n_fut += int(bool(fp))
                wm_samples.append({
                    "task": task, "trace_path": wps[i], "future_path": fp, "ppo": False,
                })
            continue

        n_wrm += 1
        tps = e.get("trace_paths", [])
        T = len(tps)
        key = (task, int(e["seed"]), pair_id)
        if adv_mode == "paired_vla":
            baseline = float(vla_succ.get(key, 0.0))
            rets = paired_vla_returns(bool(e["success"]), bool(baseline), T, gamma)
        else:
            rets = episode_returns(bool(e["success"]), T, gamma)
        idxs = [i for i in range(T) if tps[i]]
        if max_chunks_per_ep > 0 and len(idxs) > max_chunks_per_ep:
            idxs = sorted(random.sample(idxs, max_chunks_per_ep))
        for i in idxs:
            fp = fps[i] if i < len(fps) else ""
            n_fut += int(bool(fp))
            ppo_samples.append({
                "task": task, "trace_path": tps[i], "future_path": fp,
                "ret": float(rets[i]), "ppo": True,
            })

    stats = {
        "n_episodes": n_eps, "n_wrm_episodes": n_wrm, "n_vla_zero_episodes": n_vla,
        "n_success": n_succ,
        "success_rate": round(n_succ / max(n_eps, 1), 4),
        "n_chunk_samples": len(ppo_samples), "n_wm_samples": len(wm_samples),
        "n_with_future": n_fut,
    }
    return ppo_samples, wm_samples, stats


def _set_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def set_pass(mode: str, wan_params, action_params, value_params, und_params, freeze_wan: bool):
    """Toggle requires_grad for gradient isolation (explicit param lists only).

    video pass  -> WAN(+und) trainable, action/value frozen
    action pass -> action+value(+und) trainable, WAN frozen (no policy grad)
    und (shared hub) trains in both passes. VAE / anything not in these lists
    stays exactly as-is (never accidentally enabled).
    """
    _set_grad(wan_params, (mode == "video") and (not freeze_wan))
    _set_grad(action_params, mode == "action")
    _set_grad(value_params, mode == "action")
    _set_grad(und_params, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="one or more tasks updated jointly")
    ap.add_argument("--rl_root", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="theta_k (policy that generated this round)")
    ap.add_argument("--out_ckpt", required=True, help="where to save theta_{k+1}")
    ap.add_argument("--wan_dir", default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    ap.add_argument("--delta_stats", default=None, help="None => use buffers baked in ckpt")
    # PPO
    ap.add_argument("--gamma", type=float, default=0.99, help="per-chunk discount")
    ap.add_argument("--clip_eps", type=float, default=0.2, help="PPO ratio clip")
    ap.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip")
    ap.add_argument("--c_value", type=float, default=0.0,
                    help="value loss weight. 0 => no critic (advantage from group baseline). "
                         "The learned critic is COLD at round0 (value_head is a fresh head "
                         "absent from the SFT ckpt) so V(s)~0 => adv~return>=0 => no negative "
                         "signal and saturated tasks get uniformly reinforced (regression). "
                         "Group baseline fixes this without a warm critic.")
    ap.add_argument("--group_adv", type=int, default=1,
                    help="1 => advantage = return - per-task mean return (RLinf/Flow-GRPO "
                         "group-relative baseline). Gives contrastive +/- signal (failure "
                         "chunks negative, success positive) and ~0 on saturated tasks. "
                         "0 => legacy critic baseline (return - V(s)).")
    ap.add_argument("--group_adv_std", type=int, default=0,
                    help="1 => also divide grouped advantage by per-task std. OFF by default: "
                         "on near-saturated tasks std is tiny (only gamma-position jitter) and "
                         "dividing would spuriously amplify noise. Mean-subtraction alone keeps "
                         "the natural ~[-0.5,0.5] scale.")
    ap.add_argument("--adv_mode", default="paired_vla", choices=["paired_vla", "group"],
                    help="paired_vla: A = discounted(success_wrm - success_vla_paired); "
                         "group: legacy per-task mean baseline (requires --group_adv 1).")
    ap.add_argument("--saturated_tasks", default="",
                    help="JSON listing tasks to skip in learner (rollout skip is orchestrator-side).")
    ap.add_argument("--video_beta", type=float, default=0.5, help="supervised WM loss weight")
    ap.add_argument("--kl_beta", type=float, default=0.0, help="trust-region penalty on logratio^2")
    ap.add_argument("--norm_adv", action="store_true", help="scale advantages by their std (no mean shift)")
    # optimization
    ap.add_argument("--lr", type=float, default=1e-5, help="Action+Und+Value lr")
    ap.add_argument("--wan_lr", type=float, default=1e-5, help="WAN lr (supervised only)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=8, help="chunks per optimizer step (effective)")
    ap.add_argument("--epochs", type=int, default=2, help="passes over this round's data (>=2 so clip engages)")
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--target_kl", type=float, default=0.5,
                    help="stop remaining PPO epochs once a step's mean KL exceeds this "
                         "(standard PPO trust-region guard; 0=off). At lr=1e-6 epoch0 is "
                         "rock-stable; drift accumulates over epochs, so this caps it.")
    ap.add_argument("--adv_clip", type=float, default=4.0, help="clamp advantages to +/- this")
    ap.add_argument("--collapse_margin", type=float, default=0.2,
                    help="if this round's policy success < best_seen - margin, keep best & STOP")
    ap.add_argument("--video_mode", default="denoise", choices=["denoise", "skip"])
    ap.add_argument("--freeze_wan", action="store_true", help="smoke: freeze WAN, skip WM pass")
    ap.add_argument("--wan_lora", action="store_true", help="train WAN via LoRA (supervised)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--no_checkpoint", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--max_samples", type=int, default=0, help="global cap on chunk samples (0=all)")
    ap.add_argument("--max_chunks_per_ep", type=int, default=6, help="keep <= this many chunks/episode")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="WRM_WAM")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_id", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- DDP (data-parallel learner) ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
    is_main = (rank == 0)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_name = f"learner_round{args.round}.log" if is_main else f"learner_round{args.round}_rank{rank}.log"
    setup_logger(Path(args.rl_root) / log_name)
    if not is_main:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.info("PPO_START rank=%d/%d %s", rank, world_size, json.dumps(vars(args)))

    wb = None
    if args.wandb and is_main:
        try:
            import wandb as wb
            rid = args.wandb_id or Path(args.rl_root).name
            wb.init(project=args.wandb_project, entity=args.wandb_entity, id=rid, name=rid,
                    resume="allow", config=vars(args))
        except Exception as e:
            logger.warning("wandb init failed (%s); continuing without it", e)
            wb = None
    gstep = args.round * 100000

    def _finish():
        if wb is not None:
            wb.finish()
        if ddp:
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    # ---- data ----
    saturated: set[str] = set()
    if args.saturated_tasks:
        sat_path = Path(args.saturated_tasks)
        if sat_path.exists():
            saturated = set(json.loads(sat_path.read_text()).get("tasks", []))
            logger.info("SATURATED skip %d tasks: %s", len(saturated), sorted(saturated))

    ppo_samples: list[dict] = []
    wm_samples: list[dict] = []
    stats = {"per_task": {}}
    for task in args.tasks:
        if task in saturated:
            stats["per_task"][task] = {
                "n_episodes": 0, "n_success": 0, "success_rate": 0.0,
                "n_chunk_samples": 0, "n_wm_samples": 0, "skipped_saturated": True,
            }
            continue
        round_dir = Path(args.rl_root) / task / f"round{args.round}"
        s_ppo, s_wm, st_t = build_samples(
            round_dir, task, args.gamma, args.max_chunks_per_ep, adv_mode=args.adv_mode,
        )
        ppo_samples.extend(s_ppo)
        wm_samples.extend(s_wm)
        stats["per_task"][task] = st_t
    stats["n_chunk_samples"] = len(ppo_samples)
    stats["n_wm_samples"] = len(wm_samples)
    logger.info("ROUND_STATS %s", json.dumps(stats))

    # ---- keep-best / collapse early-stop (same as GRPO learner) ----
    agg_eps0 = sum(t["n_episodes"] for t in stats["per_task"].values())
    agg_succ0 = sum(t["n_success"] for t in stats["per_task"].values())
    cur_success = agg_succ0 / max(agg_eps0, 1)
    best_path = Path(args.rl_root) / "best.json"
    stop_path = Path(args.rl_root) / "STOP"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    logger.info("KEEP_BEST cur_success=%.4f best=%s", cur_success,
                json.dumps(best) if best else "none")
    if best is not None and (best["success"] - cur_success) > args.collapse_margin:
        logger.warning("COLLAPSE_STOP cur=%.4f best=%.4f -> keep best ckpt %s",
                       cur_success, best["success"], best["ckpt"])
        if is_main:
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.load(best["ckpt"], map_location="cpu"), args.out_ckpt)
            stop_path.write_text(json.dumps(
                {"reason": "collapse", "round": args.round,
                 "cur_success": cur_success, "best": best}))
            logger.info("PPO_DONE (collapse-stop) kept=%s", best["ckpt"])
            if wb is not None:
                wb.log({"round_success_rate": cur_success, "collapse_stop": 1}, step=gstep)
        _finish()
        return

    if wb is not None:
        wb.log({"round": args.round, "n_chunk_samples": stats["n_chunk_samples"],
                "n_wm_samples": stats.get("n_wm_samples", 0),
                "round_success_rate": cur_success,
                **{f"success_rate/{k}": v["success_rate"] for k, v in stats["per_task"].items()},
                **{f"n_success/{k}": v["n_success"] for k, v in stats["per_task"].items()}},
               step=gstep)

    use_group_adv = args.group_adv and args.adv_mode == "group"
    samples = ppo_samples
    if args.adv_mode == "paired_vla":
        for s in samples:
            s["adv"] = s["ret"]
        adv_all = [s["adv"] for s in samples]
        n_pos = sum(1 for a in adv_all if a > 0.02)
        n_neg = sum(1 for a in adv_all if a < -0.02)
        logger.info(
            "PAIRED_VLA adv mean=%.4f std=%.4f pos=%d neg=%d zero=%d | wm_samples=%d",
            float(np.mean(adv_all)) if adv_all else 0.0,
            float(np.std(adv_all)) if adv_all else 0.0,
            n_pos, n_neg, len(adv_all) - n_pos - n_neg, len(wm_samples),
        )
    elif use_group_adv:
        from collections import defaultdict
        by_task: dict[str, list] = defaultdict(list)
        for s in samples:
            by_task[s["task"]].append(s["ret"])
        task_mu = {t: float(np.mean(v)) for t, v in by_task.items()}
        task_sd = {t: float(np.std(v)) for t, v in by_task.items()}
        for s in samples:
            a = s["ret"] - task_mu[s["task"]]
            if args.group_adv_std:
                a = a / (task_sd[s["task"]] + 1e-4)
            s["adv"] = a
        adv_all = [s["adv"] for s in samples]
        n_pos = sum(1 for a in adv_all if a > 0.02)
        n_neg = sum(1 for a in adv_all if a < -0.02)
        logger.info("GROUP_ADV mean-subtracted std_div=%d | adv mean=%.4f std=%.4f pos=%d neg=%d zero=%d | tasks=%d",
                    args.group_adv_std, float(np.mean(adv_all)), float(np.std(adv_all)),
                    n_pos, n_neg, len(adv_all) - n_pos - n_neg, len(task_mu))

    if args.max_samples > 0:
        random.shuffle(samples)
        samples = samples[: args.max_samples]
    # WM-only vla_zero samples only matter when WAN is training
    all_wm = wm_samples if (not args.freeze_wan and args.video_beta > 0) else []

    # ---- shard samples across DDP ranks ----
    if ddp:
        random.Random(args.seed).shuffle(samples)  # identical order on all ranks
        n_local = len(samples) // world_size
        samples = samples[rank * n_local:(rank + 1) * n_local]
        logger.info("DDP shard rank=%d -> %d/%d samples", rank, len(samples), stats["n_chunk_samples"])

    if not samples:
        logger.warning("no samples this round; copying ckpt unchanged.")
        if is_main:
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.load(args.ckpt, map_location="cpu"), args.out_ckpt)
            logger.info("PPO_DONE (noop) -> %s", args.out_ckpt)
        _finish()
        return

    # ---- model + optimizer ----
    model = build_model(args.ckpt, args.wan_dir, args.delta_stats)
    model.train()
    use_lora = args.wan_lora and not args.freeze_wan
    lora_wrapped = {}
    if use_lora:
        from wrm_rl.lora import apply_lora_to_wan, lora_parameters
        model.video_model.requires_grad_(False)
        lora_wrapped = apply_lora_to_wan(model, r=args.lora_r, alpha=args.lora_alpha)
        logger.info("WAN LoRA: %d layers (r=%d alpha=%d)", len(lora_wrapped), args.lora_r, args.lora_alpha)

    # explicit trainable param lists (VAE and any other frozen submodule excluded)
    und_params = list(model.und_expert.parameters())
    action_params = list(model.action_expert.parameters())
    value_params = list(model.value_head.parameters())
    if args.freeze_wan:
        wan_params = []
    elif use_lora:
        wan_params = list(lora_parameters(lora_wrapped))
    else:
        # WAN transformer only (video_model.wan_model.*); the VAE stays frozen.
        wan_params = [p for n, p in model.named_parameters() if n.startswith("video_model.wan_model.")]

    # value_head only enters the optimizer when the critic is actually used
    # (c_value>0); otherwise it stays a frozen random head (excluded from weight
    # decay and DDP all-reduce).
    value_params_opt = value_params if args.c_value > 0 else []
    groups = [{"params": action_params + und_params + value_params_opt, "lr": args.lr}]
    if wan_params:
        groups.append({"params": wan_params, "lr": args.wan_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # fixed union of params that ever receive grad -> stable DDP all-reduce set
    reduce_params = action_params + und_params + value_params_opt + wan_params
    n_train = sum(p.numel() for p in reduce_params)
    logger.info("PARAMS %s", json.dumps({
        "trainable_M": round(n_train / 1e6, 1),
        "wan_mode": "lora" if use_lora else ("frozen" if args.freeze_wan else "full"),
        "n_action": len(action_params), "n_und": len(und_params),
        "n_value": len(value_params), "n_wan": len(wan_params),
    }))

    use_ckpt = not args.no_checkpoint and not args.freeze_wan
    device = "cuda"
    local_accum = max(1, args.grad_accum // world_size)
    t0 = time.time()

    # ---- precompute reference old_logp (+ V_old only if critic is used) ----
    need_value = (args.c_value > 0) or (use_group_adv is False and args.adv_mode != "paired_vla")
    logger.info("precompute old_logp%s at theta_k for %d PPO samples ...",
                " + V_old" if need_value else "", len(samples))
    with torch.no_grad():
        for s in samples:
            tr = load_trace(s["trace_path"])
            lang = load_lang(tr)
            s["old_logp"] = float(compute_logp(model, tr, lang, args.video_mode, use_ckpt=False).item())
            s["v_old"] = float(compute_value(model, tr).item()) if need_value else 0.0
            if not use_group_adv and args.adv_mode != "paired_vla":
                s["adv"] = s["ret"] - s["v_old"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not use_group_adv and args.adv_mode != "paired_vla" and args.norm_adv and len(samples) > 1:
        std = float(np.std([s["adv"] for s in samples])) + 1e-6
        for s in samples:
            s["adv"] = s["adv"] / std
        logger.info("advantages std-scaled by %.4f", std)
    adv_all = [s["adv"] for s in samples]
    logger.info("reference ready (%.1fs); adv mean=%.4f std=%.4f",
                time.time() - t0, float(np.mean(adv_all)), float(np.std(adv_all)))

    # ---- PPO epochs ----
    n_step = 0
    opt.zero_grad(set_to_none=True)
    run = {k: 0.0 for k in ["pg", "vloss", "video", "ratio", "adv", "clipfrac", "kl", "v", "n", "n_vid"]}
    stop_training = False

    for epoch in range(args.epochs):
        if stop_training:
            break
        train_items = [(s, True) for s in samples] + [(w, False) for w in all_wm]
        random.shuffle(train_items)
        for i, (s, is_ppo) in enumerate(train_items):
            tr = load_trace(s["trace_path"])
            lang = load_lang(tr)

            # ---- Pass 1: WM supervised (video) ----
            if args.video_beta > 0 and not args.freeze_wan and s.get("future_path") \
                    and Path(s["future_path"]).exists():
                set_pass("video", wan_params, action_params, value_params, und_params, args.freeze_wan)
                v_loss = compute_video_loss(model, tr, s["future_path"], lang)
                (args.video_beta * v_loss / local_accum).backward()
                run["video"] += float(v_loss.item())
                run["n_vid"] += 1

            if not is_ppo:
                if (i + 1) % local_accum == 0:
                    if ddp:
                        for p in reduce_params:
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= world_size
                    gn = torch.nn.utils.clip_grad_norm_(reduce_params, args.grad_clip)
                    if torch.isfinite(gn):
                        opt.step()
                    else:
                        logger.warning("non-finite grad_norm at wm step -> skip")
                    opt.zero_grad(set_to_none=True)
                    n_step += 1
                    gstep += 1
                continue

            A = torch.tensor(s["adv"], device=device, dtype=torch.float32).clamp(-args.adv_clip, args.adv_clip)
            ret = torch.tensor(s["ret"], device=device, dtype=torch.float32)
            v_old = torch.tensor(s["v_old"], device=device, dtype=torch.float32)
            old_logp = torch.tensor(s["old_logp"], device=device, dtype=torch.float32)

            # ---- Pass 2: action PPO ----
            set_pass("action", wan_params, action_params, value_params, und_params, args.freeze_wan)
            logp_new = compute_logp(model, tr, lang, args.video_mode, use_ckpt)  # [1]
            ratio = torch.exp(logp_new.squeeze(0) - old_logp)
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A
            pg_loss = -torch.min(surr1, surr2)

            if args.c_value > 0:
                v_new = compute_value(model, tr).squeeze(0)  # []
                v_unclipped = (v_new - ret) ** 2
                v_clipped_val = v_old + torch.clamp(v_new - v_old, -args.value_clip, args.value_clip)
                v_clipped = (v_clipped_val - ret) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped)
            else:
                v_new = torch.zeros((), device=device)
                value_loss = torch.zeros((), device=device)

            kl_pen = (logp_new.squeeze(0) - old_logp) ** 2
            a_loss = pg_loss + args.c_value * value_loss + args.kl_beta * kl_pen
            (a_loss / local_accum).backward()

            run["pg"] += float(pg_loss.item())
            run["vloss"] += float(value_loss.item())
            run["ratio"] += float(ratio.item())
            run["adv"] += float(A.item())
            run["clipfrac"] += float((surr2 < surr1).float().item())
            run["kl"] += float(kl_pen.item())
            run["v"] += float(v_new.item())
            run["n"] += 1
            del tr, lang, logp_new, ratio, surr1, surr2, pg_loss, a_loss, kl_pen, value_loss, v_new
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % local_accum == 0:
                if ddp:
                    for p in reduce_params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
                gn = torch.nn.utils.clip_grad_norm_(reduce_params, args.grad_clip)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    logger.warning("non-finite grad_norm at step %d -> skip", n_step + 1)
                opt.zero_grad(set_to_none=True)
                n_step += 1
                gstep += 1
                k = max(run["n"], 1)
                kv = max(run["n_vid"], 1)
                metrics = {
                    "pg": run["pg"] / k, "value_loss": run["vloss"] / k,
                    "video_loss": run["video"] / kv, "ratio": run["ratio"] / k,
                    "adv": run["adv"] / k, "clipfrac": run["clipfrac"] / k,
                    "kl": run["kl"] / k, "v_mean": run["v"] / k, "grad_norm": float(gn),
                }
                if is_main:
                    logger.info("STEP %s", json.dumps({
                        "epoch": epoch, "step": n_step, "sample": f"{i+1}/{len(train_items)}",
                        **{kk: round(vv, 5) for kk, vv in metrics.items()},
                        "elapsed_s": round(time.time() - t0, 1),
                    }))
                if wb is not None:
                    wb.log({f"train/{kk}": vv for kk, vv in metrics.items()}, step=gstep)
                run = {kk: 0.0 for kk in run}

                # PPO trust-region guard: once a step's mean KL blows past target,
                # the policy has moved far enough this round -> stop (avoids the
                # epoch-1 ratio runaway observed at higher lr). The KL is averaged
                # across DDP ranks so every rank makes the SAME stop decision
                # (otherwise a partial break would deadlock the next all-reduce).
                if args.target_kl > 0:
                    kl_val = metrics["kl"]
                    if ddp:
                        t = torch.tensor([kl_val], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        kl_val = t.item() / world_size
                    if kl_val > args.target_kl:
                        if is_main:
                            logger.info("TARGET_KL_STOP kl=%.3f > %.3f at epoch %d step %d",
                                        kl_val, args.target_kl, epoch, n_step)
                        stop_training = True
                        break

    # ---- save theta_{k+1} (rank0) ----
    agg_eps = sum(t["n_episodes"] for t in stats["per_task"].values())
    agg_succ = sum(t["n_success"] for t in stats["per_task"].values())
    if is_main:
        if use_lora:
            from wrm_rl.lora import merge_and_collect_base
            merged_wan = merge_and_collect_base(lora_wrapped)
            sd = model.state_dict()
            keep = {}
            for n, p in model.named_parameters():
                if p.requires_grad and (".lora_A" not in n and ".lora_B" not in n):
                    keep[n] = sd[n]
            # value_head + action + und (requires_grad may be toggled off at save; collect explicitly)
            for n, v in sd.items():
                if n.startswith(("value_head.", "und_expert.", "action_expert.")) or n.startswith("delta_"):
                    keep[n] = v
            keep.update(merged_wan)
            logger.info("LoRA merged into %d WAN tensors", len(merged_wan))
        else:
            sd = model.state_dict()
            keep = {}
            for n, v in sd.items():
                if n.startswith(("video_model.", "action_expert.", "und_expert.", "value_head.")) \
                        or n.startswith("delta_"):
                    if args.freeze_wan and n.startswith("video_model."):
                        continue
                    keep[n] = v
        out = {"step": args.round + 1, "trainable_state_dict": keep,
               "config": {"rl": True, "algo": "ppo", **vars(args)}}
        Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.out_ckpt)
        logger.info("PPO_DONE %s", json.dumps({
            "out_ckpt": args.out_ckpt, "tensors": len(keep), "opt_steps": n_step,
            "round_success_rate": round(agg_succ / max(agg_eps, 1), 4),
            "per_task_success": {k: v["success_rate"] for k, v in stats["per_task"].items()},
            "elapsed_s": round(time.time() - t0, 1),
        }))
        if best is None or cur_success >= best["success"]:
            best_path.write_text(json.dumps(
                {"success": cur_success, "ckpt": args.ckpt, "round": args.round}))
            logger.info("BEST updated -> success=%.4f ckpt=%s", cur_success, args.ckpt)
        if wb is not None:
            wb.log({"round_success_rate": agg_succ / max(agg_eps, 1)}, step=gstep)

    _finish()


if __name__ == "__main__":
    main()
