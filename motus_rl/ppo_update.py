"""Offline PPO update for Motus (one round) — Flow-SDE PPO + supervised WM.

Phase 2 of the single-version pipeline (see ``Motus/MOTUS_PLAN.md``). Isolated:

  * **Action expert** — PPO on Flow-SDE denoise log-probs
    (``Motus.sample_actions_sde`` / ``action_logprob_from_trace``).
  * **Video (WAN)** — supervised FM on simulator-observed futures
    (``Motus.video_supervised_loss``); never receives policy gradients.
  * **Und** — shared hub, trains in both passes.

Trace schema (per chunk, saved by Motus RL rollout server — see MOTUS_PLAN.md)::

    {
      "action_latents": [K+1, chunk, D],   # or [K+1, 1, chunk, D]
      "video_latents":  [K+1, C, n, H, W],
      "timesteps": [K+1], "eta": float, "old_logprob": float,
      "state": [14], "action": [chunk, 14],   # executed absolute qpos
      "first_frame": [3,H,W] in [0,1],        # Motus composite
      "instruction": str,
      # optional caches:
      "cam_high/left/right": uint8,           # if first_frame not stored
      "und_tokens": [...],                    # skip VLM re-extract if present
    }

Run (motus env, MOTUS_INFER_ROOT set)::

    cd Motus
    python -m motus_rl.ppo_update \\
        --tasks stack_blocks_three --rl_root /path/to/motus_rl_runs --round 0 \\
        --motus_ckpt /path/to/Motus_robotwin2 --wan /path/to/Wan2.2 \\
        --vlm /path/to/Qwen3-VL-2B --out_ckpt .../round1_action.pt
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
MOTUS_INFER = Path(os.environ.get(
    "MOTUS_INFER_ROOT", str(ROOT / "inference" / "robotwin" / "Motus")
))
sys.path.insert(0, str(MOTUS_INFER))
sys.path.insert(0, str(ROOT))

from motus_rl.reward import episode_returns, paired_baseline_returns  # noqa: E402

logger = logging.getLogger("motus_ppo")


def load_trace(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _ensure_bk(t: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize stored latents to [K+1, B, ...]."""
    if t.dim() >= 2 and t.shape[1] != 1 and name.startswith("action"):
        # [K+1, chunk, D] -> [K+1, 1, chunk, D]
        if t.dim() == 3:
            return t.unsqueeze(1)
    if t.dim() >= 2 and name.startswith("video") and t.shape[1] != 1:
        if t.dim() == 5:  # [K+1, C, n, H, W]
            return t.unsqueeze(1)
    return t


def _build_t5_vlm(policy, ff: torch.Tensor, instruction: str):
    """Build Motus T5 + VLM conditioning from a composite first-frame [1,3,H,W]."""
    scene_prefix = (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: "
    )
    full_instr = f"{scene_prefix}{instruction}"
    t5_out = policy.t5_encoder([full_instr], policy.device)
    if isinstance(t5_out, torch.Tensor):
        t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
    else:
        t5_list = t5_out
    first_frame_pil = policy._tensor_to_pil_image(ff.squeeze(0).cpu())
    vlm_inputs = policy._preprocess_vlm_messages(full_instr, first_frame_pil)
    return t5_list, vlm_inputs


def prep_cond(policy, tr: dict):
    """Rebuild Motus T5 + VLM conditioning from a stored trace.

    Preferred path uses the composite ``first_frame`` stored by the rollout server.
    Fallback builds the same composite from raw 3-view cams via ``policy.update_obs``.
    """
    instruction = tr.get("instruction") or tr.get("meta", {}).get("instruction", "")
    if "first_frame" in tr and tr["first_frame"] is not None:
        ff = tr["first_frame"].float()
        if ff.dim() == 3:
            ff = ff.unsqueeze(0)
        state = tr["state"].float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        policy.set_instruction(instruction)
        policy.obs_cache = [ff.to(policy.device)]
        policy.current_state = state.to(policy.device)
        t5_list, vlm_inputs = _build_t5_vlm(policy, ff, instruction)
        return ff.to(policy.device), state.to(policy.device), t5_list, vlm_inputs

    # Fallback: 3 raw cams -> composite via the same pipeline as deploy inference.
    state_np = tr["state"].numpy() if torch.is_tensor(tr["state"]) else np.asarray(tr["state"])
    obs = {
        "observation": {
            "head_camera": {"rgb": np.asarray(tr["cam_high"])},
            "left_camera": {"rgb": np.asarray(tr["cam_left"])},
            "right_camera": {"rgb": np.asarray(tr["cam_right"])},
        },
        "joint_action": {"vector": np.asarray(state_np, dtype=np.float32)},
    }
    policy.set_instruction(instruction)
    policy.update_obs(obs)
    ff = policy.obs_cache[-1]
    state = policy.current_state
    t5_list, vlm_inputs = _build_t5_vlm(policy, ff, instruction)
    return ff, state, t5_list, vlm_inputs


def compute_logp(model, policy, tr, use_ckpt: bool):
    ff, state, t5_list, vlm_inputs = prep_cond(policy, tr)
    al = _ensure_bk(tr["action_latents"].float(), "action")
    vl = _ensure_bk(tr["video_latents"].float(), "video")
    und = tr.get("und_tokens")
    if und is not None:
        und = und.float()
        if und.dim() == 2:
            und = und.unsqueeze(0)
    return model.action_logprob_from_trace(
        state=state,
        language_embeddings=t5_list,
        vlm_inputs=[vlm_inputs],
        action_latents=al.to(model.device),
        video_latents=vl.to(model.device),
        timesteps=tr["timesteps"].float().to(model.device),
        eta=float(tr["eta"] if not torch.is_tensor(tr["eta"]) else tr["eta"].item()),
        use_checkpoint=use_ckpt,
        und_tokens=und.to(model.device) if und is not None else None,
    )  # [B]


def compute_video_loss(model, policy, tr, future_path: str):
    ff, state, t5_list, vlm_inputs = prep_cond(policy, tr)
    fut = np.load(future_path)["frames"]  # [T,H,W,3] or [T,C,H,W]
    if fut.ndim == 4 and fut.shape[-1] == 3:
        vf = torch.from_numpy(fut).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    else:
        vf = torch.from_numpy(fut).float().unsqueeze(0)
        if vf.max() > 1.5:
            vf = vf / 255.0
    action = tr["action"].float()
    if action.dim() == 2:
        action = action.unsqueeze(0)
    if ff.dim() == 3:
        ff = ff.unsqueeze(0)
    return model.video_supervised_loss(
        first_frame=ff.to(model.device),
        video_frames=vf.to(model.device),
        state=state.to(model.device),
        actions_exec=action.to(model.device),
        language_embeddings=t5_list,
        vlm_inputs=[vlm_inputs],
    )


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def build_samples(round_dir: Path, task: str, gamma: float, max_chunks_per_ep: int,
                  adv_mode: str = "group"):
    """Build PPO samples from Motus RL episodes.

    Episode JSON fields (Motus):
      mode: "motus" | "vla_ref" (optional paired baseline)
      trace_paths: list of Motus SDE .pt paths
      future_paths: list of observed-future .npz
      success, seed, ep, pair_id
    """
    ep_files = sorted((round_dir / "episodes").glob(f"{task}_seed*_ep*.json"))
    ref_succ: dict[tuple, bool] = {}
    for f in ep_files:
        e = json.loads(f.read_text())
        if e.get("mode") in ("vla_ref", "vla_zero", "ref"):
            key = (task, int(e["seed"]), int(e.get("pair_id", e["ep"] // 2)))
            ref_succ[key] = bool(e["success"])

    ppo_samples, wm_samples = [], []
    n_eps = n_succ = n_fut = n_pol = n_ref = 0
    for f in ep_files:
        e = json.loads(f.read_text())
        mode = e.get("mode", "motus")
        pair_id = int(e.get("pair_id", e["ep"] // 2))
        n_eps += 1
        n_succ += int(e["success"])
        fps = e.get("future_paths", [])
        if mode in ("vla_ref", "vla_zero", "ref"):
            n_ref += 1
            # reference arm: WM only (no PPO)
            wps = e.get("wm_trace_paths", e.get("trace_paths", []))
            idxs = [i for i in range(len(wps)) if wps[i]]
            if max_chunks_per_ep > 0 and len(idxs) > max_chunks_per_ep:
                idxs = sorted(random.sample(idxs, max_chunks_per_ep))
            for i in idxs:
                fp = fps[i] if i < len(fps) else ""
                n_fut += int(bool(fp))
                wm_samples.append({"task": task, "trace_path": wps[i], "future_path": fp, "ppo": False})
            continue

        n_pol += 1
        tps = e.get("trace_paths", [])
        T = len(tps)
        key = (task, int(e["seed"]), pair_id)
        if adv_mode == "paired" and key in ref_succ:
            rets = paired_baseline_returns(bool(e["success"]), bool(ref_succ[key]), T, gamma)
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
            if fp:
                wm_samples.append({
                    "task": task, "trace_path": tps[i], "future_path": fp, "ppo": False,
                })

    stats = {
        "n_episodes": n_eps, "n_policy_episodes": n_pol, "n_ref_episodes": n_ref,
        "n_success": n_succ, "success_rate": round(n_succ / max(n_eps, 1), 4),
        "n_chunk_samples": len(ppo_samples), "n_wm_samples": len(wm_samples),
        "n_with_future": n_fut,
    }
    return ppo_samples, wm_samples, stats


def _set_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def set_pass(mode: str, wan_params, action_params, und_params, freeze_wan: bool):
    _set_grad(wan_params, (mode == "video") and (not freeze_wan))
    _set_grad(action_params, mode == "action")
    _set_grad(und_params, True)


def main():
    ap = argparse.ArgumentParser(description="Motus Flow-SDE PPO + supervised WM")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--rl_root", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--motus_ckpt", required=True, help="Motus_robotwin2 dir or DeepSpeed pt")
    ap.add_argument("--wan", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--ft_action_ckpt", default="", help="optional BC-finetuned action expert to init")
    ap.add_argument("--out_ckpt", required=True, help="save action(+und) state_dict here")
    ap.add_argument("--config", default="", help="robotwin.yml; default under MOTUS_INFER")
    # PPO
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--group_adv", type=int, default=1)
    ap.add_argument("--group_adv_std", type=int, default=0)
    ap.add_argument("--adv_mode", default="group", choices=["group", "paired"])
    ap.add_argument("--saturated_tasks", default="")
    ap.add_argument("--video_beta", type=float, default=0.5)
    ap.add_argument("--kl_beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--wan_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--target_kl", type=float, default=0.5)
    ap.add_argument("--adv_clip", type=float, default=4.0)
    ap.add_argument("--collapse_margin", type=float, default=0.2)
    ap.add_argument("--freeze_wan", action="store_true")
    ap.add_argument("--no_checkpoint", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_chunks_per_ep", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
    is_main = rank == 0
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed); torch.manual_seed(args.seed)
    setup_logger(Path(args.rl_root) / f"motus_learner_round{args.round}.log")
    logger.info("MOTUS_PPO_START rank=%d/%d %s", rank, world_size, json.dumps(vars(args)))

    # ---- Motus load via MotusPolicy (same as finetune / SDEdit server) ----
    from deploy_policy import MotusPolicy  # type: ignore
    cfg = args.config or str(MOTUS_INFER / "utils" / "robotwin.yml")
    policy = MotusPolicy(
        checkpoint_path=args.motus_ckpt, config_path=cfg,
        wan_path=args.wan, vlm_path=args.vlm, device=str(device),
        task_name="motus_ppo",
    )
    policy.save_images = False
    model = policy.model
    if args.ft_action_ckpt:
        sd = torch.load(args.ft_action_ckpt, map_location=device)
        missing, unexpected = model.action_expert.load_state_dict(sd, strict=False)
        logger.info("loaded FT action_expert missing=%d unexpected=%d", len(missing), len(unexpected))
    model.train()

    # ---- data ----
    saturated: set[str] = set()
    if args.saturated_tasks and Path(args.saturated_tasks).exists():
        saturated = set(json.loads(Path(args.saturated_tasks).read_text()).get("tasks", []))

    ppo_samples, wm_samples, stats = [], [], {"per_task": {}}
    for task in args.tasks:
        if task in saturated:
            stats["per_task"][task] = {"n_chunk_samples": 0, "skipped_saturated": True}
            continue
        round_dir = Path(args.rl_root) / task / f"round{args.round}"
        s_ppo, s_wm, st = build_samples(round_dir, task, args.gamma, args.max_chunks_per_ep, args.adv_mode)
        ppo_samples.extend(s_ppo); wm_samples.extend(s_wm); stats["per_task"][task] = st
    stats["n_chunk_samples"] = len(ppo_samples)
    stats["n_wm_samples"] = len(wm_samples)
    logger.info("ROUND_STATS %s", json.dumps(stats))

    if not ppo_samples:
        logger.warning("no PPO samples; exiting")
        return

    # group-relative advantage
    samples = ppo_samples
    if args.group_adv and args.adv_mode == "group":
        by_task: dict[str, list] = {}
        for s in samples:
            by_task.setdefault(s["task"], []).append(s)
        for task, lst in by_task.items():
            rets = [x["ret"] for x in lst]
            mean = sum(rets) / max(len(rets), 1)
            if args.group_adv_std:
                var = sum((r - mean) ** 2 for r in rets) / max(len(rets), 1)
                std = max(var ** 0.5, 1e-4)
            else:
                std = 1.0
            for x in lst:
                x["adv"] = (x["ret"] - mean) / std
    else:
        for s in samples:
            s["adv"] = s["ret"]

    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = random.sample(samples, args.max_samples)
    random.shuffle(samples)

    # param groups
    wan_params = [p for p in model.video_model.parameters() if p.requires_grad]
    # also video MoT modules if separate
    if hasattr(model, "video_module"):
        wan_params += [p for p in model.video_module.parameters() if p.requires_grad]
    action_params = list(model.action_expert.parameters())
    und_params = list(model.und_expert.parameters()) if hasattr(model, "und_expert") else []
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([
        {"params": action_params, "lr": args.lr},
        {"params": und_params, "lr": args.lr},
        {"params": wan_params, "lr": args.wan_lr},
    ], weight_decay=args.weight_decay)

    use_ckpt = not args.no_checkpoint
    stop_epochs = False
    for epoch in range(args.epochs):
        if stop_epochs:
            break
        logger.info("EPOCH %d/%d n=%d", epoch + 1, args.epochs, len(samples))
        opt.zero_grad(set_to_none=True)
        acc = 0
        kl_buf = []
        for si, s in enumerate(samples):
            tr = load_trace(s["trace_path"])
            # old logp (no grad)
            with torch.no_grad():
                set_pass("action", wan_params, action_params, und_params, freeze_wan=True)
                for p in action_params + und_params:
                    p.requires_grad_(False)
                logp_old = compute_logp(model, policy, tr, use_ckpt=False).detach()
                if "old_logprob" in tr:
                    # sanity: should be ~equal at epoch0 step0
                    stored = float(tr["old_logprob"] if not torch.is_tensor(tr["old_logprob"])
                                   else tr["old_logprob"].reshape(-1)[0])
                    if epoch == 0 and si < 3:
                        logger.info("LOGP_CHECK stored=%.4f recomputed=%.4f", stored, float(logp_old.mean()))

            # --- video supervised pass (accumulate; do NOT step/zero yet) ---
            if (s.get("future_path") or "").strip() and not args.freeze_wan:
                set_pass("video", wan_params, action_params, und_params, freeze_wan=False)
                try:
                    v_loss = compute_video_loss(model, policy, tr, s["future_path"])
                    (args.video_beta * v_loss / args.grad_accum).backward()
                except Exception as e:
                    logger.warning("video_loss skip: %s", e)

            # --- action PPO pass ---
            set_pass("action", wan_params, action_params, und_params, freeze_wan=True)
            logp_new = compute_logp(model, policy, tr, use_ckpt=use_ckpt)
            adv = float(np.clip(s["adv"], -args.adv_clip, args.adv_clip))
            ratio = torch.exp(logp_new - logp_old.to(logp_new.device))
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            a_loss = -torch.min(surr1, surr2).mean()
            if args.kl_beta > 0:
                a_loss = a_loss + args.kl_beta * (logp_new - logp_old.to(logp_new.device)).pow(2).mean()
            (a_loss / args.grad_accum).backward()

            with torch.no_grad():
                kl = float((logp_old.to(logp_new.device) - logp_new).mean().abs())
                kl_buf.append(kl)

            acc += 1
            if acc % args.grad_accum == 0 or si == len(samples) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in action_params + und_params + wan_params if p.requires_grad],
                    args.grad_clip,
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                mean_kl = sum(kl_buf[-args.grad_accum:]) / max(len(kl_buf[-args.grad_accum:]), 1)
                if si % (args.grad_accum * 5) == 0:
                    logger.info("step=%d epoch=%d a_loss=%.4f kl=%.4f adv=%.3f ratio=%.3f",
                                si, epoch, float(a_loss), mean_kl, adv, float(ratio.mean()))
                if args.target_kl > 0 and mean_kl > args.target_kl:
                    logger.warning("TARGET_KL stop epoch (kl=%.3f > %.3f)", mean_kl, args.target_kl)
                    stop_epochs = True
                    break

    # save action(+und) for deploy --ft_action_ckpt / full expert
    if is_main:
        Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_expert": model.action_expert.state_dict(),
            "und_expert": model.und_expert.state_dict() if hasattr(model, "und_expert") else {},
            "round": args.round,
            "args": vars(args),
        }
        torch.save(payload, args.out_ckpt)
        # also dump bare action_expert for motus_sdedit_server --ft_action_ckpt
        bare = Path(args.out_ckpt).with_name(Path(args.out_ckpt).stem + "_action_expert.pt")
        torch.save(model.action_expert.state_dict(), bare)
        logger.info("SAVED %s and %s", args.out_ckpt, bare)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    logger.info("MOTUS_PPO_DONE")


if __name__ == "__main__":
    main()
