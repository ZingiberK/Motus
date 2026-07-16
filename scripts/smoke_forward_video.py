"""1-GPU forward(+optional step) smoke: verify video_loss and diagnose NaN after update."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not (ROOT / "wan").exists() and (ROOT / "bak" / "wan").exists():
    os.symlink("bak/wan", ROOT / "wan")

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import collate_fn
from data.vla_rollout_dataset import VLARolloutDataset
from models.motus import Motus, MotusConfig


def build_model(cfg):
    mc = MotusConfig(
        wan_checkpoint_path=cfg.model.wan.checkpoint_path,
        vae_path=cfg.model.wan.vae_path,
        wan_config_path=cfg.model.wan.config_path,
        vlm_checkpoint_path=cfg.model.vlm.checkpoint_path,
        video_precision=cfg.model.wan.precision,
        action_state_dim=cfg.common.state_dim,
        action_dim=cfg.common.action_dim,
        action_expert_dim=cfg.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=cfg.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=cfg.model.action_expert.norm_eps,
        und_expert_hidden_size=cfg.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=cfg.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=cfg.model.und_expert.norm_eps,
        vlm_adapter_input_dim=cfg.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=cfg.model.und_expert.vlm.projector_type,
        global_downsample_rate=cfg.common.global_downsample_rate,
        video_action_freq_ratio=cfg.common.video_action_freq_ratio,
        num_video_frames=cfg.common.num_video_frames,
        video_height=cfg.common.video_height,
        video_width=cfg.common.video_width,
        batch_size=1,
        video_loss_weight=cfg.model.loss_weights.video_loss_weight,
        action_loss_weight=cfg.model.loss_weights.action_loss_weight,
        training_mode="finetune",
    )
    model = Motus(mc)
    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in {id(x) for x in wan_params}]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": 5e-5}, {"params": wan_params, "lr": 5e-5}],
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    return model, opt


def main():
    cfg = OmegaConf.load("configs/motus_finetune_smoke.yaml")
    print("building model...")
    model, optimizer = build_model(cfg)
    model.cuda()
    ckpt = cfg.finetune.checkpoint_path
    print(f"loading finetune weights from {ckpt}")
    model.load_pretrain_weights(ckpt)

    ds = VLARolloutDataset(
        dataset_dir=cfg.dataset.dataset_dir,
        global_downsample_rate=cfg.common.global_downsample_rate,
        video_action_freq_ratio=cfg.common.video_action_freq_ratio,
        num_video_frames=cfg.common.num_video_frames,
        video_size=(cfg.common.video_height, cfg.common.video_width),
        vlm_checkpoint_path=cfg.model.vlm.checkpoint_path,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    do_update = os.environ.get("DO_UPDATE", "0") == "1"
    n = int(os.environ.get("N_STEPS", "8"))
    print(f"running {n} steps, do_update={do_update}")

    for i, batch in enumerate(dl):
        if i >= n:
            break
        if batch is None:
            print(i, "None batch"); continue
        first = batch["first_frame"].cuda().bfloat16()
        video = batch["video_frames"].cuda().bfloat16()
        state = batch["initial_state"].cuda().bfloat16()
        actions = batch["action_sequence"].cuda().bfloat16()
        lang = batch["language_embedding"]
        if lang is not None:
            lang = lang.cuda().bfloat16()
        vlm = batch["vlm_inputs"]
        if vlm is not None:
            vlm = {k: v.cuda() if torch.is_tensor(v) else v for k, v in vlm.items()}

        # sanity on inputs
        def chk(name, t):
            if t is None: return
            bad = (~torch.isfinite(t.float())).any().item()
            print(f"  {name}: shape={tuple(t.shape)} finite={not bad} "
                  f"absmax={t.float().abs().max().item():.4f}")
        print(f"--- step {i} ---")
        chk("first", first); chk("video", video); chk("state", state)
        chk("actions", actions); chk("lang", lang)

        optimizer.zero_grad(set_to_none=True)
        out = model.training_step(
            first_frame=first, video_frames=video, state=state, actions=actions,
            language_embeddings=lang, vlm_inputs=vlm, return_dict=True,
        )
        v = out["video_loss"].item(); a = out["action_loss"].item(); t = out["total_loss"].item()
        print(f"  loss total={t:.4f} video={v:.4f} action={a:.4f}")

        if do_update:
            out["total_loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            print(f"  grad_norm(clipped report)={float(gn):.4f}")
            # raw check: any non-finite grad?
            bad_g = 0; max_g = 0.0
            for p in model.parameters():
                if p.grad is None: continue
                g = p.grad.detach().float()
                if not torch.isfinite(g).all():
                    bad_g += 1
                max_g = max(max_g, g.abs().max().item())
            print(f"  grad bad_tensors={bad_g} absmax={max_g:.4f}")
            optimizer.step()
            bad_p = sum(1 for p in model.parameters() if p.requires_grad and not torch.isfinite(p).all())
            print(f"  params nonfinite_tensors={bad_p}")


if __name__ == "__main__":
    main()
