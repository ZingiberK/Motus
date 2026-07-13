"""Zero-dependency LoRA for the WAN video branch of WRMUnd.

We wrap selected ``nn.Linear`` layers inside ``video_model.wan_model.blocks`` with a
low-rank bypass.  The base weight is frozen; only ``lora_A`` / ``lora_B`` are trained.
``lora_B`` is zero-initialised so the wrapped layer is identical to the original at
step 0 -> the GRPO importance ratio starts at exactly 1.

At save time we ``merge()`` the low-rank update back into the base weight and emit a
plain state-dict under the *original* key names, so the deployment / rollout server
needs no LoRA-aware code: it just loads slightly more WAN weights via strict=False.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        # fp32 master weights for stable optimisation (autocast casts at matmul time)
        self.lora_A.weight.data = self.lora_A.weight.data.float()
        self.lora_B.weight.data = self.lora_B.weight.data.float()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # follow the (already-placed) base layer's device so freshly created LoRA
        # params don't end up stranded on CPU when the model is already on GPU.
        self.lora_A.to(self.base.weight.device)
        self.lora_B.to(self.base.weight.device)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_B(self.lora_A(x))
        return out + self.scaling * lora.to(out.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        dw = self.scaling * (self.lora_B.weight @ self.lora_A.weight)  # [out, in]
        self.base.weight.data.add_(dw.to(self.base.weight.dtype))


def apply_lora_to_wan(
    model,
    r: int = 16,
    alpha: int = 16,
    targets=("self_attn", "cross_attn", "ffn"),
):
    """Wrap WAN attention/ffn linears in-place. Returns {orig_key_prefix: LoRALinear}.

    ``orig_key_prefix`` is the dotted module path of the wrapped Linear in the *original*
    (unwrapped) model, e.g. ``video_model.wan_model.blocks.0.self_attn.q``.
    """
    wrapped = {}
    blocks = model.video_model.wan_model.blocks
    for bi, block in enumerate(blocks):
        base_prefix = f"video_model.wan_model.blocks.{bi}"

        for attn_name in ("self_attn", "cross_attn"):
            if attn_name not in targets:
                continue
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for proj in ("q", "k", "v", "o"):
                lin = getattr(attn, proj, None)
                if not isinstance(lin, nn.Linear):
                    continue
                w = LoRALinear(lin, r, alpha)
                setattr(attn, proj, w)
                wrapped[f"{base_prefix}.{attn_name}.{proj}"] = w

        if "ffn" in targets and isinstance(getattr(block, "ffn", None), nn.Sequential):
            for idx, sub in enumerate(block.ffn):
                if isinstance(sub, nn.Linear):
                    w = LoRALinear(sub, r, alpha)
                    block.ffn[idx] = w
                    wrapped[f"{base_prefix}.ffn.{idx}"] = w

    return wrapped


@torch.no_grad()
def merge_and_collect_base(wrapped) -> dict:
    """Merge every LoRA update and return {orig_key: tensor} for the merged base layers.

    Keys use the original (deploy) names: ``<prefix>.weight`` / ``<prefix>.bias``.
    """
    merged = {}
    for prefix, w in wrapped.items():
        w.merge()
        merged[f"{prefix}.weight"] = w.base.weight.data.clone()
        if w.base.bias is not None:
            merged[f"{prefix}.bias"] = w.base.bias.data.clone()
    return merged


def lora_parameters(wrapped):
    for w in wrapped.values():
        yield from (w.lora_A.weight, w.lora_B.weight)
