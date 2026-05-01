import types
from typing import List, Optional

import torch
from torch import Tensor

from src.model.vla.pizero import PiZero


def _make_acg_v_proj(orig_forward_v_proj):
    def patched(self, x):
        v = orig_forward_v_proj(x)  # [B, num_kv_heads, seq, head_dim]
        self._acg_v_stash = v
        return v

    return patched


def _make_acg_o_proj(orig_forward_o_proj):
    def patched(self, x):
        v = self._acg_v_stash  # [B, num_kv_heads, seq, head_dim]
        if self.num_key_value_groups > 1:
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        # [B, num_heads, seq, head_dim] -> [B, seq, num_heads * head_dim]
        bsz, _, seq, _ = v.shape
        v = v.transpose(1, 2).contiguous().view(bsz, seq, -1)
        if v.dtype != self.o_proj.weight.dtype:
            v = v.to(self.o_proj.weight.dtype)
        return orig_forward_o_proj(v)

    return patched


def _patch_action_attention(action_mixture, skip_blocks: List[int]):
    """Patch action mixture's V/O projections at skip_blocks for ACG perturbation.

    Returns backups for restoration via `_unpatch_action_attention`.
    """
    backups = []
    n_layers = len(action_mixture.layers)
    for i in skip_blocks:
        if i < 0 or i >= n_layers:
            continue
        attn = action_mixture.layers[i].self_attn
        orig_v = attn.forward_v_proj
        orig_o = attn.forward_o_proj
        backups.append((i, orig_v, orig_o))
        attn.forward_v_proj = types.MethodType(_make_acg_v_proj(orig_v), attn)
        attn.forward_o_proj = types.MethodType(_make_acg_o_proj(orig_o), attn)
    return backups


def _unpatch_action_attention(action_mixture, backups) -> None:
    for i, orig_v, orig_o in backups:
        attn = action_mixture.layers[i].self_attn
        attn.forward_v_proj = orig_v
        attn.forward_o_proj = orig_o
        if hasattr(attn, "_acg_v_stash"):
            del attn._acg_v_stash


class PiZero_ACG(PiZero):
    """PiZero with ACG inference-time guidance on the action expert mixture."""

    @torch.no_grad()
    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        scale: float = 3.0,
        skip_blocks: Optional[List[int]] = None,
    ) -> Tensor:
        if skip_blocks is None:
            skip_blocks = [7, 9, 11]

        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = self.proprio_encoder(proprios)

        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)

        action_mixture = self.joint_model.mixtures["action"]
        run_acg = scale != 1.0 and len(skip_blocks) > 0

        for _ in range(self.num_inference_steps):
            time_cond = self.time_embedding(t)
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)

            cond_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",
            )["action"]
            action_vel = self.action_decoder(cond_embeds)

            if run_acg:
                if self.action_expert_adaptive_mode:
                    action_embeds_p = self.action_encoder(action)
                else:
                    action_embeds_p = self.action_encoder(action, time_cond)

                backups = _patch_action_attention(action_mixture, skip_blocks)
                try:
                    perturb_embeds = self.joint_model(
                        attention_mask=action_mask,
                        position_ids_all={"action": action_position_ids},
                        embeds_all={"action": action_embeds_p},
                        time_cond=time_cond,
                        kv_caches=kv_caches,
                        cache_mode="append_non_active",
                    )["action"]
                finally:
                    _unpatch_action_attention(action_mixture, backups)
                action_vel_perturb = self.action_decoder(perturb_embeds)
                action_vel = action_vel + (scale - 1) * (action_vel - action_vel_perturb)

            action = action + delta_t * action_vel
            t = t + delta_t

        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action, -self.final_action_clip_value, self.final_action_clip_value
            )
        return action


def apply_acg_to_pizero(
    pizero_model: PiZero,
    scale: float = 3.0,
    skip_blocks: Optional[List[int]] = None,
) -> PiZero:
    """Bind ACG-guided infer_action onto an existing PiZero instance."""
    if skip_blocks is None:
        skip_blocks = [7, 9, 11]

    base_infer_action = PiZero_ACG.infer_action

    @torch.no_grad()
    def infer_action(
        self,
        input_ids,
        pixel_values,
        image_text_proprio_mask,
        action_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
        proprios,
        scale: float = scale,
        skip_blocks: Optional[List[int]] = skip_blocks,
    ):
        return base_infer_action(
            self,
            input_ids,
            pixel_values,
            image_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
            scale=scale,
            skip_blocks=skip_blocks,
        )

    pizero_model.infer_action = types.MethodType(infer_action, pizero_model)
    return pizero_model
