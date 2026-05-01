"""CFG (Classifier-Free Guidance) for the open_pi_zero PiZero model.

Applies CFG at inference time to action generation. Builds two prefix kv_caches
- one from the real (image, text) prefix, one from an "unconditional" prefix
where the language tokens are replaced by pad tokens (matching the
`task_description = ""` ablation in `cfg.py`). Each denoising step then runs
the action mixture twice (once per cache) and blends:

    v = v_uncond + scale * (v_cond - v_uncond)

Usage (mirrors __init__.py's `modify_gr00t_policy` style):

    import types
    from robomimic.algo.guidance.cfg_pizero import PiZero_CFG

    pizero_model.infer_action = types.MethodType(
        PiZero_CFG.infer_action, pizero_model
    )
    actions = pizero_model.infer_action(
        input_ids, pixel_values, image_text_proprio_mask, action_mask,
        vlm_position_ids, proprio_position_ids, action_position_ids, proprios,
        scale=1.5,
    )
"""

import types
from typing import Optional

import torch
from torch import Tensor

from src.model.vla.pizero import PiZero


def _build_uncond_input_ids(input_ids: torch.LongTensor, image_token_index: int, pad_token_id: int) -> torch.LongTensor:
    """Replace all non-image, non-pad tokens with pad_token_id (drop language conditioning)."""
    out = input_ids.clone()
    text_mask = (out != image_token_index) & (out != pad_token_id)
    out[text_mask] = pad_token_id
    return out


class PiZero_CFG(PiZero):
    """PiZero with CFG inference-time guidance on action generation."""

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
        scale: float = 1.5,
    ) -> Tensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        # Conditional prefill
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

        run_cfg = scale != 1.0
        if run_cfg:
            # Unconditional prefill: text tokens replaced with pad_token_id
            uncond_input_ids = _build_uncond_input_ids(
                input_ids, self.image_token_index, self.pad_token_id
            )
            kv_caches_uncond = self.joint_model.build_mixture_caches()
            inputs_embeds_uncond = self._forward_siglip_and_text_embedding(
                uncond_input_ids, pixel_values
            )
            proprio_embeds_uncond = self.proprio_encoder(proprios)
            _, kv_caches_uncond = self.joint_model(
                attention_mask=image_text_proprio_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds_uncond,
                    "proprio": proprio_embeds_uncond,
                },
                kv_caches=kv_caches_uncond,
                return_caches=True,
            )

        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)

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

            if run_cfg:
                if self.action_expert_adaptive_mode:
                    action_embeds_u = self.action_encoder(action)
                else:
                    action_embeds_u = self.action_encoder(action, time_cond)
                uncond_embeds = self.joint_model(
                    attention_mask=action_mask,
                    position_ids_all={"action": action_position_ids},
                    embeds_all={"action": action_embeds_u},
                    time_cond=time_cond,
                    kv_caches=kv_caches_uncond,
                    cache_mode="append_non_active",
                )["action"]
                action_vel_uncond = self.action_decoder(uncond_embeds)
                action_vel = action_vel_uncond + scale * (action_vel - action_vel_uncond)

            action = action + delta_t * action_vel
            t = t + delta_t

        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action, -self.final_action_clip_value, self.final_action_clip_value
            )
        return action


def apply_cfg_to_pizero(pizero_model: PiZero, scale: float = 3.0) -> PiZero:
    """Bind CFG-guided infer_action onto an existing PiZero instance."""
    base_infer_action = PiZero_CFG.infer_action

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
        )

    pizero_model.infer_action = types.MethodType(infer_action, pizero_model)
    return pizero_model
