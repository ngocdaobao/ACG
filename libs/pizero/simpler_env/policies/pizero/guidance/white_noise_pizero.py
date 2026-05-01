"""White-Noise Guidance for the open_pi_zero PiZero model.

Applies white-noise perturbation guidance at inference time. At specified
blocks of `joint_model.mixtures["action"].layers`, the action mixture's input
to that block is corrupted with additive Gaussian noise (std=`noise_std`)
before the block's input_layernorm. The denoising-step velocity is then
guidance-augmented:

    v = v_perturb + scale * (v - v_perturb)

This mirrors `white_noise.py:FlowmatchingActionHead_White_Noise` (which adds
`torch.randn_like(hidden_states) * noise_std` at the start of selected
BasicTransformerBlock.forward calls), adapted to pizero's layer-method
dispatch (`MixtureDecoderLayer.forward_norm("input_layernorm", ...)`).

Usage (mirrors __init__.py's `modify_gr00t_policy` style):

    import types
    from robomimic.algo.guidance.white_noise_pizero import PiZero_White_Noise

    pizero_model.infer_action = types.MethodType(
        PiZero_White_Noise.infer_action, pizero_model
    )
    actions = pizero_model.infer_action(
        input_ids, pixel_values, image_text_proprio_mask, action_mask,
        vlm_position_ids, proprio_position_ids, action_position_ids, proprios,
        scale=1.5, skip_blocks=[7, 9, 11], noise_std=1.0,
    )
"""

import types
from typing import List, Optional

import torch
from torch import Tensor

from src.model.vla.pizero import PiZero


def _make_noisy_forward_norm(orig_forward_norm, noise_std: float):
    def patched(self, norm_name, x, cond=None):
        if norm_name == "input_layernorm":
            x = x + torch.randn_like(x) * noise_std
        return orig_forward_norm(norm_name, x, cond)

    return patched


def _patch_action_layers(action_mixture, skip_blocks: List[int], noise_std: float):
    """Wrap forward_norm of action layers at skip_blocks to inject input noise.

    Returns backups for restoration via `_unpatch_action_layers`.
    """
    backups = []
    n_layers = len(action_mixture.layers)
    for i in skip_blocks:
        if i < 0 or i >= n_layers:
            continue
        layer = action_mixture.layers[i]
        orig = layer.forward_norm
        backups.append((i, orig))
        layer.forward_norm = types.MethodType(
            _make_noisy_forward_norm(orig, noise_std), layer
        )
    return backups


def _unpatch_action_layers(action_mixture, backups) -> None:
    for i, orig in backups:
        action_mixture.layers[i].forward_norm = orig


class PiZero_White_Noise(PiZero):
    """PiZero with white-noise inference-time guidance on the action expert."""

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
        skip_blocks: Optional[List[int]] = None,
        noise_std: float = 1.0,
    ) -> Tensor:
        if skip_blocks is None:
            skip_blocks = []

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
        run_wn = scale != 1.0 and len(skip_blocks) > 0

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

            if run_wn:
                if self.action_expert_adaptive_mode:
                    action_embeds_p = self.action_encoder(action)
                else:
                    action_embeds_p = self.action_encoder(action, time_cond)

                backups = _patch_action_layers(action_mixture, skip_blocks, noise_std)
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
                    _unpatch_action_layers(action_mixture, backups)
                action_vel_perturb = self.action_decoder(perturb_embeds)
                action_vel = action_vel_perturb + scale * (action_vel - action_vel_perturb)

            action = action + delta_t * action_vel
            t = t + delta_t

        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action, -self.final_action_clip_value, self.final_action_clip_value
            )
        return action
 

def apply_white_noise_to_pizero(
    pizero_model: PiZero,
    scale: float = 1.5,
    skip_blocks: Optional[List[int]] = None,
    noise_std: float = 1.0,
) -> PiZero:
    """Bind white-noise-guided infer_action onto an existing PiZero instance."""
    if skip_blocks is None:
        skip_blocks = []

    base_infer_action = PiZero_White_Noise.infer_action

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
        noise_std: float = noise_std,
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
            noise_std=noise_std,
        )

    pizero_model.infer_action = types.MethodType(infer_action, pizero_model)
    return pizero_model
