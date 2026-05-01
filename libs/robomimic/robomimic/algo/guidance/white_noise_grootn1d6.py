import types
from typing import Any, Dict, List, Optional

import copy
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6ActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy as Gr00tPolicy_N1d6
from gr00t.policy.gr00t_policy import _rec_to_dtype
from gr00t.data.types import MessageType
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.modules.dit import BasicTransformerBlock, _sdpa_context
COMPUTE_DTYPE = torch.bfloat16


class Gr00t_N1d6_Policy_WNG(Gr00tPolicy_N1d6):
    def _get_action(
        self, observation: dict[str, Any], 
        options: dict[str, Any] | None = None,
        scale: float = 3.0,
        skip_blocks: List[int] = [0, 8, 16, 24],
        noise_std: float = 1.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(
                **collated_inputs, 
                scale=scale, 
                skip_blocks=skip_blocks, 
                noise_std=noise_std
            )
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def get_action(
        self, observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        scale: float = 3.0,
        skip_blocks: List[int] = [0, 8, 16, 24],
        noise_std: float = 1.0,
    ):
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(
            observation, 
            options,
            scale,
            skip_blocks,
            noise_std,
        )
        if self.strict:
            self.check_action(action)
        return action, info        

class GR00T_N1d6_WNG(Gr00tN1d6):

    def get_action(
        self,
        inputs: dict,
        scale: float = 3.0,
        skip_blocks: List[int] = [0, 8, 16, 24],
        noise_std: float = 1.0,
        **kwargs,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            scale=scale,
            skip_blocks=skip_blocks,
            noise_std=noise_std,
        )

        return action_head_outputs


class Gr00tN1d6ActionHead_WNG(Gr00tN1d6ActionHead):
    @torch.no_grad()
    def get_action_with_features(
        self, backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        scale: float = 3.0,
        skip_blocks: List[int] = [0, 8, 16, 24],
        noise_std: float = 1.0,
        **kwargs,
    ) -> BatchFeature:
        print('use wng')
        def convert_to_bad_model(skip_blocks: List[int] = [0, 8, 16, 24]) -> None:
            n_blocks = len(self.model.transformer_blocks)
            for i in range(n_blocks):
                if i in skip_blocks:
                    # Create a closure that captures noise_std
                    def create_forward_with_perturb(noise_std):
                        def forward_with_perturb_bound(
                            self,
                            hidden_states: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            encoder_hidden_states: Optional[torch.Tensor] = None,
                            encoder_attention_mask: Optional[torch.Tensor] = None,
                            temb: Optional[torch.LongTensor] = None,
                        ) -> torch.Tensor:
                            hidden_states = hidden_states + torch.randn_like(hidden_states) * noise_std

                            # 0. Self-Attention
                            if self.norm_type == "ada_norm":
                                norm_hidden_states = self.norm1(hidden_states, temb)
                            else:
                                norm_hidden_states = self.norm1(hidden_states)

                            if self.pos_embed is not None:
                                norm_hidden_states = self.pos_embed(norm_hidden_states)

                            with _sdpa_context():
                                attn_output = self.attn1(
                                    norm_hidden_states,
                                    encoder_hidden_states=encoder_hidden_states,
                                    attention_mask=(
                                        encoder_attention_mask
                                        if encoder_hidden_states is not None
                                        else attention_mask
                                    ),
                                )
                            if self.final_dropout:
                                attn_output = self.final_dropout(attn_output)

                            hidden_states = attn_output + hidden_states
                            if hidden_states.ndim == 4:
                                hidden_states = hidden_states.squeeze(1)

                            # 4. Feed-forward
                            norm_hidden_states = self.norm3(hidden_states)
                            ff_output = self.ff(norm_hidden_states)

                            hidden_states = ff_output + hidden_states
                            if hidden_states.ndim == 4:
                                hidden_states = hidden_states.squeeze(1)
                            return hidden_states
                        return forward_with_perturb_bound

                    self.model.transformer_blocks[i].forward = types.MethodType(
                        create_forward_with_perturb(noise_std), self.model.transformer_blocks[i]
                    )

        def convert_to_original_model(skip_blocks: List[int] = [0, 8, 16, 24]) -> None:
            n_blocks = len(self.model.transformer_blocks)
            for i in range(n_blocks):
                if i in skip_blocks:
                    self.model.transformer_blocks[i].forward = types.MethodType(
                        BasicTransformerBlock.forward, self.model.transformer_blocks[i]
                    )

        # Get vision and language embeddings.
        vl_embeds = backbone_features  # (B, 99, 1536)
        vl_embeds_perturb = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )  # (B, T=16, D=32)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)  # (B, T=16, 1536)
            # Maybe add position embedding.
            if self.config.add_pos_embed:  # True
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds
            vl_embs_perturb = vl_embeds_perturb

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)  # (B, T=17, 1536)

            convert_to_original_model(skip_blocks)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            if scale != 1.0:
                convert_to_bad_model(skip_blocks)
                if self.config.use_alternate_vl_dit:
                    model_output_perturb = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embs_perturb,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output_perturb = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embs,
                        timestep=timesteps_tensor,
                    )
                pred_perturb = self.action_decoder(model_output_perturb, embodiment_id)
                pred = pred_perturb + scale * (pred - pred_perturb)

            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )
    
    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        scale: float = 3.0,
        skip_blocks: List[int] = [0, 8, 16, 24],
        noise_std: float = 1.0,
    ):
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            scale=scale,
            skip_blocks=skip_blocks,
            noise_std=noise_std,
        )
