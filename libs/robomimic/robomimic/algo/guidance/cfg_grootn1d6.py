# import copy
# from typing import Any, Dict

# import torch
# from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
# from gr00t.model.gr00t_n1 import GR00T_N1
# from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
# from transformers.feature_extraction_utils import BatchFeature

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

COMPUTE_DTYPE = torch.bfloat16


class Gr00t_N1d6_Policy_CFG(Gr00tPolicy_N1d6):
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None,
        scale: float = 7.0,
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

        #Prepare uncond
        unbatched_observation_uncond = copy.deepcopy(unbatched_observations)
        for obs in unbatched_observation_uncond:
            obs["language"][self.language_key][0]=""
        process_uncond_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        states_uncond = []
        for obs in unbatched_observation_uncond:
            vla_step_data_uncond = self._to_vla_step_data(obs)
            states_uncond.append(vla_step_data_uncond.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages_uncond = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data_uncond}]
            process_uncond_inputs.append(self.processor(messages_uncond))
        

        # Step 3: Collate processed inputs into a single batch for model
        # The collator wraps output in {"inputs": batch}; extract the inner dict
        # so that backbone.forward can access input_ids, attention_mask, pixel_values directly.
        collated_inputs = self.collate_fn(processed_inputs)["inputs"]
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        collated_uncond_inputs = self.collate_fn(process_uncond_inputs)["inputs"]
        collated_uncond_inputs = _rec_to_dtype(collated_uncond_inputs, dtype=torch.bfloat16)
        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(inputs=collated_inputs, inputs_uncond=collated_uncond_inputs, scale=scale)
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
        scale: float = 7.0,
    ):
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options, scale)
        if self.strict:
            self.check_action(action)
        return action, info        

class GR00T_N1d6_CFG(Gr00tN1d6):
    def get_action(
        self,
        inputs: dict,
        inputs_uncond: dict,
        scale: float = 7.0,
    ) -> BatchFeature:
            
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_inputs_uncond, _ = self.prepare_input(inputs_uncond)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs_uncond = self.backbone(backbone_inputs_uncond)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            backbone_outputs_uncond,
            scale=scale,
        )
        
        return action_head_outputs


class Gr00tN1d6ActionHead_CFG(Gr00tN1d6ActionHead):

    @torch.no_grad()
    def get_action_with_features(
        self, backbone_features: BatchFeature,
        backbone_features_uncond: BatchFeature,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        backbone_output_uncond: BatchFeature,
        scale: float = 7.0,
    ) -> BatchFeature:
        print('use cfg')
        # Get vision and language embeddings.
        vl_embeds = backbone_features  # (B, 99, 1536)
        vl_embeds_uncond = backbone_features_uncond  # (B, 99, 1536)
    
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
            vl_embs_uncond = vl_embeds_uncond

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)  # (B, T=17, 1536)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
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
                if self.config.use_alternate_vl_dit:
                    model_output_uncond = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds_uncond,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output_uncond.image_mask,
                        backbone_attention_mask=backbone_output_uncond.backbone_attention_mask,
                    )
                else:
                    model_output_uncond = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embs_uncond,
                        timestep=timesteps_tensor,
                    )
                pred_uncond = self.action_decoder(model_output_uncond, embodiment_id)
                pred = pred_uncond + scale * (pred - pred_uncond)

            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self, backbone_output: BatchFeature,
        action_input: BatchFeature,
        backbone_output_uncond: BatchFeature,
        scale: float = 7.0,
    ) -> BatchFeature:

        features = self._encode_features(backbone_output, action_input)
        features_uncond = self._encode_features(backbone_output_uncond, action_input)

        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            backbone_features_uncond=features_uncond.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            backbone_output_uncond=backbone_output_uncond,
            scale=scale,
        )
