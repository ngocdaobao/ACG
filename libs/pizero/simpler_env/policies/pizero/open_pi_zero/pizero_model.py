import hydra
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# This file lives INSIDE open_pi_zero/. We need THREE paths on sys.path so all
# Hydra _target_ strings in bridge.yaml / fractal.yaml resolve:
#   - open_pi_zero/                           -> `from src.model... import ...`
#   - .../pizero/        (parent)             -> `open_pi_zero.src.model...`
#   - .../libs/pizero/   (great-grandparent)  -> `simpler_env.policies.pizero...`
_OPEN_PI_ZERO_DIR = osp.dirname(osp.abspath(__file__))
_PIZERO_POLICY_DIR = osp.dirname(_OPEN_PI_ZERO_DIR)            # .../pizero/
_LIBS_PIZERO_DIR = osp.dirname(osp.dirname(osp.dirname(_PIZERO_POLICY_DIR)))  # .../libs/pizero/
for _p in (_OPEN_PI_ZERO_DIR, _PIZERO_POLICY_DIR, _LIBS_PIZERO_DIR):
    if _p not in sys.path:
        sys.path.append(_p)
from src.model.vla.pizero import PiZero

# Pre-import every module referenced by `_target_` strings in
# config/eval/{bridge,fractal}.yaml so they are cached in sys.modules.
# Hydra's `_locate("src.X.Y")` then succeeds via the cache, sidestepping
# namespace-package quirks where `import_module("src")` returns a partial
# namespace object on which `getattr(..., 'model')` may not auto-import.
import importlib as _importlib
for _mod in (
    "src.agent.eval",
    "src.model.paligemma.siglip",
    "src.model.vla.joint_model",
    "simpler_env.policies.pizero.simpler_adapter",
):
    try:
        _importlib.import_module(_mod)
    except ModuleNotFoundError:
        pass  # let hydra report the real failure if any are genuinely missing


def setup_torch_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)


class PiZeroInference:
    def __init__(self,
                 guidance, #guidance dict keys: type, scale, skip_blocks (for acg/wng), noise_std (for wng)
                 cfg_dir,
                 checkpoint_path,
                 policy_setup="widowx_bridge",
                 flow_sampling='beta',
                 use_ddp=False,
                 use_naive=False,
                 use_torch_compile=False,
                 seed=0,
    ):
        self.use_naive = use_naive
        
        if policy_setup == "widowx_bridge":
            cfg = OmegaConf.load('open_pi_zero/config/eval/bridge.yaml')
            if flow_sampling == 'beta':
                checkpoint_path = ('open_pi_zero/config/eval/bridge_beta_step19296_2024-12-26_22-30_42.pt')
            elif flow_sampling == 'uniform':
                checkpoint_path = ('open_pi_zero/config/eval/bridge_uniform_step19296_2024-12-26_22-31_42.pt')
            else:
                raise ValueError(f"Invalid flow_sampling: {flow_sampling}")
            
        elif policy_setup == "google_robot":
            cfg = OmegaConf.load('open_pi_zero/config/eval/fractal.yaml')
            if flow_sampling == 'beta':
                checkpoint_path = ('open_pi_zero/config/eval/fractal_beta_step29576_2024-12-29_13-10_42.pt')
            elif flow_sampling == 'uniform':
                checkpoint_path = ('open_pi_zero/config/eval/fractal_uniform_step29576_2024-12-31_22-26_42.pt')
            else:
                raise ValueError(f"Invalid flow_sampling: {flow_sampling}")
        
        cfg.flow_sampling = flow_sampling
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda')
        self.model = PiZero(cfg, use_ddp=use_ddp)
        load_checkpoint(self.model, checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        
        if guidance is not None and guidance.get('type') in ('acg', 'cfg', 'wng'):
            # Load the guidance modules directly from file to avoid triggering
            # robomimic/__init__.py -> robomimic/algo/__init__.py, which pulls
            # in `transformers.AutoModel` etc. that the pizero venv can't load.
            import importlib.util as _ilu
            _GUIDANCE_DIR = osp.join(
                _LIBS_PIZERO_DIR, "..", "robomimic", "robomimic", "algo", "guidance"
            )
            _GUIDANCE_DIR = osp.normpath(_GUIDANCE_DIR)

            def _load_guidance(name):
                spec = _ilu.spec_from_file_location(
                    name, osp.join(_GUIDANCE_DIR, f"{name}.py")
                )
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            if guidance['type'] == 'acg':
                _load_guidance("acg_pizero").apply_acg_to_pizero(
                    self.model, guidance['scale'], guidance['skip_blocks'],
                )
            elif guidance['type'] == 'cfg':
                _load_guidance("cfg_pizero").apply_cfg_to_pizero(
                    self.model, guidance['scale'],
                )
            elif guidance['type'] == 'wng':
                _load_guidance("white_noise_pizero").apply_white_noise_to_pizero(
                    self.model, guidance['scale'],
                    guidance['skip_blocks'], guidance['noise_std'],
                )

        # torch.compile (a) doesn't compose with the runtime monkey-patching
        # done by acg/cfg/wng guidance, and (b) requires triton>=3.0 whose
        # `triton.backends` is unavailable in this venv (we pinned triton<3.0
        # for bitsandbytes<0.45). Skip compile when guidance is active.
        _guidance_active = (
            guidance is not None and guidance.get('type') in ('acg', 'cfg', 'wng')
        )
        if use_torch_compile and not _guidance_active:
            self.model = torch.compile(self.model, mode='default')
        self.model.eval()

        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)
        self.env_adapter.reset()

    def reset(self, instruction, seed=None):
        self.env_adapter.reset()
        if seed is not None:
            setup_torch_seed(seed)
    
    def step(self, image, instruction, proprio, *args, **kwargs):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        raw_actions = self.forward_actions(inputs)
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions

    def preprocess_inputs(self, image, instruction, proprio):
        inputs = self.env_adapter.preprocess(image, instruction, proprio)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = \
            (self.model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=self.dtype))
        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(causal_mask)
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(self.dtype),
        }

        if self.use_naive:
            inputs.update({"causal_mask": causal_mask})
        else:
            inputs.update({
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
            })
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def forward_actions(self, inputs):
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_action_naive(**inputs)
            else:
                actions = self.model.infer_action(**inputs)
        return actions
