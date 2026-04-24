from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro
import types
 
DEFAULT_MODEL_SERVER_PORT = 5555

 
@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    use_acg_guidance: bool = True
    """Whether to modify the policy to use ACG guidance"""

def modified_policy_for_acg(policy: Gr00tPolicy):
    """
        Modify the given GR00T-N1.6 policy to use the ACG guidance method. 
        This is done by replacing the get_action method of the policy and its model with the ACG versions.
    """
    try:
        acg_module = importlib.import_module("robomimic.algo.guidance.acg_grootn1d6")
    except ImportError:
        # Support running this file directly by adding sibling libs/robomimic.
        robomimic_repo = Path(__file__).resolve().parents[2].parent / "robomimic"
        if str(robomimic_repo) not in sys.path:
            sys.path.insert(0, str(robomimic_repo))
        acg_module = importlib.import_module("robomimic.algo.guidance.acg_grootn1d6")

    GR00T_N1d6_ACG = acg_module.GR00T_N1d6_ACG
    Gr00tN1d6ActionHead_ACG = acg_module.Gr00tN1d6ActionHead_ACG
    Gr00t_N1d6_Policy_ACG = acg_module.Gr00t_N1d6_Policy_ACG

    policy.get_action = types.MethodType(Gr00t_N1d6_Policy_ACG.get_action, policy)
    policy.model.get_action = types.MethodType(GR00T_N1d6_ACG.get_action, policy.model)
    policy.model.action_head.get_action_with_features = types.MethodType(Gr00tN1d6ActionHead_ACG.get_action_with_features, policy.model.action_head)
    policy.model.action_head.get_action = types.MethodType(Gr00tN1d6ActionHead_ACG.get_action, policy.model.action_head)

    return policy

def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")
    
    # Modify policy for ACG guidance if needed
    if config.use_acg_guidance:
        policy = modified_policy_for_acg(policy)

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
