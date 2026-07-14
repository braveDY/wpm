"""A1 RSL-RL 配置。"""

from importlib import import_module

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeA1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """A1 粗糙地形 PPO 配置。"""

    num_steps_per_env = 24
    obs_groups = {"actor": ["policy", "height_map"], "critic": ["policy", "height_map"]}
    max_iterations = 2500
    save_interval = 50
    experiment_name = "unitree_a1_rough"
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 64],
            kernel_size=[5, 3, 3],
            stride=[2, 1, 1],
            padding="zeros",
            norm=["batch", "batch", "batch"],
            activation="elu",
            global_pool="avg",
        ),
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 64],
            kernel_size=[5, 3, 3],
            stride=[2, 1, 1],
            padding="zeros",
            norm=["batch", "batch", "batch"],
            activation="elu",
            global_pool="avg",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=True,
    )

    def __post_init__(self):
        env_cfg = import_module("wmp.robots.a1.env_cfg")
        if env_cfg.FINETUNE:
            self.experiment_name = "unitree_a1_rough_finetune"
