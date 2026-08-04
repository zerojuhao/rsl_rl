from __future__ import annotations

import torch
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.modules import AMPDiscriminator
from rsl_rl.modules.amp import LossType
from rsl_rl.storage import CircularBuffer, RolloutStorage


class MultiRewardPPOAMP(PPOAMP):
    """PPOAMP variant with separate task/style reward and value streams.

    The legacy :class:`PPOAMP` remains a scalar-reward algorithm. This subclass
    initializes the multi-head PPO state directly and only overrides reward
    composition; it inherits the established AMP discriminator update.
    """

    def __init__(
        self,
        policy,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        amp_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        num_reward_heads: int = 3,
        advantage_weights: list[float] | tuple[float, ...] | None = None,
        reward_head_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if num_reward_heads < 2:
            raise ValueError(
                "MultiRewardPPOAMP requires at least two reward heads; "
                "use PPOAMP for scalar AMP training."
            )
        if reward_head_names is None and num_reward_heads == 3:
            # SSR default critic-head order: locomotion, foothold, style.
            reward_head_names = ["locomotion", "foothold", "style"]

        PPO.__init__(
            self,
            policy=policy,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
            num_reward_heads=num_reward_heads,
            advantage_weights=advantage_weights,
            reward_head_names=reward_head_names,
        )

        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError(
                "AMP configuration must be provided for MultiRewardPPOAMP."
            )

        try:
            self.loss_type = LossType[self.amp_cfg["loss_type"]]
        except KeyError as error:
            raise ValueError(
                f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. "
                "Expected 'GAN', 'LSGAN', or 'WGAN'."
            ) from error

        self.amp_discriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            device=device,
            **self.amp_cfg.get("amp_discriminator", {}),
        ).to(self.device)

        discriminator_parameters = [
            {
                "name": "disc_trunk",
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],
            },
        ]
        self.disc_optimizer = optim.Adam(
            discriminator_parameters,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)
        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(
            obs, flatten_history_dim=False
        )
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(
            obs, flatten_history_dim=False
        )
        self.style_rewards, self.disc_score = (
            self.amp_discriminator.predict_style_reward(
                disc_obs, dt=self.amp_cfg["step_dt"]
            )
        )

        expected_task_heads = self.num_reward_heads - 1
        if rewards.ndim != 2 or rewards.shape[-1] != expected_task_heads:
            raise ValueError(
                "MultiRewardPPOAMP expects one environment reward per non-style "
                f"head: expected [num_envs, {expected_task_heads}], "
                f"got {tuple(rewards.shape)}."
            )
        training_rewards = torch.cat(
            (rewards, self.style_rewards.unsqueeze(-1)), dim=-1
        )

        # Scalar summaries are for the existing AMP logger only. Training uses
        # the unscaled reward heads; PPO.compute_returns() normalizes each head
        # then weight-sums advantages (HoST/SSR multi-critic).
        task_weights = self.advantage_weights[:-1]
        self.task_rewards = (rewards * task_weights).sum(dim=-1)
        self.rewards_lerp = (
            training_rewards * self.advantage_weights
        ).sum(dim=-1)

        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        PPO.process_env_step(self, obs, training_rewards, dones, extras)
