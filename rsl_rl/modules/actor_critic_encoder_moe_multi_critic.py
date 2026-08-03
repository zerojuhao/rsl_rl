from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_encoder_moe import EncoderMoEActorCritic
from rsl_rl.networks import MLP


class IndependentCritics(nn.Module):
    """Independent scalar MLP value functions over a shared observation."""

    def __init__(
        self,
        input_dim: int,
        num_critics: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        super().__init__()
        if num_critics < 2:
            raise ValueError(
                "IndependentCritics requires at least two critics; "
                "use a standard actor-critic for one value function."
            )
        self.critics = nn.ModuleList(
            MLP(input_dim, 1, hidden_dims, activation)
            for _ in range(num_critics)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [critic(observations) for critic in self.critics], dim=-1
        )


class EncoderMoEActorMultiCritic(EncoderMoEActorCritic):
    """Encoder/MoE actor paired with independent MLP value functions."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        **kwargs: dict[str, Any],
    ) -> None:
        num_reward_heads = int(kwargs.pop("num_reward_heads", 1))
        if num_reward_heads < 2:
            raise ValueError(
                "EncoderMoEActorMultiCritic requires num_reward_heads >= 2."
            )
        self.num_reward_heads = num_reward_heads
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _build_critic(
        self,
        num_critic_obs: int,
        critic_hidden_dims: tuple[int] | list[int],
        activation: str,
    ) -> nn.Module:
        return IndependentCritics(
            input_dim=num_critic_obs,
            num_critics=self.num_reward_heads,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
