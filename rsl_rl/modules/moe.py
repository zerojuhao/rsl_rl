# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from tensordict import TensorDict


class MoeLayer(nn.Module):
    """Soft mixture-of-experts: gate mixes expert MLP outputs."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        output_dim: int,
        *,
        activation: str = "elu",
        expert_hidden_dims: list[int] | None = None,
        gate_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        expert_hidden_dims = list(expert_hidden_dims or [])
        gate_hidden_dims = list(gate_hidden_dims or [])
        self._activation_name = activation
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
        self.experts = nn.ModuleList(
            [
                self._build_expert(input_dim, output_dim, expert_hidden_dims)
                for _ in range(num_experts)
            ]
        )

    def _build_gate(self, input_dim: int, num_experts: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(resolve_nn_activation(self._activation_name))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, num_experts))
        return nn.Sequential(*layers)

    def _build_expert(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(resolve_nn_activation(self._activation_name))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=-1)

    @torch.no_grad()
    def gate_stats(self, x: torch.Tensor) -> dict[str, float]:
        gate_scores = self.gate_scores(x)
        mean_weights = gate_scores.mean(dim=0)
        entropy = -(gate_scores * (gate_scores + 1e-8).log()).sum(dim=-1).mean()
        stats = {f"expert_{i}": mean_weights[i].item() for i in range(gate_scores.shape[-1])}
        stats["gate_entropy"] = entropy.item()
        stats["max_weight"] = gate_scores.max(dim=-1).values.mean().item()
        return stats

    @torch.no_grad()
    def gate_stats_per_group(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor,
        group_names: list[str],
        *,
        prefix: str = "Actor_MoE_Terrain",
    ) -> dict[str, float]:
        """Mean gate weights for each named group (0 if a group has no samples)."""
        gate_scores = self.gate_scores(x)
        num_experts = gate_scores.shape[-1]
        stats: dict[str, float] = {}
        for gid, name in enumerate(group_names):
            mask = group_ids == gid
            if bool(mask.any()):
                mean_weights = gate_scores[mask].mean(dim=0)
                for i in range(num_experts):
                    stats[f"{prefix}/{name}/expert_{i}"] = mean_weights[i].item()
            else:
                for i in range(num_experts):
                    stats[f"{prefix}/{name}/expert_{i}"] = 0.0
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate_scores(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", gate_scores, expert_outputs)


def collect_actor_moe_gate_log(
    policy,
    obs: TensorDict,
    *,
    terrain_ids: torch.Tensor | None = None,
    terrain_names: list[str] | None = None,
) -> dict[str, float]:
    """Return actor MoE gate statistics for logging (empty dict if policy is not MoE).

    When ``terrain_ids`` / ``terrain_names`` are provided, also emits
    ``Actor_MoE_Terrain/<sub_terrain>/expert_<i>``.
    """
    actor = getattr(policy, "actor", None)
    if not isinstance(actor, MoeLayer):
        return {}
    if not hasattr(policy, "get_actor_obs"):
        return {}
    with torch.no_grad():
        actor_obs = policy.get_actor_obs(obs)
        if hasattr(policy, "actor_obs_normalizer"):
            actor_obs = policy.actor_obs_normalizer(actor_obs)
        stats = {f"Actor_MoE/{key}": value for key, value in actor.gate_stats(actor_obs).items()}
        if terrain_ids is not None and terrain_names:
            if terrain_ids.shape[0] != actor_obs.shape[0]:
                raise ValueError(
                    "terrain_ids batch size must match actor obs: "
                    f"{terrain_ids.shape[0]} != {actor_obs.shape[0]}."
                )
            stats.update(
                actor.gate_stats_per_group(
                    actor_obs,
                    terrain_ids,
                    terrain_names,
                    prefix="Actor_MoE_Terrain",
                )
            )
    return stats
