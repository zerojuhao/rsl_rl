# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_encoder_moe import EncoderMoEActorCritic
from rsl_rl.modules.moe import MoeLayer
from rsl_rl.modules.ssr_estimation import (
    SSREstimationModule,
    resolve_estimation_dims,
)
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
            MLP(input_dim, 1, hidden_dims, activation) for _ in range(num_critics)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat([critic(observations) for critic in self.critics], dim=-1)


class _SSRDeployActorOnnx(nn.Module):
    """Deployment graph: ``concat(flat_proprio, depth_latent) -> estimation -> MoE``."""

    def __init__(
        self,
        estimation: SSREstimationModule,
        actor: nn.Module,
        proprio_flat_dims: list[int],
        actor_obs_normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.estimation = estimation
        self.actor = actor
        self.proprio_flat_dims = list(proprio_flat_dims)
        self.proprio_total = int(sum(self.proprio_flat_dims))
        self.actor_obs_normalizer = actor_obs_normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proprio = x[..., : self.proprio_total]
        depth_latent = x[..., self.proprio_total :]
        terms: list[torch.Tensor] = []
        offset = 0
        for dim in self.proprio_flat_dims:
            terms.append(proprio[..., offset : offset + dim])
            offset += dim
        features, _, _ = self.estimation(
            terms,
            depth_latent,
            critic_group=None,
            store_aux_targets=False,
        )
        if self.actor_obs_normalizer is not None:
            features = self.actor_obs_normalizer(features)
        return self.actor(features)


class EncoderMoEActorMultiCritic(EncoderMoEActorCritic):
    """Encoder/MoE actor with independent critics and optional SSR estimation heads.

    With ``enable_ssr_estimation``, actor features are
    ``[proprio_feat, stopgrad(v_hat), stopgrad(f_hat), stopgrad(h_hat), depth_latent]``.
    """

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

        self.enable_ssr_estimation = bool(kwargs.pop("enable_ssr_estimation", False))
        self.proprio_history_length = int(kwargs.pop("proprio_history_length", 8))
        self.estimation_proprio_encoder_hidden_dims = list(
            kwargs.pop("estimation_proprio_encoder_hidden_dims", [512, 256])
        )
        proprio_latent_dim = kwargs.pop("estimation_proprio_latent_dim", None)
        self.estimation_proprio_latent_dim = (
            None if proprio_latent_dim is None else int(proprio_latent_dim)
        )
        # Legacy fusion-z bottleneck removed; heads read concat(proprio, depth).
        kwargs.pop("estimation_latent_dim", None)
        kwargs.pop("estimation_latent_head_dims", None)
        kwargs.pop("estimation_fusion_hidden_dims", None)
        self.estimation_estimator_hidden_dims = list(
            kwargs.pop("estimation_estimator_hidden_dims", [512, 256])
        )
        self.estimation_decoder_hidden_dims = list(
            kwargs.pop("estimation_decoder_hidden_dims", [256, 128])
        )
        self.estimation_actor_use_proprio_latent = bool(
            kwargs.pop("estimation_actor_use_proprio_latent", True)
        )
        self.enable_actor_foothold = bool(kwargs.pop("enable_actor_foothold", True))
        self.enable_actor_foot_height = bool(kwargs.pop("enable_actor_foot_height", True))
        self.estimation_foot_height_actor_dim = int(
            kwargs.pop("estimation_foot_height_actor_dim", 16)
        )
        self.gate_foothold_estimation_on_predictor_ready = bool(
            kwargs.pop("gate_foothold_estimation_on_predictor_ready", True)
        )
        # Legacy / removed flags.
        kwargs.pop("share_foothold_with_critic", None)
        kwargs.pop("aux_base_height_coef", None)
        kwargs.pop("aux_proprio_coef", None)
        self.aux_velocity_coef = float(kwargs.pop("aux_velocity_coef", 2.0))
        self.aux_foot_height_coef = float(kwargs.pop("aux_foot_height_coef", 1.0))
        self.aux_foothold_coef = float(kwargs.pop("aux_foothold_coef", 1.0))

        self._actor_hidden_dims = list(kwargs.get("actor_hidden_dims", [256, 128, 64]))
        self._activation_name = str(kwargs.get("activation", "elu"))
        self._num_actions = int(num_actions)

        super().__init__(obs, obs_groups, num_actions, **kwargs)

        self.estimation: SSREstimationModule | None = None
        if self.enable_ssr_estimation:
            self._build_ssr_estimation(obs)
            print(
                "SSR estimation enabled:"
                f" proprio_latent={self.estimation.dims.resolved_proprio_latent_dim},"
                f" fused={self.estimation.dims.fused_dim},"
                f" actor_foothold={self.enable_actor_foothold},"
                f" actor_foot_height={self.enable_actor_foot_height},"
                f" gate_foothold={self.gate_foothold_estimation_on_predictor_ready},"
                f" actor_feat={self.estimation.actor_feature_dim}"
            )
            if self.gate_foothold_estimation_on_predictor_ready:
                self.set_foothold_teacher_ready(False)

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

    def _build_actor(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int],
        activation: str,
    ) -> nn.Module:
        # Estimation mode rebuilds the MoE on estimated features after init.
        if self.enable_ssr_estimation:
            return nn.Identity()
        return super()._build_actor(
            num_actor_obs, num_actions, actor_hidden_dims, activation
        )

    def _build_ssr_estimation(self, obs: TensorDict) -> None:
        dims = resolve_estimation_dims(
            policy_group=obs[self.actor_obs_group],
            critic_group=obs[self.critic_obs_group],
            encoder_obs_names=self.actor_encoder_obs_groups,
            depth_latent_dim=self._encoder_output_size(
                self.actor_encoders, self.actor_encoder_obs_groups
            ),
            history_length=self.proprio_history_length,
            proprio_latent_dim=self.estimation_proprio_latent_dim,
            actor_use_proprio_latent=self.estimation_actor_use_proprio_latent,
            foot_height_actor_dim=self.estimation_foot_height_actor_dim,
        )
        self.estimation = SSREstimationModule(
            dims=dims,
            proprio_encoder_hidden_dims=self.estimation_proprio_encoder_hidden_dims,
            estimator_hidden_dims=self.estimation_estimator_hidden_dims,
            decoder_hidden_dims=self.estimation_decoder_hidden_dims,
            activation=self._activation_name,
            enable_actor_foothold=self.enable_actor_foothold,
            enable_actor_foot_height=self.enable_actor_foot_height,
        )
        actor_input_dim = self.estimation.actor_feature_dim
        self.actor = MoeLayer(
            actor_input_dim,
            self.num_moe_experts,
            self._num_actions,
            activation=self._activation_name,
            expert_hidden_dims=self._actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        if self.actor_obs_normalization:
            from rsl_rl.networks import EmpiricalNormalization

            self.actor_obs_normalizer = EmpiricalNormalization(actor_input_dim)
        print(f"SSR estimation actor network: {self.actor}")

    def _policy_proprio_terms(self, group_obs: TensorDict) -> list[torch.Tensor]:
        return [
            component_obs
            for component_name, component_obs in group_obs.items()
            if component_name not in self.actor_encoder_obs_groups
        ]

    def _encode_depth(self, group_obs: TensorDict) -> torch.Tensor:
        latents = [
            self.actor_encoders[name](group_obs[name])
            for name in self.actor_encoder_obs_groups
        ]
        return torch.cat(latents, dim=-1) if len(latents) > 1 else latents[0]

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        if self.estimation is None:
            return super().get_actor_obs(obs)

        group_obs = obs[self.actor_obs_group]
        critic_group = (
            obs[self.critic_obs_group] if self.critic_obs_group in obs.keys() else None
        )
        actor_features, _, _ = self.estimation(
            self._policy_proprio_terms(group_obs),
            self._encode_depth(group_obs),
            critic_group=critic_group,
            store_aux_targets=True,
        )
        return actor_features

    def set_foothold_teacher_ready(self, ready: bool) -> None:
        """Enable/disable SSR foothold aux based on privileged predictor readiness."""
        if self.estimation is None:
            return
        self.estimation.foothold_teacher_ready = bool(ready)

    def set_foothold_actor_mask(self, mask: torch.Tensor | None) -> None:
        if self.estimation is None:
            return
        self.estimation.set_foothold_actor_mask(mask)

    def set_foothold_teacher_target(self, teacher: torch.Tensor | None) -> None:
        """Forward minibatch foothold teacher into the estimation aux cache."""
        if self.estimation is None:
            return
        self.estimation.set_foothold_teacher_target(teacher)

    def get_aux_loss(self) -> torch.Tensor | None:
        loss, _ = self.get_aux_loss_and_metrics()
        return loss

    def get_aux_loss_and_metrics(self) -> tuple[torch.Tensor | None, dict[str, float]]:
        if self.estimation is None:
            return None, {}
        foothold_coef = self.aux_foothold_coef
        if (
            self.gate_foothold_estimation_on_predictor_ready
            and not self.estimation.foothold_teacher_ready
        ):
            foothold_coef = 0.0
        return self.estimation.aux_loss_and_metrics(
            velocity_coef=self.aux_velocity_coef,
            foot_height_coef=self.aux_foot_height_coef,
            foothold_coef=foothold_coef,
        )

    def export_as_onnx(self, obs: TensorDict, filedir: str) -> None:
        """Export depth encoder(s) and a deployable SSR actor graph.

        Actor ONNX input is ``concat(flat_proprio_history, depth_latent)`` and embeds
        estimation (``v̂`` / ``f̂`` / ``ĥ``) plus the MoE policy. Privileged foothold
        predictor / critic heads are not exported.
        """
        if self.estimation is None:
            super().export_as_onnx(obs, filedir)
            return
        if self.state_dependent_std:
            raise NotImplementedError(
                "export_as_onnx does not support state_dependent_std=True."
            )

        self.eval()
        stems = self.encoder_onnx_stems or {}
        seq = self.encoder_onnx_sequential_idx
        with torch.no_grad():
            group_obs = obs[self.actor_obs_group]
            for component_name in self.actor_encoder_obs_groups:
                stem = stems.get(component_name, component_name)
                enc_in = group_obs[component_name]
                enc = self.actor_encoders[component_name]
                filename = f"{seq}-{stem}.onnx" if seq is not None else f"{stem}.onnx"
                out_path = os.path.join(filedir, filename)
                torch.onnx.export(
                    enc,
                    enc_in,
                    out_path,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=12,
                )
                print(f"Exported encoder '{component_name}' to {out_path}")

            proprio_terms = self._policy_proprio_terms(group_obs)
            proprio_flat_dims = [int(term.shape[-1]) for term in proprio_terms]
            flat_proprio = torch.cat(proprio_terms, dim=-1)
            depth_latent = self._encode_depth(group_obs)
            sample = torch.cat((flat_proprio, depth_latent), dim=-1)

            normalizer = (
                self.actor_obs_normalizer if self.actor_obs_normalization else None
            )
            deploy = _SSRDeployActorOnnx(
                self.estimation,
                self.actor,
                proprio_flat_dims,
                actor_obs_normalizer=normalizer,
            )
            actor_path = os.path.join(filedir, self.actor_onnx_filename)
            torch.onnx.export(
                deploy,
                sample,
                actor_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
            print(
                f"Exported SSR actor (estimation+MoE) to {actor_path} "
                f"(input_dim={sample.shape[-1]}, proprio={sum(proprio_flat_dims)}, "
                f"depth_latent={depth_latent.shape[-1]})"
            )
