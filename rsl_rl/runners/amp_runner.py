from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import MultiRewardPPOAMP, PPO, PPOAMP
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    EncoderActorCritic,
    EncoderMoEActorCritic,
    EncoderMoEActorMultiCritic,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
    resolve_amp_config,
)
from rsl_rl.modules.moe import collect_actor_moe_gate_log
from rsl_rl.modules.ssr_estimation import FOOTHOLD_TEACHER_DIM
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.amp_logger import LoggerAMP
from rsl_rl.runners import OnPolicyRunner


class AMPRunner(OnPolicyRunner):
    
    alg: PPOAMP

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        
        self._foothold_env = getattr(self.env, "unwrapped", self.env)
        foothold_cfg = self.cfg.get("foothold_imagination")
        foothold_enabled = (
            foothold_cfg.get("enabled", False)
            if isinstance(foothold_cfg, dict)
            else bool(getattr(foothold_cfg, "enabled", False))
        )
        if foothold_enabled:
            configure = getattr(
                self._foothold_env, "configure_foothold_imagination", None
            )
            if configure is None:
                raise RuntimeError(
                    "The environment does not support foothold imagination configuration."
                )
            configure(foothold_cfg)
        self._foothold_guidance = getattr(self._foothold_env, "foothold_guidance", None)
        self._sync_foothold_estimation_gate()

        self.logger = LoggerAMP(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            max_episode_length_s=(self.env.max_episode_length*self.env.unwrapped.step_dt),
        )
        self.logger.configure_terrain_episode_length(self.env)

    def prepare_foothold_prediction_step(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        enable_inference_guidance: bool = False,
    ) -> None:
        """Run the privileged foothold teacher and store its pose on the transition."""
        if self._foothold_guidance is None:
            return
        applied_actions = actions
        clip_actions = getattr(self.env, "clip_actions", None)
        if clip_actions is not None:
            applied_actions = torch.clamp(applied_actions, -clip_actions, clip_actions)
        self._foothold_env.prepare_foothold_prediction_step(
            obs["critic"],
            applied_actions,
            enable_inference_guidance=enable_inference_guidance,
        )
        self._record_foothold_teacher()

    def _record_foothold_teacher(self) -> None:
        """Copy teacher XY+yaw into the current transition for Estimation aux."""
        if self._foothold_guidance is None:
            return
        transition = getattr(getattr(self, "alg", None), "transition", None)
        if transition is None:
            return
        teacher = self._foothold_env.foothold_guidance.latest_mu_flat()
        transition.foothold_teacher_xy = teacher.detach().clone()

    def _sync_foothold_estimation_gate(self) -> None:
        """Gate SSR foothold aux on privileged predictor ``reward_enabled``."""
        policy = getattr(getattr(self, "alg", None), "policy", None)
        if policy is None or not hasattr(policy, "set_foothold_teacher_ready"):
            return
        if not getattr(policy, "gate_foothold_estimation_on_predictor_ready", False):
            policy.set_foothold_teacher_ready(True)
            return
        ready = False
        guidance = self._foothold_guidance
        if guidance is not None:
            trainer = getattr(guidance, "trainer", None)
            if trainer is not None:
                ready = bool(trainer.reward_enabled)
        policy.set_foothold_teacher_ready(ready)

    def _apply_foothold_actor_mask(self) -> None:
        """Zero actor ``f_hat`` on non-stairs environments."""
        policy = getattr(getattr(self, "alg", None), "policy", None)
        if policy is None or not hasattr(policy, "set_foothold_actor_mask"):
            return
        mask = None
        if self._foothold_guidance is not None:
            mask = self._foothold_guidance.stairs_active_mask()
        policy.set_foothold_actor_mask(mask)
        transition = getattr(getattr(self, "alg", None), "transition", None)
        if transition is not None:
            transition.foothold_actor_active = (
                None if mask is None else mask.detach().clone()
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    self._apply_foothold_actor_mask()
                    actions = self.alg.act(obs)
                    self.prepare_foothold_prediction_step(obs, actions)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Extract AMP rewards (only for logging)
                    style_rewards = self.alg.style_rewards
                    total_rewards = self.alg.rewards_lerp
                    # Book keeping
                    self.logger.process_env_step(
                        getattr(self.alg, "task_rewards", rewards),
                        dones,
                        extras,
                        intrinsic_rewards,
                        style_rewards,
                        total_rewards,
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()
            self._apply_foothold_actor_mask()
            terrain_names, terrain_ids = self.logger.terrain_moe_log_context()
            metrics_dict = collect_actor_moe_gate_log(
                self.alg.policy,
                obs,
                terrain_ids=terrain_ids,
                terrain_names=terrain_names,
            )
            # Move Estimation/* diagnostics out of Loss/ so they share the Foothold/* namespace style.
            for key in list(loss_dict.keys()):
                if key.startswith("Estimation/"):
                    metrics_dict[key] = loss_dict.pop(key)
            if self._foothold_guidance is not None:
                metrics_dict.update(self._foothold_env.update_foothold_predictor())
                self._sync_foothold_estimation_gate()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                metrics_dict=metrics_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )
            
            # Save model (rank 0 only in distributed mode)
            if it % self.cfg["save_interval"] == 0 and not self.logger.disable_logs:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "num_reward_heads": self.alg.num_reward_heads,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Save AMP model
        saved_dict["amp_discriminator_state_dict"] = self.alg.amp_discriminator.state_dict()
        saved_dict["amp_discriminator_normalizer_state_dict"] = self.alg.amp_discriminator.disc_obs_normalizer.state_dict()
        saved_dict["amp_discriminator_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        if self._foothold_guidance is not None:
            saved_dict["foothold_guidance_state_dict"] = self._foothold_guidance.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        checkpoint_heads = int(loaded_dict.get("num_reward_heads", 1))
        if checkpoint_heads != self.alg.num_reward_heads:
            raise RuntimeError(
                "Checkpoint critic head count is incompatible with this run: "
                f"{checkpoint_heads} != {self.alg.num_reward_heads}. "
                "Multi-critic training must start from a compatible checkpoint."
            )
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load AMP model (tolerates disc obs-layout drift for play/export).
        try:
            self.alg.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])
            self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
                loaded_dict["amp_discriminator_normalizer_state_dict"]
            )
        except RuntimeError as exc:
            print(
                "[WARN] Skipping AMP discriminator load due to incompatible shapes "
                f"(policy still loaded): {exc}"
            )
        if self._foothold_guidance is not None and "foothold_guidance_state_dict" in loaded_dict:
            self._foothold_guidance.load_state_dict(
                loaded_dict["foothold_guidance_state_dict"],
                load_optimizer=load_optimizer,
            )
        self._sync_foothold_estimation_gate()
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # AMP discriminator optimizer
            self.alg.disc_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        super().train_mode()
        self.alg.amp_discriminator.train()
        self.alg.amp_discriminator.disc_obs_normalizer.train()
        
    def eval_mode(self):
        super().eval_mode()
        self.alg.amp_discriminator.eval()
        self.alg.amp_discriminator.disc_obs_normalizer.eval()
    
    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        
        # Resolve AMP config
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        num_reward_heads = int(self.alg_cfg.get("num_reward_heads", 1))
        if num_reward_heads > 1:
            policy_num_reward_heads = int(
                self.policy_cfg.get("num_reward_heads", num_reward_heads)
            )
            if policy_num_reward_heads != num_reward_heads:
                raise ValueError(
                    "Policy and algorithm disagree on num_reward_heads: "
                    f"{policy_num_reward_heads} != {num_reward_heads}."
                )
            self.policy_cfg["num_reward_heads"] = num_reward_heads

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN | EncoderActorCritic | EncoderMoEActorCritic | EncoderMoEActorMultiCritic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        foothold_cfg = self.cfg.get("foothold_imagination")
        foothold_enabled = (
            foothold_cfg.get("enabled", False)
            if isinstance(foothold_cfg, dict)
            else bool(getattr(foothold_cfg, "enabled", False))
        )
        foothold_teacher_dim = FOOTHOLD_TEACHER_DIM if foothold_enabled else None

        # Initialize the storage
        storage = RolloutStorage(
            "rl",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            self.device,
            num_reward_heads=num_reward_heads,
            foothold_teacher_dim=foothold_teacher_dim,
        )
        
        # Initialize AMP discriminator observation buffers
        disc_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPOAMP = alg_class(
            actor_critic, storage, disc_obs_buffer, disc_demo_obs_buffer, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
