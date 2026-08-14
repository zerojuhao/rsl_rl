from __future__ import annotations

import statistics
import time
import torch
from collections import deque
from typing import Any

import rsl_rl

from rsl_rl.utils.logger import Logger


class LoggerAMP(Logger):
    """Logger class for AMP runners and algorithms."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
        max_episode_length_s: float,
    ) -> None:
        super().__init__(
            log_dir,
            cfg,
            env_cfg,
            num_envs,
            is_distributed,
            gpu_world_size,
            gpu_global_rank,
            device,
        )

        # Create buffers for logging AMP rewards and other info
        self.total_rewbuffer = deque(maxlen=100)
        self.style_rewbuffer = deque(maxlen=100)
        self.cur_total_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_style_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.max_episode_length_s = max_episode_length_s

        # Per-sub-terrain episode length / curriculum (see ``configure_terrain_episode_length``).
        self._terrain_env: Any | None = None
        self._terrain_names: list[str] = []
        self.terrain_lenbuffers: dict[str, deque] = {}
        self._ep_terrain_ids: torch.Tensor | None = None

    def configure_terrain_episode_length(self, env: Any) -> None:
        """Enable per-sub-terrain episode-length and curriculum-level logging."""
        unwrapped = getattr(env, "unwrapped", env)
        terrain = getattr(getattr(unwrapped, "scene", None), "terrain", None)
        if terrain is None:
            return
        if getattr(terrain.cfg, "terrain_type", None) != "generator":
            return
        gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
        terrain_gen = getattr(terrain, "terrain_generator", None)
        if gen_cfg is None or terrain_gen is None:
            return
        if not hasattr(terrain_gen, "get_subterrain_indices"):
            return
        names = list(gen_cfg.sub_terrains.keys())
        if not names:
            return
        self._terrain_env = unwrapped
        self._terrain_names = names
        self.terrain_lenbuffers = {name: deque(maxlen=100) for name in names}
        self._ep_terrain_ids = self._fetch_subterrain_ids()
        print(
            "[INFO] Terrain logging enabled for "
            f"{len(names)} sub-terrains "
            "(Terrain/mean_episode_length/*, TerrainCurriculum/mean_level/*, "
            f"Actor_MoE_Terrain/*): {', '.join(names)}"
        )

    def _fetch_subterrain_ids(self) -> torch.Tensor | None:
        if self._terrain_env is None or not self._terrain_names:
            return None
        terrain = self._terrain_env.scene.terrain
        return terrain.terrain_generator.get_subterrain_indices(
            terrain.terrain_levels,
            terrain.terrain_types,
            device=self.device,
        )

    def terrain_moe_log_context(self) -> tuple[list[str], torch.Tensor | None]:
        """Return ``(sub_terrain_names, per-env subterrain ids)`` for MoE logging."""
        if not self._terrain_names:
            return [], None
        return self._terrain_names, self._fetch_subterrain_ids()

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
        style_rewards: torch.Tensor | None = None,
        total_rewards: torch.Tensor | None = None,
    ) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.log_dir is not None:
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])

            # Update rewards and episode length
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            if style_rewards is not None:
                self.cur_style_reward_sum += style_rewards
            if total_rewards is not None:
                self.cur_total_reward_sum += total_rewards
            self.cur_episode_length += 1

            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            # Per-terrain lengths use cached ids from the finished episode (before refresh).
            if len(new_ids) > 0 and self._ep_terrain_ids is not None and self._terrain_names:
                done_env_ids = new_ids[:, 0]
                lengths = self.cur_episode_length[done_env_ids]
                terrain_ids = self._ep_terrain_ids[done_env_ids]
                for length, terrain_id in zip(lengths.tolist(), terrain_ids.tolist()):
                    tid = int(terrain_id)
                    if 0 <= tid < len(self._terrain_names):
                        self.terrain_lenbuffers[self._terrain_names[tid]].append(float(length))

            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0
            if style_rewards is not None and total_rewards is not None:
                amp_new_ids = new_ids if len(new_ids) > 0 else slice(None)
                style_rew_episode_mean = torch.mean(self.cur_style_reward_sum[amp_new_ids]) / (
                    self.max_episode_length_s
                )
                if len(new_ids) > 0:
                    self.ep_extras[-1]["Episode_Reward/style"] = style_rew_episode_mean.item()
                self.total_rewbuffer.extend(self.cur_total_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.style_rewbuffer.extend(self.cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_style_reward_sum[new_ids] = 0
                self.cur_total_reward_sum[new_ids] = 0

            # After env.step reset, refresh terrain ids only for finished envs.
            if len(new_ids) > 0 and self._ep_terrain_ids is not None and self._terrain_names:
                done_env_ids = new_ids[:, 0]
                terrain = self._terrain_env.scene.terrain
                self._ep_terrain_ids[done_env_ids] = terrain.terrain_generator.get_subterrain_indices(
                    terrain.terrain_levels[done_env_ids],
                    terrain.terrain_types[done_env_ids],
                    device=self.device,
                )

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        metrics_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console."""
        if self.log_dir is not None and not self.disable_logs:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            # Log episode extras
            extras_string = ""
            if self.ep_extras:
                # Iterate over all keys in the episode info dictionary
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    # Iterate over all steps
                    for ep_info in self.ep_extras:
                        # Handle missing, scalar, and zero dimensional tensors
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)
                        extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

            # Log losses
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            for key, value in metrics_dict.items():
                self.writer.add_scalar(key, value, it)
            self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

            # Log noise std
            self.writer.add_scalar("Policy/mean_noise_std", action_std.mean().item(), it)

            # Log performance
            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            # Log rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)
                self.writer.add_scalar("AMP/mean_total_reward", statistics.mean(self.total_rewbuffer), it)
                self.writer.add_scalar("AMP/mean_style_reward", statistics.mean(self.style_rewbuffer), it)
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                self.writer.add_scalar("Train/max_episode_length", max(self.lenbuffer), it)

            # Per-sub-terrain mean episode length (0 when no finished episodes in the window).
            for name in self._terrain_names:
                buf = self.terrain_lenbuffers[name]
                value = statistics.mean(buf) if len(buf) > 0 else 0.0
                self.writer.add_scalar(f"Terrain/mean_episode_length/{name}", value, it)

            # Per-sub-terrain mean curriculum level (current env assignment; 0 if unused).
            if self._terrain_env is not None and self._terrain_names:
                terrain = self._terrain_env.scene.terrain
                levels = terrain.terrain_levels.float()
                terrain_ids = terrain.terrain_generator.get_subterrain_indices(
                    terrain.terrain_levels,
                    terrain.terrain_types,
                    device=self.device,
                )
                for tid, name in enumerate(self._terrain_names):
                    mask = terrain_ids == tid
                    value = float(levels[mask].mean().item()) if bool(mask.any()) else 0.0
                    self.writer.add_scalar(f"TerrainCurriculum/mean_level/{name}", value, it)

            # Print to console
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            # Print run name if provided
            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            # Print performance
            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )

            # Print losses
            for key, value in loss_dict.items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

            # Print key estimation / foothold diagnostics (already written to the writer).
            for key in (
                "Estimation/velocity_rmse",
                "Estimation/foothold_xy_rmse",
                "Estimation/foothold_yaw_rmse",
                "Estimation/foot_height_rmse",
                "Foothold/Accuracy/xy_rmse_m",
                "Foothold/Accuracy/xy_rmse_near_m",
                "Foothold/Accuracy/x_rmse_m",
                "Foothold/Accuracy/y_rmse_m",
                "Foothold/Accuracy/x_bias_m",
                "Foothold/Accuracy/y_bias_m",
                "Foothold/Accuracy/yaw_rmse_rad",
                "Foothold/Accuracy/yaw_bias_rad",
                "Foothold/Distribution/sigma_mean_m",
                "Foothold/Horizon/h0_3/xy_rmse_m",
                "Foothold/Horizon/h16p/xy_rmse_m",
                "Foothold/Terrain/stairs_down/xy_rmse_m",
                "Foothold/Terrain/stairs_up/xy_rmse_m",
                "Foothold/Contact/support_ratio",
                "Foothold/Guidance/reward_enabled",
                "Foothold/Guidance/inv_stairs_level",
                "Foothold/Guidance/eval_sigma_m",
                "Foothold/Guidance/horizon_weight",
                "Foothold/Guidance/reward",
                "Foothold/Guidance/swing_deficiency",
            ):
                if key in metrics_dict:
                    log_string += f"""{f"{key}:":>{pad}} {metrics_dict[key]:.4f}\n"""

            # Print rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{"Mean AMP total reward:":>{pad}} {statistics.mean(self.total_rewbuffer):.2f}\n"""
                log_string += f"""{"Mean AMP style reward:":>{pad}} {statistics.mean(self.style_rewbuffer):.2f}\n"""
                log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            # Print noise std
            log_string += f"""{"Mean action noise std:":>{pad}} {action_std.mean().item():.2f}\n"""

            # Print episode extras
            if not print_minimal:
                log_string += extras_string

            # Print footer
            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            # Clear extras buffer
            self.ep_extras.clear()
