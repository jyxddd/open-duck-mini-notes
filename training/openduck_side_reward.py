"""od24: side-step technique reward (the user's idea: tell the duck HOW to side-step).

For a side command (|vy| > vy_threshold), every foot that is in the air is paid for
how far it has moved, since it lifted off, in the commanded side direction (in the
body's heading frame). Moving the swing foot the wrong way costs the same amount.
Dense (every control step while swinging), clipped at +-max_disp, so the most a
swing can earn is 1 per step. Lift height is already shaped by foot_clearance /
foot_swing_height (3 cm).
Feet: contact sensor order is LEFT, RIGHT; sites are resolved with preserve_order.
"""
from __future__ import annotations

import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)


class side_step_technique:
    def __init__(self, cfg: RewardTermCfg, env):
        n = env.num_envs
        self.liftoff = torch.zeros((n, 2, 3), device=env.device)
        self.valid = torch.zeros((n, 2), device=env.device, dtype=torch.bool)
        self.step_dt = env.step_dt

    def reset(self, env_ids=None):
        if env_ids is None:
            self.valid[:] = False
        else:
            self.valid[env_ids] = False

    def __call__(self, env, sensor_name: str, command_name: str, vy_threshold: float = 0.05,
                 max_disp: float = 0.04, asset_cfg: SceneEntityCfg = FEET) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        contact = env.scene[sensor_name]
        cmd = env.command_manager.get_command(command_name)
        vy = cmd[:, 1]
        pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
        first_air = contact.compute_first_air(dt=self.step_dt)  # [B, 2]
        self.liftoff = torch.where(first_air.unsqueeze(-1), pos, self.liftoff)
        self.valid = self.valid | first_air
        in_air = (contact.data.found == 0) & self.valid
        q = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1)
        disp_b = quat_apply_inverse(q, pos - self.liftoff)  # heading frame
        side = torch.clamp(disp_b[..., 1] * torch.sign(vy).unsqueeze(1), -max_disp, max_disp) / max_disp
        active = (vy.abs() > vy_threshold).float()
        r = torch.sum(side * in_air.float(), dim=1) * active
        env.extras["log"]["Metrics/side_step_disp"] = torch.sum(side * in_air.float() * active.unsqueeze(1)) / torch.clamp(
            torch.sum(in_air.float() * active.unsqueeze(1)), min=1)
        return r


class back_step_technique(side_step_technique):
    """od25 (user's backward technique + human backward-gait studies: SHORT steps).
    For a backward command (vx < -vx_threshold), each swinging foot is paid for moving
    BACKWARD since lift-off (heading frame): full pay at `full` (2.5 cm), nothing extra
    beyond, and a penalty for over-long steps past `too_long` (4 cm). Forward = negative."""

    def __call__(self, env, sensor_name: str, command_name: str, vx_threshold: float = 0.05,
                 full: float = 0.025, too_long: float = 0.04, asset_cfg: SceneEntityCfg = FEET) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        contact = env.scene[sensor_name]
        cmd = env.command_manager.get_command(command_name)
        vx = cmd[:, 0]
        pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        first_air = contact.compute_first_air(dt=self.step_dt)
        self.liftoff = torch.where(first_air.unsqueeze(-1), pos, self.liftoff)
        self.valid = self.valid | first_air
        in_air = (contact.data.found == 0) & self.valid
        q = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1)
        back = -quat_apply_inverse(q, pos - self.liftoff)[..., 0]          # + = moved backward
        score = torch.clamp(back, -full, full) / full - torch.clamp(back - too_long, min=0.0) / 0.02
        active = (vx < -vx_threshold).float()
        r = torch.sum(score * in_air.float(), dim=1) * active
        env.extras["log"]["Metrics/back_step_disp"] = torch.sum(back * in_air.float() * active.unsqueeze(1)) / torch.clamp(
            torch.sum(in_air.float() * active.unsqueeze(1)), min=1)
        return r
