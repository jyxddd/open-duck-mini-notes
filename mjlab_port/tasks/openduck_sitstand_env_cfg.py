"""Open Duck commanded sit <-> stand on the Microduck sitstand recipe.

Same posture rewards; the pose, heights, clip box, IMU lag, obs noise, servo kp
DR and action rate come from the Open Duck velocity task.

SIT is a feet-flat squat (10-31). sit1 used a kneeling sit (shins on the floor,
z 0.103): a plain 3 s ramp from it to HOME tips over backward at 1.9 N.m and after
3000 iterations the policy still could not rise. This squat ramps to HOME and back
with tilt <= 3 deg at most ~0.7 N.m in the sim.
"""

from dataclasses import replace

from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.openduck_constants import JOINT_TARGET_LIMITS, OPENDUCK_GROUND_ROBOT_CFG
from mjlab_microduck.tasks import microduck_sitstand_env_cfg as ss
from mjlab_microduck.tasks.openduck_velocity_env_cfg import (
    ACTION_RATE_SCALE,
    HEAD_BODY_NAMES,
    IMU_LAG,
    KP_RANGE,
    OBS_NOISE,
    randomize_servo_kp,
    target_clip_excess,
)

SIT_ACTION_RATE_SCALE = 2 * ACTION_RATE_SCALE
SIT_Z, STAND_Z = 0.137, 0.162  # root z settled in the sim, full model
SIT_OVERRIDES = {2: -0.95, 3: 1.75, 4: -0.8, 11: 0.95, 12: 1.75, 13: -0.8}
MODULE_OVERRIDES = {
    "MICRODUCK_STANDUP_ROBOT_CFG": OPENDUCK_GROUND_ROBOT_CFG,
    "SITTING_TARGET_OVERRIDES": SIT_OVERRIDES,
    "SIT_Z": SIT_Z,
    "STAND_Z": STAND_Z,
    "SIT_UPRIGHT_Z": SIT_Z + 0.005,
    "STAND_UPRIGHT_Z": STAND_Z - 0.007,
}


def make_openduck_sitstand_env_cfg(play: bool = False, rough: bool = False):
    saved = {k: getattr(ss, k) for k in MODULE_OVERRIDES}
    for k, v in MODULE_OVERRIDES.items():
        setattr(ss, k, v)
    try:
        cfg = ss.make_microduck_sitstand_env_cfg(play=play, rough=rough)
    finally:
        for k, v in saved.items():
            setattr(ss, k, v)

    cfg.actions["joint_pos"].clip = dict(JOINT_TARGET_LIMITS)
    cfg.rewards["target_clip_excess"] = RewardTermCfg(func=target_clip_excess, weight=-1.0)
    cfg.rewards["rise_bootstrap"].params["max_height"] = STAND_Z + 0.01

    cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES)
    cfg.events["reset_base"].params["pose_range"]["z"] = (STAND_Z, STAND_Z + 0.01)
    p = cfg.events["set_ground_state"].params
    p["sitting_z_min"], p["sitting_z_max"] = SIT_Z, SIT_Z + 0.015
    p["standing_z_min"], p["standing_z_max"] = STAND_Z, STAND_Z + 0.01
    cfg.events["randomize_motor_gains"] = EventTermCfg(
        func=randomize_servo_kp, mode="reset", params={"kp_range": KP_RANGE}
    )

    actor = cfg.observations["actor"].terms
    actor["base_ang_vel"].delay_min_lag, actor["base_ang_vel"].delay_max_lag = IMU_LAG
    actor["projected_gravity"].delay_min_lag, actor["projected_gravity"].delay_max_lag = IMU_LAG
    actor["base_ang_vel"].noise = Unoise(n_min=-OBS_NOISE["base_ang_vel"], n_max=OBS_NOISE["base_ang_vel"])
    actor["projected_gravity"].noise = Unoise(n_min=-OBS_NOISE["raw_accelerometer"], n_max=OBS_NOISE["raw_accelerometer"])

    # sit3 (warm start from sit2 model_2500): twice the walk scale. sit2 buzzed on the
    # real floor (gyro std 0.08, walk od11 0.001) and hardly squatted.
    cfg.rewards["action_rate_l2"].weight *= SIT_ACTION_RATE_SCALE
    if "action_rate_weight" in cfg.curriculum:
        for stage in cfg.curriculum["action_rate_weight"].params["weight_stages"]:
            stage["weight"] *= SIT_ACTION_RATE_SCALE
    return cfg


OpenDuckSitStandRlCfg = replace(
    ss.MicroduckSitStandRlCfg,
    experiment_name="openduck_sitstand",
    run_name="openduck_sitstand",
    wandb_project="mjlab_openduck",
    max_iterations=6000,
)
