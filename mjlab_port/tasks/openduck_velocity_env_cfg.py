"""Open Duck Mini v2 velocity (walking) environment.

The Microduck velocity recipe (BAM actuators, voltage / delay / friction /
armature / CoM / IMU DR, pushes, turn-in-place bucket, standing curriculum,
head-pose commands, 61D obs) on the Open Duck model with STS3215 servos.

Differences from Microduck, all for the bigger, heavier, weaker duck:
  - raw accelerometer instead of projected gravity (the duck's runtime reads
    BNO055 accel; same unit-vector convention as projected gravity)
  - speeds: lin x -0.15..0.3, y +-0.15 (Microduck 0.4 / 0.3)
  - head pose caps sized to the Heima head (a 0.4 rad head step tipped it over)
  - foot clearance / swing target 3 cm, reset height 0.15 m
  - position targets clipped to JOINT_TARGET_LIMITS, a box around HOME clear
    of every joint stop, plus a penalty for asking beyond it; the sim knee stop
    widened (od1 carried its weight on the XML knee stop, the real knee has
    none; od2, clipped at the stops, splayed its legs against them)
  - left-right mirror loss (od1 drifted sideways ~4 cm/s while walking)
  - od11: ONE change on top of od10 (warm start from od10 model_11999): action
    rate penalty x2. od10 walks well but its standing target chatter in the sim
    went from 0.5 to 0.9-2.0 deg per step, od8 9000 stays at 0.4.
  - od10: ONE change on top of od9 (warm start from od9 model_10000): model CoM
    12 mm further back, measured on the real duck (robot/openduck_constants.py).
    od9's gyro noise stays although it did not reduce the buzz in the sim exam.
  - od9: ONE change on top of od8 (warm start from od8 model_9000): gyro noise
    +-0.1 instead of +-0.03. od8 stood on the real floor but buzzed: the real
    gyro jitters ~0.1 rad/s between steps while standing and the policy
    amplifies it into ~0.9 deg per step of target chatter (same in the sim
    with +-0.1 gyro noise).
  - od8: ONE change on top of od7 (warm start from od7 model_8000; later od7
    checkpoints drifted in yaw / stopped starting at small lag): servo kp DR
    0.75-1.3 instead of 1.0 (see KP_RANGE).
  - od7: ONE change on top of od6 (warm start from od6 model_7000): IMU lag
    0-5 control steps instead of 2-5 (see IMU_LAG).
  - od6: ONE change on top of od3 (warm start from od3): IMU lag 2-5 control
    steps. The gait / noise / kp DR / action-rate knobs below
    stay at od3 values; od4 (four changes from scratch) never stepped, od5
    (slower gait) walked but did not address the real-floor fall.
    Knobs: rougher actor obs noise, servo kp DR, stronger action-rate penalty,
    slower gait (air-time window 0.2-0.4 s; Microduck's 0.125-0.3 gave ~5
    touch-downs/s, the official Open Duck policy steps ~3/s on the real duck).
    od3 stood still in every sim test but on the real floor it amplified the
    ~4-5 Hz start-up body shake (the official policy damps it) and fell;
    Microduck's tiny noise lets the policy lean hard on gyro feedback.
"""

from dataclasses import replace

import torch
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.openduck_constants import JOINT_TARGET_LIMITS, OPENDUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import microduck_velocity_env_cfg as base
from mjlab_microduck.tasks.openduck_symmetry import SYMMETRY_CFG

HEAD_BODY_NAMES = ("neck_pitch_assembly", "head_pitch_to_yaw", "neck_yaw_assembly", "head_assembly")
LIN_VEL_X = (-0.15, 0.3)
LIN_VEL_Y = (-0.15, 0.15)
ANG_VEL_Z = (-1.0, 1.0)
# Final head-pose caps (deltas from HOME): neck_pitch, head_pitch, head_yaw, head_roll
HEAD_CAPS = (0.3, 0.5, 0.8, 0.3)
FOOT_TARGET_HEIGHT = 0.03
NOMINAL_HEIGHT = 0.15
# Actor obs noise (uniform +-): Microduck's narrowed values were 0.03 / 0.01 /
# 0.001 / 0.25, its originals 0.2 / 0.15 / 0.05 / 2.0.
OBS_NOISE = {"base_ang_vel": 0.1, "raw_accelerometer": 0.01, "joint_pos": 0.001, "joint_vel": 0.25}
KP_RANGE = (0.75, 1.3)  # firmware P scale per episode (legs kp_fw 20 -> 15..26)
AIR_TIME_WINDOW = (0.125, 0.3)  # s, rewarded swing duration (Microduck; od5 tried 0.2-0.4)
# IMU (gyro + accelerometer) lag in 20 ms control steps. Microduck assumes a
# fast IMU (0-1); on the real Open Duck the BNO055 path (NDOF mode, runtime
# reader thread with a 1-slot queue, bit-banged I2C) lags the body by 3-4
# steps: open-loop replays of real floor starts match the sim only with the
# real gyro shifted 3-4 steps (joints match at 0). With 3-4 steps od3 falls
# in the sim exactly like on the real floor.
# od6 (2-5) walks at 3-6 steps of lag but stands still at 0-2, and the
# runtime fresh_imu reader will cut the real lag to an unknown 0-2: od7 = 0-5.
IMU_LAG = (0, 5)
ACTION_RATE_SCALE = 2.0  # x the Microduck action_rate_l2 curriculum


def target_clip_excess(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared amount (rad^2, summed over joints) by which the policy asks for
    targets beyond the clip box: without it the policy can park joints on the
    clip and bang-bang the servos, which the real duck would not forgive."""
    term = env.action_manager.get_term(action_name)
    target = term._raw_actions * term._scale + term._offset
    lo, hi = term._clip[..., 0], term._clip[..., 1]
    excess = (target - hi).clamp(min=0) + (lo - target).clamp(min=0)
    return (excess**2).sum(dim=-1)


def randomize_servo_kp(env, env_ids, kp_range, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """One firmware-P scale per env and actuator group (legs, head), drawn from
    kp_range. mdp.randomize_delayed_actuator_gains averages per-joint draws,
    which squeezes (0.75, 1.3) down to about +-5%."""
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    for actuator in env.scene[asset_cfg.name].actuators:
        if not isinstance(actuator, BamActuator):
            continue
        kp = torch.empty(len(env_ids), 1, device=env.device).uniform_(*kp_range)
        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids, kp_scale=kp, kd_scale=torch.ones_like(kp))


def make_openduck_velocity_env_cfg(play: bool = False, rough: bool = False):
    base.USE_PROJECTED_GRAVITY = False
    try:
        cfg = base.make_microduck_velocity_env_cfg(play=play, rough=rough)
    finally:
        base.USE_PROJECTED_GRAVITY = True

    cfg.scene.entities = {"robot": OPENDUCK_WALK_ROBOT_CFG}
    cfg.actions["joint_pos"].clip = dict(JOINT_TARGET_LIMITS)
    cfg.rewards["target_clip_excess"] = RewardTermCfg(func=target_clip_excess, weight=-1.0)

    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=HEAD_BODY_NAMES
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.15, 0.16)

    for name in ("base_ang_vel", "raw_accelerometer"):
        term = cfg.observations["actor"].terms[name]
        term.delay_min_lag, term.delay_max_lag = IMU_LAG
    for name, n in OBS_NOISE.items():
        cfg.observations["actor"].terms[name].noise = Unoise(n_min=-n, n_max=n)
    cfg.events["randomize_motor_gains"] = EventTermCfg(
        func=randomize_servo_kp, mode="reset", params={"kp_range": KP_RANGE}
    )
    cfg.rewards["air_time"].params["threshold_min"], cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW
    cfg.rewards["action_rate_l2"].weight *= ACTION_RATE_SCALE
    for stage in cfg.curriculum["action_rate_weight"].params["weight_stages"]:
        stage["weight"] *= ACTION_RATE_SCALE

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = LIN_VEL_X
    twist.ranges.lin_vel_y = LIN_VEL_Y
    twist.ranges.ang_vel_z = ANG_VEL_Z

    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] = FOOT_TARGET_HEIGHT
    cfg.rewards["body_pose_tracking"].params["nominal_height"] = NOMINAL_HEIGHT

    # Same 5-stage widening as Microduck (5% -> 100% of each cap).
    stages = []
    for step, frac in ((0, 0.05), (500, 0.15), (1000, 0.35), (1500, 0.65), (2000, 1.0)):
        stages.append({"step": step * base.NUM_STEPS_PER_ENV,
                       "ranges": tuple((-c * frac, c * frac) for c in HEAD_CAPS)})
    cfg.commands["head_pose"].ranges = stages[0]["ranges"]
    cfg.curriculum["head_pose_range"].params["range_stages"] = stages

    return cfg


OpenDuckRlCfg = replace(
    base.MicroduckRlCfg,
    experiment_name="openduck_velocity",
    run_name="openduck_velocity",
    wandb_project="mjlab_openduck",
    algorithm=replace(base.MicroduckRlCfg.algorithm, symmetry_cfg=SYMMETRY_CFG),
)
