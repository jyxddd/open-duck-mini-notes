"""Open Duck Mini v2 (STS3215 servos) robot cfg for the Microduck recipes.

Same 14-joint layout and names as Microduck, so the task code is shared; only
the model, HOME pose and actuators differ.
"""
import os
from pathlib import Path

import bam
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator.sts3215_bam import STS3215BamActuatorCfg
from mjlab_microduck.robot.microduck_constants import FULL_COLLISION

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "openduck"
OPENDUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
assert OPENDUCK_WALK_XML.exists(), f"XML not found: {OPENDUCK_WALK_XML} (run openduck/make_openduck_xml.py)"

# BAM identification of the Feetech STS3215 at 7.4 V (model m1: kt, R,
# armature, Coulomb + viscous friction). The only STS3215 fit BAM ships.
STS3215_BAM_JSON = Path(bam.__file__).parent / "params" / "feetech_sts3215_7_4V" / "m1.json"
assert STS3215_BAM_JSON.exists(), STS3215_BAM_JSON


# The XML gives the knees a +-1.57 rad hard stop, but HOME already bends them
# 1.37 rad and the real knee bends on past 1.9 rad. od1 learned to press the
# knees into that stop to carry its weight (actions of +2..+10 rad) and on the
# real duck the knees folded. So the sim knee stop goes out to the real
# mechanism and the commanded targets are clipped instead (JOINT_TARGET_LIMITS,
# same numbers in pi/mjlab_walk.py): the servos carry the weight, as on the duck.
KNEE_RANGE = 2.4


# The real duck's CoM sits ~12 mm further back than this model: ankle-step tap
# tests on the floor (10-30) tip it backward at ~6 deg (model: ~12 deg) and the
# sim only reproduces them with the CoM moved back plus ~1 deg of backlash.
# Whole-robot -12 mm = this shift on the bodies above the hips. head_assembly is
# left out: its inertial comes from its geoms, so ipos has no effect on it.
UPPER_BODY_COM_SHIFT_X = -0.029
UPPER_BODIES = ("trunk_base", "neck_pitch_assembly", "head_pitch_to_yaw", "neck_yaw_assembly")


def get_walk_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(OPENDUCK_WALK_XML))
    for b in spec.bodies:
        if b.name in UPPER_BODIES:
            b.ipos[0] += UPPER_BODY_COM_SHIFT_X
    for j in spec.joints:
        if j.name.endswith("_knee"):
            j.range = [-KNEE_RANGE, KNEE_RANGE]
    return spec


# Clip for the position targets (rad): a box around HOME that keeps every
# joint >= ~0.1 rad clear of its XML stop. od2 clipped at the stops themselves
# and learned to splay its legs against them (hip yaw/roll/pitch and knee all
# at the limit, no steps). Same numbers in pi/mjlab_walk.py and
# tools/mjlab_sim_check.py. HOME: hip_pitch -0.63/+0.635, knee 1.37, ankle -0.78.
JOINT_TARGET_LIMITS = {
    r".*_hip_yaw": (-0.35, 0.35),
    r".*_hip_roll": (-0.3, 0.3),
    "left_hip_pitch": (-1.1, 0.3),
    "right_hip_pitch": (-0.3, 1.1),
    r".*_knee": (0.6, 1.75),
    r".*_ankle": (-1.45, 0.3),
    "neck_pitch": (-0.3, 1.0),
    "head_pitch": (-0.7, 0.7),
    "head_yaw": (-1.2, 1.2),
    "head_roll": (-0.4, 0.4),
}


# Playground "home" keyframe (= runtime init_pos). Head level, legs bent.
HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.15),
    joint_pos={
        "left_hip_yaw": 0.002,
        "left_hip_roll": 0.053,
        "left_hip_pitch": -0.63,
        "left_knee": 1.368,
        "left_ankle": -0.784,
        "neck_pitch": 0.0,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "right_hip_yaw": -0.003,
        "right_hip_roll": -0.065,
        "right_hip_pitch": 0.635,
        "right_knee": 1.379,
        "right_ankle": -0.796,
    },
    joint_vel={".*": 0.0},
)

# Firmware P gains as set on the real duck by the runtime (register 21):
# legs 20 (the shiver fix), neck/head 8. Battery: 2S Li-ion, 6.9 V cut-off,
# 8.4 V full; sag under load is randomized.
_BAM_COMMON = dict(
    json_path=str(STS3215_BAM_JSON),
    vin_range=(7.0, 8.3),
    vin_drop_gain_range=(0.0, 0.1),
    vin_min=6.5,
    delay_min_lag=3,
    delay_max_lag=6,
)
leg_actuators = STS3215BamActuatorCfg(
    target_names_expr=(r".*(hip|knee|ankle).*",), kp_fw=20.0, **_BAM_COMMON
)
head_actuators = STS3215BamActuatorCfg(
    target_names_expr=(r".*(neck|head).*",), kp_fw=8.0, **_BAM_COMMON
)

OPENDUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(leg_actuators, head_actuators),
        soft_joint_pos_limit_factor=0.9,
    ),
)
