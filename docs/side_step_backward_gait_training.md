# Side-step and backward gait training (Open Duck Mini v2, mjlab)

[中文版](横移后退步态训练记录.md)

2026-11-02 to 11-08. Setup: mjlab + rsl_rl (Microduck port), RTX 4060 laptop, 4096 parallel environments per run.
Rules: **one change per run**, **exam every 1000 iterations**. The exam is `tools/mjlab_sim_check.py`: BAM servo model, IMU delay, backlash, CoM 12 mm further back (measured on the real duck), friction 0.35 / 1.0 / 2.0, kp 20 / 25.

## Summary

Tuning reward weights, command ranges or pose limits never taught the duck to side-step. **Writing the move's technique into a reward** (pay the swing foot for moving the commanded way) worked on the first try, and backward walking was learned the same way.

## Starting point

od17, the model running on the real duck at the time, walked forward and turned well, but:
- side steps: for a ±0.15 m/s command the real lateral speed was ≤ 0.04 m/s. It stepped in place and crept forward;
- backward: a -0.15 command gave only -0.03 to -0.06 m/s.

## Attempts that failed (one change each)

| Run | Change | Result |
|---|---|---|
| od14 | walking hip-roll pose std 0.05 → 0.12 | no change in side steps |
| od15 | side command range ±0.3 (from od13) | lost backward and turns, knees on their stops 38–98 % of the time |
| od16 | same, from scratch | no steps at all, crouched shuffle |
| od20 | knee torque ×2.05 (STS3046 servo) | lost backward during training, still no side steps |
| od21 | velocity tracking weight ×2 | stood still for 0.15 commands |
| od22 | od20 + hip-roll std 0.12 | walks, side vy still ≤ 0.05 |
| od23 | side command range ±0.25 | side vy ±0.01, no better |

## What worked: the technique as a reward

### od24: side-step technique (the user's idea: "tell it how to side-step")

`side_step_technique`: while there is a side command (|vy| > 0.05), **every foot in the air** is paid for how far it has moved in the commanded direction since it lifted off (in the body's heading frame). Moving the wrong way costs the same. The payment is clipped at ±4 cm and the weight is 1.0. The existing foot-clearance reward keeps the foot lifted.

Result after 1000 iterations (friction 1.0): vy **+0.115 / -0.088 m/s** for ±0.15. Forward walking and turns were unchanged, with no falls. The swing-foot side displacement metric rose from 0.02 to 0.47 within 700 iterations.

### od25: backward technique (the user's description + human backward-gait studies: short steps)

`back_step_technique`: while there is a backward command (vx < -0.05), each swing foot is paid for moving **backward** since lift-off. The full payment is reached at 2.5 cm, and steps longer than 4 cm are penalised so the duck does not take long, unstable steps. Weight 1.0, resumed from od24 iteration 19250.

Results (od25 iteration 20500, simulation, knees ×2.05):
- backward: -0.09 to -0.11 m/s at friction 1.0, -0.04 to -0.05 at 0.35, -0.19 to -0.21 at 2.0;
- side steps worked in all 6 exams, with no falls;
- random-condition Monte-Carlo (CoM, friction, servo stiffness, a push): 0 falls in 338 runs as long as the CoM is not further back than the measured one.

### Tried afterwards, not kept

| Run | Change | Result |
|---|---|---|
| od26 | "pause with both feet down" after every step (dense reward) | stood still whenever told to go backward |
| od26b | same idea, paid only at lift-off | pause unchanged, a new fall at friction 2.0 |

## Lessons from the real robot

- **The model must match the hardware.** od20–od25 were trained with STS3046 knees (torque ×2.05). On the real duck, which still had the stock STS3215 knees, od25 rolled ±20–50° on its first steps and fell. The simulation test of od25 with stock knees showed no falls, so **that test was not reliable**.
- The other way round, od17 (trained with stock knees) is fine in simulation with stronger knees. But with only the left knee swapped on the real duck, the new servo is about twice as stiff at the same kp, and the duck **kept veering left**.
- So side steps and walking backward wait until both knees are STS3046, then the od25 line goes onto the real robot.

## Code

Reward functions: [`training/openduck_side_reward.py`](../training/openduck_side_reward.py)

Hooked into the mjlab velocity environment config:

```python
from mjlab_microduck.tasks.openduck_side_reward import FEET as SIDE_FEET, side_step_technique, back_step_technique

cfg.rewards["side_step_technique"] = RewardTermCfg(
    func=side_step_technique, weight=1.0,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist", "asset_cfg": SIDE_FEET})
cfg.rewards["back_step_technique"] = RewardTermCfg(
    func=back_step_technique, weight=1.0,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist", "asset_cfg": SIDE_FEET})
```

Note: `asset_cfg` must be passed in `params`, or the foot site_ids are never resolved.

## Takeaways

1. When the reward only states the goal (track the speed), the policy finds shortcuts such as standing still at little cost. **Rewarding how to do it** (where the swing foot goes, and how far) works better.
2. Give the technique as a direction and a rough size, not exact joint angles, so the policy can still adapt to the real robot.
3. Add one technique per run and check every 1000 iterations, so it is clear what helped.
