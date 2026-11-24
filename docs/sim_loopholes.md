# Simulation loopholes (home-made v2 duck, T line, 2026-11-21 to 11-24)

The T line is a free experiment: as few rewards as possible, and the duck finds motions by itself (see the [v2 simulation training summary](v2_sim_training_summary.md)). Motions learned this way often look effective in simulation while actually relying on gaps in the simulator. Below are the ones confirmed so far, how each was found and how it was closed. Simulation only, nothing here involves the real robot. 中文：[仿真漏洞记录](仿真漏洞记录.md)

## 1. Only the soles collide with the ground, so the trunk can sink into it (found 11-22 07:40)
- **What happened**:
  - After falling over, the `T_auto_5000` duck had its trunk 0.13 m under the ground (feet 0.02 m above it). It rolled 178° underground and back, and "got up" in 1 second.
  - `T_roll_5000` relied on the same gap. It rolled sideways about 355° every 0.43 s, and 58 of 64 ducks reached the 5 m circle.
- **Cause**: at the time only `*_foot_collision` collided with the ground (`FULL_COLLISION` selected only `.*_collision`), so the trunk, head and legs could pass through the floor.
- **How it was found**: replaying the recording with a half-transparent floor (`render_trip_npz.py`, `FLOOR_ALPHA=0.35 CAM_ELEV=0`) and checking whether the trunk height was below the ground.
- **Fix**: whole-body ground collision from 11-22 (`T_COLLISION`); `penetration_npz.py` reports the deepest penetration.
- **Outcome**: neither checkpoint counts as a kept motion; both were deleted on 11-24.

## 2. Progress was paid while lying down, so shuffling on the ground counted (found 11-22)
- **What happened**:
  - With whole-body collision, `T_sidecrawl_1000` fell onto its side in 0.7 s, then pushed and shuffled while lying at a 30–60° tilt.
  - It covered about 2.7 m in 10 s, all 64 ducks the same.
  - Forward progress was paid up to 60° of tilt, so lying down still earned reward.
- **Fix**: pay progress only below 30° of tilt (`PAY_GZ`); end the episode after more than 0.2 s down (`DOWN_MAX_S`).
- **Outcome**: impossible on the real robot; deleted on 11-24.

## 3. Sitting earned the standing reward (found 11-23)
- **What happened**: after reaching 5 m, `T_runsit_6000` "stood" with the trunk at 0.089 m, shins on the ground and a 24° tilt (< 30°). It still earned the standing reward.
- **Fix**: the standing reward is paid only when the trunk is at least 0.11 m high (`STAND_Z`).

## 4. The simulator ignores thigh–trunk collisions (found 11-24)
- **What happened**: the simulator excludes collisions between the thigh and the trunk, so motions used poses the real robot cannot reach:
  - While turning over, the hip went back ≥ 1.4 and out ≥ 0.2, so the thigh would hit the side of the trunk.
  - The hip went forward ≤ −1.0 with the knee folded, so the foot would hit the trunk.
  - When the knee angle went beyond the CAD knee / ankle fit table, the shin would hit the thigh.
- **How it was found**: a real-robot self-collision table built from the CAD, checked for every step of all 64 ducks (`record_getup_T2.py`, `record_sitstand_T.py`).
- **Fix**: a real self-collision penalty in the reward (`gu_selfcol_pen`, weight 5.0). After 2000 more iterations of the get-up task, hip / trunk violations fell from 17 ducks to 3, and knee violations to 0.

## 5. The sitting "trunk height" was met with the trunk bottom in the air (found 11-24)
- **What happened**:
  - The sit in `T_sitstand_3000 / 4000` is a squat with the feet on the ground, trunk centre at 0.066 m.
  - But the trunk bottom was 58 mm above the ground, the thighs only 1–2 mm from it, and the shins about 27 mm in the air.
  - A check of "trunk height ≤ 0.097 m" alone counts this as sitting.
- **How it was found**: measuring the ground clearance of each part. `mj_geomDistance` returning 0 was not reliable, so the contact distance was read with a larger contact margin instead.
- **Status**: not closed. The wanted pose is a long sit (legs straight and flat on the ground), written as `SS_LONGSIT`, to be trained.

## General lessons
- Replay and check contacts and floor penetration first, then describe the motion.
- Judge only by deterministic exams (no noise, no pushes), not by statistics from the training window.
- "It moves" is not the same as "it works": when training numbers look good, first suspect a gap in the reward or the collisions.
