# Squat / stand up

[中文版](蹲下站起记录.md)

Training Open Duck Mini v2 to squat and stand up with mjlab (the Microduck method), the whole process from 2026-10-31 to 11-03.

## Status (2026-11-03)

| Item | Result |
| --- | --- |
| Current model | `mj_sit8_7000` (leg kp 20), spare `mj_sit6_6750` |
| Wooden floor, no hand support | sit6 6/6, sit8 6/6 |
| Squat depth | hip 41°, knee 21°, ankle 16° (from the standing pose), body about 3.4 cm lower |
| Squat speed | 80 % in 1.4 s |
| Stand-up speed | 80 % in 0.3 s, **too fast**; sit9 is slowing it down |
| Rubber anti-slip mat | **Fails**: sit6 squats only once, sit8 falls sideways on the first try |
| Hip wiggle | Both models: 6–8°/s side and twist sway while squatting |

## How the task is built

- Microduck's SitStand task ported as `Mjlab-SitStand-Flat-OpenDuck` (`mjlab_port/tasks/openduck_sitstand_env_cfg.py`). It swaps constants in the Microduck module to use Open Duck's robot and poses.
- Squat / stand is the first command value (1 = squat, 0 = stand); the 3 walking-speed values are unused in the squat model.
- **The observation order differs from walking**: gravity comes right after the gyro (in walking models gravity comes after the command). We got it wrong once and the model flipped within 1 s. On the Pi, `robot_mjlab/mjlab_walk.py` picks the order by whether the file name contains `sit`.
- The squat model's joint target limits are wider than walking: hip pitch up to ±1.35, knee up to 1.7 rad (see the knee limit below).

## Versions

One parameter per run, continuing from the last good version.

| Version | Change | Simulation | Real robot |
| --- | --- | --- | --- |
| sit1 | Kneeling (shins on the floor) | Could not learn to stand in 3000 iterations; a forced rise needs 1.9 N·m and tips backward | Not tested |
| sit2 | Flat-foot squat instead | Squats and stands after 1000 iterations, but only 2.8 cm lower | Shakes 8–10° per step on the stand (normal when hanging); on the floor a very shallow, shaky squat |
| sit3 | Action-rate penalty ×4 | Works at kp 20 and 25 at iteration 5000 | First squat succeeded (knee 21/20°), then stuck standing |
| sit4 | Knees down to 2.1 rad | — | **Stopped**: the real knee limit measured by hand is only +1.75 rad |
| sit5 | Knee 1.75 | — | Stopped, no margin to the limit |
| sit6 | Knee 1.65, hip 1.3, ankle 0.4, simulated knee limit 1.76 | 0 falls in 11 conditions at 6750 | **Wooden floor 6/6**; stands up in 0.5–1 s, too fast |
| sit7 | Penalty when the body moves up/down faster than 5 cm/s | Slower rise, but at kp 25 falls in the first 0.5 s 8/10 | Not used |
| sit8 | Foot friction randomisation 0.7–1.3 → 0.3–2.0 | At friction 2.0 squats 4/4 (sit6 0/4), 0 falls on the way (sit6 3) | **Wooden floor 6/6**; on the anti-slip mat falls sideways on the first squat |
| sit9 | Rise ramp 2 s → 4 s | Training | — |

## Key findings

1. **The real knee limit is +1.75 rad.** Measured by bending the leg to the end by hand and reading the encoder. The 2.4 in the model was wrong; designing poses with it would jam the real knee. The simulated knee limit is now 1.76, targets clipped to 1.7.
2. **Floor friction can break the squat.** PLA soles on a varnished wooden floor have about 0.3, a rubber mat far above 1.3, both outside the original training range (0.7–1.3). In simulation, friction 2.5 reproduces "cannot squat, body leans back 7–10°, falls at kp 25". sit8 widened the range and fixed it in simulation.
3. **But the mat problem is not just friction.** sit8 fell sideways on its first squat on the mat: both hips got symmetric commands (20-odd degrees), the left hip followed, the right hip under load followed only half, and the body tilted to 41°. The mat is soft and the feet sink in, which the simulation does not have.
4. **The right leg is not broken.** A no-load test on the stand (`tools/stand_sine_test.py`): the 10 leg servos are the same left and right: swing tracking ratio 0.95–1.01, lag 60–80 ms, step response 0.10–0.12 s. The knees following only half while walking comes from load and speed, the same on both legs.
5. **A too-fast rise is hard to fix.** A ramp on the command did nothing; the model switches at about 0.35. sit7's speed penalty made it fall at the start. sit9 changes the posture ramp time in training.
6. **Squat models need kp 20.** In simulation sit6 cannot squat at kp 30, while walking models lift their feet higher at kp 30. The server sets kp per model (`MODEL_KP`).
7. **A jolt at the start.** sit8 sways 11–15° in about the first 0.4 s. In simulation it occasionally "falls in the first second"; the real duck shows a mild version of it.
8. **The hip wiggle did not come with sit8.** Compared with sit6 logs: gyro side about 3°/s, twist 6–8°/s, hip roll 11–15°, almost the same for both models.

## How to use it

- On the phone page choose "squat / stand sit8" and press "squat / stand". With the squat model selected the duck cannot walk; switch back to a walking model first.
- Use a hard floor only; it falls on the mat.
- Next: a combined "walk + squat" model: on squat the server stops, stands still for 2 s, then switches to the squat model by itself. Verified in simulation: switching after 2 s of standing 4/4 succeeded, switching right after stopping fell 1/4.

## Simulation check

```bash
python tools/mjlab_sim_check.py --onnx mj_sit8_7000.onnx \
  --schedule "3:0,6:1,5:0,8:1,5:0,8:1,3:0" --ground --kp_legs 20 --friction 2.0 \
  --vin 7.8 --knee_range 1.76 --clip --imu_delay 2 --backlash 1 --com_x -12 --gyro_noise 0.1 --seed 1
```

Each schedule segment is "seconds:squat flag"; the one above stands 3 s, squats 6 s, stands 5 s… three squats in all. Each segment prints the body height and tilt: about 0.16 m standing, about 0.128 m squatting. `--friction` sets the floor friction, `--seed` changes the random seed; run at least two seeds per candidate. For walk ↔ squat switching use `--sit_onnx`; segments whose 9th value is 1 use the squat model.

> 2026-11-07 note: the exam started the duck at 0.155 m, about 5 mm into the floor, which threw it in the first second. Training starts standing at 0.162–0.172 m; use `--start_z 0.167` for all squat/stand exams.
