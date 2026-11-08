# Switching Open Duck Mini v2 to the Microduck training method

[中文版](改造为MicroDuck模型记录.md)

This record covers moving the 2.1 kg Open Duck Mini v2 from the official Playground model to the [microduck_rl](https://github.com/pollen-robotics/microduck_rl) training method (mjlab + MuJoCo Warp + BAM servo model), and every problem met on the way and how it was solved.

Hardware: Raspberry Pi Zero 2W, 14 STS3215, a separate BNO055, a US-100 ultrasonic sensor, 2S 4000 mAh battery.

Results first:

| Item | Status |
| --- | --- |
| Standing | Done, od11 on the real duck: standing gyro jitter about 0.001 rad/s |
| Forward | Done, about 3 steps/s, about 75 s of continuous walking in one go |
| Turning | Mostly usable, closed loop; **left turns still clearly weak, unsolved** |
| Ultrasonic obstacle avoidance | **Works poorly, unsolved** |
| Squat / stand up | 6/6 unassisted on a wooden floor (sit6, sit8), not on a rubber mat, see [squat / stand notes](squat_stand_notes.md) |
| Voice control | Done (XiaoZhi + MCP); the voice "stop" is too slow to be an emergency stop |

## 1. How it was ported

- New task `Mjlab-Velocity-Flat-OpenDuck`: Microduck's reward recipe with Open Duck's model, initial pose and servos. The 14 joint names are the same, so the task code is shared.
- Servos use the BAM STS3215 7.4 V parameters (firmware kp: legs 20, head 8).
- 61-D observation: gyro 3, joint position minus HOME 14, joint velocity 14, last action 14, command 3, gravity direction 3, head command 4, body command 6.
- Action: target position = HOME + action, normalisation baked into the ONNX.
- On the Pi, `mjlab_walk.py` picks the model type from the ONNX input size and supports both the old 101-D and the new 61-D models.
- About 6 s per iteration, 1000 iterations take about 1 h 50 min. One change per run, check every 1000 iterations.

## 2. Training iterations (one change each)

| Version | Change | Result |
| --- | --- | --- |
| od1 | First version | Walks in simulation; on the real stand the knees folded: the policy pressed the knees onto the XML hard limits |
| od2 | Targets clipped to the limits | No steps, legs propped on the limits |
| od3 | Clipped to a safe box around HOME, penalty for going beyond | Fine in simulation; on the real floor it diverged at the start and fell |
| od4 | Four changes at once (counter-example) | No steps: never change several things at once |
| od5 | Slower step rate | Walks, but the floor falls remain |
| od6 | Gyro and accelerometer delay 2–5 steps | Found the main cause of the real falls |
| od7 | Delay 0–5 steps | Also handles low-delay standing |
| od8 | Servo kp randomised 0.75–1.3 | First real stand and walk, but constant jitter |
| od9 | Gyro noise ±0.1 | Jitter unchanged, a dead end |
| od10 | Model CoM 12 mm further back | Walking clearly better (forward 0.144), standing jitter larger |
| od11 | Action-rate penalty ×2 | **Final model at 13000**: standing jitter gone, knees beyond the box 6–9 % |

Note: within one run, quality does not improve monotonically. After od11 13000 the jitter came back, so keep and compare every checkpoint; the last one is not always the best.

## 3. Problems and fixes

### 3.1 The root cause of the real falls: IMU delay

Symptom: the simulation never falls, the real duck diverges on the floor and falls within 1–4 s.

Investigation: the real duck's targets were replayed open loop in the simulation. The joints matched the real ones exactly, but the gyro only matched after shifting it by 3–4 steps. That proved the problem was IMU delay, not the servos.

Fix: random 0–5 step delay on the IMU observations in training. A later single ankle-step test measured about 2 steps (40 ms) on clean data; a better reading thread gave no clear benefit, so the original reader stayed.

### 3.2 Folded knees

Symptom: od1 on the stand folded its knees to about 1.94 rad.

Cause: the ±1.57 hard limit in the simulation carried the weight for it, and the policy learned to press on the limit.

Fix: the simulation limit moved out to the real mechanism (2.4), replaced by a target clipping box plus a penalty beyond it. The Pi uses the same box.

### 3.3 Standing jitter

Symptom: od8 kept shaking while standing at 3.7 Hz and 19 Hz; a touch stopped it, sometimes it stopped by itself. A self-excited loop of policy, servos, body and gyro.

What did not help: more gyro noise (od9).

What helped: action-rate penalty ×2 (od11).

Measure: compare the per-step target change in simulation. The simulation only reproduces it with ±0.1 gyro noise, and the numbers can be compared but do not predict the real robot. Squat model: 0.6° in simulation, 1.2° on the real duck.

### 3.4 Centre of mass

Replaying an "ankle step while standing on the floor" log, the simulation leaned back twice as much as the real duck. Foot stiffness, kp and structural flex did not explain it. Moving the model CoM back about 12 mm plus 1° of gear backlash finally matched.

Pitfall: changing `head_assembly`'s ipos has no effect (its inertia comes from its geoms), so the first try moved the CoM only 8.2 mm. Print the whole-robot CoM after a change to verify it.

### 3.5 A servo speed limit left in RAM (a big pitfall)

Symptom: suddenly no model could walk; small steps in place, very stable body.

Cause: a test script had written goal speed = 300 (about 26°/s). The value stays in the servo's RAM, survives a service restart, and the walking program never rewrites it. The real knee speed sat exactly at 28°/s.

Fix: tools clear it after use; when walking looks odd, read registers 44 and 46 first. The limit also removed the jitter, which nearly made a model look better than it was.

### 3.6 Frozen IMU

Symptom: after a hard landing, about 1000 I2C errors, the reading thread hung, the data froze, and the fall guard released a duck that was standing.

Fix:
- Rerouted the wire out of the right hip's range of motion (one real cause).
- Watchdog: data unchanged for 0.2 s → pause and hold torque.
- The fall check uses only fresh data.

### 3.7 Pausing while walking falls

Cause: pause stopped the policy at once and locked the legs mid-step.

Fix: soft pause. While moving, pause lets the model keep balancing for 2.5 s before locking.

### 3.8 Left/right turn asymmetry (not fully solved)

Symptom: large left/right bias, and on the same floor it drifted left in the afternoon and right in the evening.

Tried:
- Fixed trims −0.2, −0.1, −0.05: none right, turn rate is not proportional to the command.
- Closed-loop turning: the turn command is corrected by the integral of the low-passed gyro. Both directions reach about 13°/s.

Status: the left-turn command is near its limit (+0.9) and still weak; the real afternoon test also showed clearly weaker left turns.

New lead: in simulation, a 3° zero error on a single joint changes the straight-walk yaw like this (baseline −1°/s):

| Joint | +3° | −3° |
| --- | --- | --- |
| Left hip pitch | +4.4 | −2.6 |
| Left knee | +3.3 | −1.6 |
| Left ankle | +3.9 | −1.5 |
| Right hip pitch | +3.2 | −4.4 |
| Right knee | −3.1 | +0.4 |
| Right ankle | −5.0 | +2.9 |
| Hip yaw, hip roll | about unchanged | about unchanged |

The real duck drifts right about −5 to −7°/s, the same order as a 3° zero error in the pitch chain, so the next step is to recalibrate the leg pitch, knee and ankle zeros, especially the right ankle. This is only a simulation inference, not yet verified on the real duck.

### 3.9 Ultrasonic sensor (unsolved)

- Close up there are false 20–26 cm readings that pull the median; the stop decision now uses the second-smallest reading in a window.
- Real obstacle avoidance is still poor and needs more work.

### 3.10 Phone page and tests

- While walking press "stop", not "pause".
- The full route test script `tools/duck_test_suite.py` uses HTTP and refreshes the command every 0.2 s; if the PC or Wi-Fi drops, the service stops the servos within 0.5 s.
- A short voltage dip triggers the low-battery alarm; it must last 3 s to count.
- Logs where a hand touched the duck are not clean data; always note it.

## 4. Squat / stand up

> Later progress (sit4–sit9, real-robot results) is in [squat_stand_notes.md](squat_stand_notes.md); below are the early notes.

- Microduck's SitStand task ported to `Mjlab-SitStand-Flat-OpenDuck`.
- The observation order differs from walking: gravity comes right after the gyro. We got it wrong once and the model flipped within 1 s.
- A kneeling pose (shins on the floor) could not learn to stand in 3000 iterations: a plain interpolated rise needs 1.9 N·m and tips backward.
- Switched to a flat-foot squat: squat and stand in 1000 iterations, but only about 2.8 cm lower in simulation.
- Real duck: shaking 8–10° per step when hanging (the simulation does the same when lifted, normal). Still shaking on the floor (gyro about 0.08), squat shallow.
- sit3 doubled the action-rate penalty again; real-robot result pending.

sit3 in simulation: at iteration 5000 it squats and stands at kp 20 and 25, simulated jitter about 0.5° per step, the next real-robot candidate. The squat is still only about 2.5 cm deep; that comes from the pose, and a visible squat needs a new target pose.

### Head nod

Both servos can nod, but the program sent the command only to `head_pitch` and `neck_pitch` stayed at 0, so the neck looked frozen. Changed to share it between the two. Both have the same positive direction; balance to be checked on the real duck.

## 5. Voice (XiaoZhi)

- Waveshare ESP32-S3-AUDIO-Board flashed with v2.4.0, through the XiaoZhi cloud.
- The Pi runs `xiaozhi_bridge.py`, which dials out to the XiaoZhi MCP endpoint and forwards tool calls to `/cmd`, so no port has to be opened at home. The endpoint address works like a key: keep it in a private file on the Pi, never in the repository.
- Only safe tools are exposed: stop, walk (max 5 s), turn (max 90°), status, power off. Power on, start and relax are not.
- Do not use a boolean parameter for confirmations: the language model does not fill it; split it into two tools.
- Voice-to-action delay can exceed ten seconds; the voice "stop" is not an emergency stop.
- Own firmware: the patches in `xiaozhi/`. Adds a countdown timer (the cloud says one sentence when it ends), several local wake words, and a local "stop" sent straight to the Pi. Reminder texts must be short, the cloud refuses long ones.

## 6. Advice for whoever comes next

1. Change one parameter at a time.
2. When the simulation does not reproduce a real symptom, replay open loop first to find which part differs.
3. Read the registers before acting; do not be fooled by leftover state.
4. Compare data with a good baseline, not a worse one.
5. Measure the CoM again after moving or changing the weight of printed parts.
6. Keep and compare every checkpoint; the last one is not always the best.
