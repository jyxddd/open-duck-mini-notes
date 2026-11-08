# Open Duck Mini assembly and debugging notes

[中文版](组装与调试问题记录.md)

Oct 25, 2026 · @zz

## Overview

From installing the OS on Oct 11 to Oct 25, the duck went to **walking half a minute by keyboard control without falling, holding its heading by itself**. Forward, turning and turning while walking all work. Walking backward still falls; a model with real-robot parameters is being retrained.

Hardware: Raspberry Pi Zero 2W + the official runtime (Open\_Duck\_Mini\_Runtime v2) + Heima (icheima) printed parts and shell, Waveshare servo board, 14 STS3215 (7.4 V) servos, BNO055 IMU, Heima foot switches and power board, 2S 7.4 V battery. The walking model is the official BEST\_WALK\_ONNX\_2.

| Date | Milestone |
| --- | --- |
| 10-25 afternoon | Servo acceleration register, P = 20, automatic heading hold → steady 30 s walk; retraining started |
| 10-25 morning | IMU glitches fixed; first unassisted 30 s stand; first few steps |
| 10-24 | All 14 servos calibrated and direction-checked; IMU, foot switches, electronics on the body; hanging stand test |
| 10-21 | IMU soldered and wired; battery powers the Pi; OTG cable reaches the servo board |
| 10-19 \~ 10-20 | Legs, hips, neck and head assembled; new servo board fitted |
| 10-15 \~ 10-16 | Power parts arrived and tested; old servo board's bus damaged |
| 10-12 \~ 10-13 | Servo IDs set, right lower leg assembled |
| 10-11 | Pi OS and runtime installed; hardware route decided |

## Preparation and software environment

The biggest obstacle was the network: the Pi could not reach GitHub, so all code was downloaded on the PC first and copied over.

| Problem | Cause | Fix |
| --- | --- | --- |
| The Pi cannot pull code from GitHub | Connections reset on the local network | Download on the PC and scp to the Pi; pip through the Tsinghua mirror |
| duckpi.local does not find the Pi | The PC does not support mDNS | Use the IP 192.168.50.139; if it changes, look for a MAC starting 88-A2-9E in the ARP table |
| Scripts say "no module numpy" | Virtual environment not activated | `source ~/.virtualenvs/open-duck-mini-runtime/bin/activate` first |
| cv2 import fails | Missing system library | `apt install libgl1` |
| Out of GPU memory in training | RTX 4060 laptop has only 8 GB | Force 512 parallel environments; 160 M steps take about 1 h 50 min |
| The official runtime has no "stand" script | Only walking is provided | Wrote v2\_rl\_stand.py with a self-trained standing model |
| The seller has no model for the Heima shell | The Heima course just runs the official training | Use the official model, then retrain for the real robot |
| The seller says a Pi Zero needs an adapter board | Heima uses a KickPi main board | "Route A": wire the Pi Zero 2W the official way and use only Heima's power board and foot boards |

## Servos and servo board

No servo was broken. A servo board and one servo cable failed. The key habit: cut servo power before plugging or unplugging servo cables.

| Problem | Cause | Fix |
| --- | --- | --- |
| FD software (1.9.8.2) has no "center calibration" button | Software version | Wrote set\_middle.py: writing 128 to address 40 makes the current position 2048 |
| New servos have max input voltage 80 (8.0 V) | Factory setting; at a full 8.4 V they shut off | Set to 90 while setting each ID |
| The board finds no servo and returns fixed garbage even with nothing attached | The board's bus side is damaged, probably by hot-plugging | New Waveshare board; cut servo power before plugging from now on |
| Unsure about the jumper position | USB control on this board is position B (I first said A) | Keep it on B |
| Scripts cannot connect after the board swap | Old board COM3, new board COM4 | Use COM4; on the Pi it is /dev/ttyACM0 |
| 12/13/14 suddenly missing, new 14 blinks red | A servo cable broke | Replace the cable; use 5264-3P cables, not Dupont |
| The board has only 2 ports | The official wiring has three chains: right leg, left leg, neck | Chain the neck off the end of the left leg (same as the official diagram) |
| Occasional checksum errors while walking, always ID 10 | ID 10's horn was loose | Tightened, zero unchanged (+0.9°); the official stance is that occasional checksum errors can be ignored |

## Mechanical assembly

The easiest way: fit a horn on any tooth and calibrate the zero in software afterwards. The most important thing: screws must not loosen.

| Problem | Cause | Fix |
| --- | --- | --- |
| Aligning horns to teeth is hard, many re-fits | 25 teeth, about 14° each | Stop aligning: after assembly hold the zero pose and calibrate with set\_middle.py |
| Horn orientation unclear | I once had the idler horn backwards | Both horns: boss toward the servo, flat face out |
| Most joints had only 2 horn screws | Speed during debugging | ID 10 loosened and walking got messy; fit all 4 and check regularly |
| Thread-locker or not | The community recommends blue thread-locker before walking | After final tuning, only on aluminium horn screws; keep it off printed parts (it attacks plastic) and servo shafts |
| Head roll (33) moves opposite to the simulation | The whole head plate was mounted 180° rotated | Turned the head plate round (wide end forward), removed the software reversal, recalibrated 32 and 33 |
| Head servos far off (up to 156°) | Head never calibrated | Calibrated 30–33 with "neck vertical, head plate level facing forward" as zero |
| Right hip roll (11) 55° off | Horn refitted during reinforcement | Recalibrated |
| Trunk bottom bearings missing | Step skipped | Pressed in 2 × 6804 bearings and caps during reinforcement |
| Purpose of the red stand misunderstood | I first took it for a body frame | It is Heima's display stand, used for hanging tests |
| Where the Pi and servo board go | Heima's structure differs from the official one | Servo board on the 4 posts of the trunk top plate, Pi at the bottom of the body with the SD card toward the removable cover |

Common joint directions (+0.3 rad): right hip pitch 12 → leg swings back, left hip pitch 22 → leg swings forward; knees 13/23 → shank forward; ankles 14/24 → toes up. All 14 joints match the simulation, no software reversal needed.

## Power

The only smoke came from a Dupont wire used for power. Since then all power goes through XH cables + WAGO connectors; the Pi runs from a UBEC and has never shown undervoltage.

| Problem | Cause | Fix |
| --- | --- | --- |
| Dupont wire from the power board to the UBEC smoked and melted | Dupont wire too thin, and a short | Never use Dupont for power; cut XH cables + WAGO connectors; battery and power board tested fine afterwards |
| Where the Pi's 5 V comes from | The Heima power board has no 5 V output | Hobbywing 3 A UBEC (jumper at 5 V) on Pi pins 2/4/6: red → 4, black → 6, pin 2 empty; check 5.3 V no-load before connecting |
| How to power the servo board | Did not want to buy DC plugs | The board has DC+/DC− green terminals; a cut XH cable from the power board output |
| The Pi has only one USB port | Pi Zero 2W limitation | 30 cm Type-C ↔ Micro OTG cable to the servo board; a future microphone needs I2S or a hub |
| The charger LED never turns green | The LED is unreliable | Go by voltage: 8.4 V is full, about 2 h of charging |
| No warning when the battery is nearly empty | The Pi cannot measure voltage | All 14 servos' minimum voltage set to 7.0 V (red blinking below it); background service battery-guard reads the servo voltage every 30 s: below 7.2 V the green LED blinks fast, 3 readings in a row below 6.8 V release the servos and shut down |
| Cutting power directly may damage the SD card | The system keeps writing | Double-click 关闭鸭子.bat (shut down the duck), wait until the green LED is off, then switch off |

Community advice: a 20 A BMS is better (some saw 10 A ones cut out under high current), and at least 18 AWG wire from the battery to the servo board. Ours is a 10 A BMS, no problem so far, keep an eye on it.

## IMU and foot switches

IMU I2C glitches were the first real reason the duck could not stand. With software I2C the glitches went from 25 % to 0 % and the duck stood unassisted for 30 s for the first time.

| Problem | Cause | Fix |
| --- | --- | --- |
| Occasional huge IMU jumps, as if the duck suddenly leaned back 7° | The Pi's hardware I2C does not handle the BNO055's clock stretching; in about 25 % of reads the top bit of the first byte flips | Turn off hardware I2C in /boot/firmware/config.txt and use software I2C (i2c-gpio) on the same pins; 0 % glitches, 104 Hz reads |
| Calibration was done on the glitchy bus | Calibrated before finding the glitches | Recalibrated after the fix: each of the six faces flat and still for 5 s, then figure eights |
| Is the IMU orientation right | The most common community pitfall (mounted backwards, x/y swapped) | Measured: nose down gives negative x, right side down gives positive y, matching the simulation (x forward, y left, z up), upside\_down=false |
| Foot switch wires too short, wrong connector | Heima foot boards use XH | Cut at the body end, WAGO to female Dupont; left foot 15+20, right foot 13+14; left, right and both feet tested correct |
| i2cdetect not found | Not on PATH | Use /usr/sbin/i2cdetect; IMU at 0x28 |

Pin summary: IMU VCC 1, SDA 3, SCL 5, GND 9; UBEC 4 (5 V) and 6 (GND); left foot 15+20; right foot 13+14.

## Software and runtime pitfalls

These raise no errors, they only make the behaviour strange, so they cost the most time.

| Problem | Cause | Fix |
| --- | --- | --- |
| The knee seems stuck at 52° and never reaches the init pose | My debugging tool wrote a goal speed (address 46) into the servo; it stays until power-off | Run clear\_speed.py after using tools, clearing addresses 46/44 |
| The self-trained standing model stands splay-footed on the foot edges | The model learned badly | Use the official walking model; at zero speed it stands |
| Pressing p freezes the duck mid-step | p is pause, motors lock in their last position | To stop while walking use space (speed to zero, keep balancing) |
| --log saved nothing | Bug in my script, the saving part was missing | Fixed, logging is now on by default |
| x does nothing, the script keeps running | The terminal window had no focus | Click the terminal first |
| The runtime's --pitch\_bias does nothing | Stored in the code but never used | Not used; simulation shows ±8° IMU angle offset does not affect the official model |
| Doubt about an inconsistent +1.3 accelerometer offset | That line never took effect in training; only the simulation inference script adds it | Our runtime does not add it, consistent with training, no change |
| A remote pkill stopped the commands after it | pkill -f matched and killed its own command | Write `pkill -f "[w]alk_zero"` |
| The official stance: occasional checksum errors can be ignored | The runtime drops that frame | Only check cables and loose parts if they become frequent |

## Standing and walking

The main reason walking was unstable was **slow servo response**: the factory acceleration limit made the servos 3 times slower than the simulation. With that removed and stiffness P lowered from 30 to 20, the duck walks steadily. Every step was confirmed by comparing `--log` data with the simulation, one variable at a time.

| Order | Symptom | Cause found | Fix | Effect |
| --- | --- | --- | --- | --- |
| 1 | Keeps falling backward while standing | IMU I2C glitches (not weight: robot 2085 g, simulation 2107 g) | Software I2C + recalibration | First unassisted 30 s stand |
| 2 | Gets messy after a few steps | ID 10 (right hip yaw) loose | Tightened, zero unchanged | Checksum errors gone |
| 3 | 5 falls in 200 s, almost all backward; walking pitches ±10–14° | Servo tracking lag 200 ms (simulation 60 ms); register 85 (max acceleration) factory value 50 limits acceleration | Register 85 set to 0 on all 14 servos (backed up first) | Hanging lag 160 → 80 ms, tracking error down to 1/5 |
| 4 | "Shivers" on the floor | Faster servos make the loop oscillate at 5 Hz | Leg P 30 → 20 (damping D = 8 tried, no effect) | No shivering, walking sway ±4° |
| 5 | No steps at speed 0.05 | A "dead zone" in the official training reward | Walk at 0.1 (press w twice) | 1.4 steps per foot per second (was 0.3) |
| 6 | Keeps veering right, about 12°/s | Left/right asymmetry, and the direction changes from run to run | A fixed trim of 0.1, then automatic heading hold from the gyro | Heading wanders ±10–18°, ends within ±3° |
| 7 | After a fall the servos keep pushing on the floor | No protection in the runtime | Release torque beyond 40° tilt, press r to recover | Verified on the real duck |
| 8 | Walking backward falls back after 2.6 s | The official model never learned backward (in simulation it does not move) | Do not use s for now; retrain | Not solved |
| 9 | A 0.1 side step alone does nothing | 0.1 lateral dead zone in the training reward | Combine with forward, or side-step at 0.2 | Retraining should fix it |

Before and after (floor, forward 0.1):

| Metric | Morning (original settings) | Afternoon (now) | Simulation |
| --- | --- | --- | --- |
| Servo tracking lag | 200 ms | 80 ms | 60 ms |
| Walking pitch sway (std) | 10–14° | 4–5° | 2° |
| Touchdowns per foot per second | 0.3 | 1.4 | 0.6–2 |
| Falls | 5 in 200 s | 0 in runs of 26 s and 50 s | 0 |

Suspicions ruled out on the way: IMU mounted crooked (±8° in simulation does not fall), foot switch signals (double support 80 %, simulation 81 %), accelerometer offset, dropped frames (always 0).

## Best settings and common commands

Double-click `D:\Claude\duck\鸭子走路.bat` to start with the best settings; double-click `关闭鸭子.bat` to shut down.

| Setting | Value | Where |
| --- | --- | --- |
| Servo register 85 (max acceleration) | 0 (factory 50) | Written into all 14 servos, kept over power-off; backup in pi/backup\_2026-10-25/servo\_regs\_factory.json |
| Leg stiffness P | 20 (head 8) | Keyboard script default |
| Heading trim | 0.1 | Keyboard script default, z/c adjust |
| Automatic heading hold | On | g toggles |
| Fall protection | Release beyond 40° tilt | r recovers |
| Data logging | On | Pi \~/walk\_logs/ |

Keys: p start/pause · w/s forward/back (twice = 0.1, do not use s for now) · a/d side step · q/e turn · space stop but keep balancing · r recover after a fall · x quit and save. Do not press the head keys i/j/k/l while walking.

Common commands (in a PC terminal):

```
ssh -t duck@192.168.50.139 "~/walk.sh"                    # keyboard walking
ssh duck@192.168.50.139 "cat ~/battery_status.txt"       # battery level
ssh duck@192.168.50.139 "sudo poweroff"                  # shut down
ssh duck@192.168.50.139 "python ~/read_all.py"          # read the 14 servo positions (activate the venv first)
```

Battery: 8.4 V full; stop walking tests and charge at about 7.3 V.

## Open problems and next steps

Most remaining problems are limits of the official model; a "real robot" version is being retrained (160 M steps, about 2 h).

- Walking backward falls; a 0.1 side step alone does nothing; at zero speed it steps in place.
- Large side-to-side sway while walking, roll up to about 25–30°; no fall from it yet, but watch it.
- The root cause of the drift is not found (the gyro corrects it for now); standing, right hip yaw ID 10 often holds +5.5°, the left only −2.3°.
- Most joints still have only 2 horn screws.

What the retraining changes (the simulation was calibrated first: real-robot simulation servo lag 90 ms, real 80 ms):

1. Servo stiffness converted from the real P = 20, head P = 8; max torque 2.0 N·m (was 3.23), randomised between 1.6 and 2.2.
2. The velocity reward loses its 0.1 lateral dead zone and gets stricter, so small speeds are worth stepping for.
3. Zero-speed commands 10 % → 20 %, a stronger "stand still" penalty, so it learns to really stand.
4. A new command every 10 s, to learn starting, stopping and changing direction.

- [ ] After training, check it in simulation with the same analyser (forward, side step, backward, standing)
- [ ] If it passes, the same full test on the real robot as the official model
- [ ] Once tuned: all 4 horn screws and thread-locker
- [ ] Longer term: voice control (in China, XiaoZhi AI + ESP32-S3 first); sit/stand following the Microduck open-source training code

More community material is on the PC in D:\\Claude\\duck\\社区资料\_Discord.md.
