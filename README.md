# Open Duck Mini v2 组装与调试笔记 / Build and debugging notes

一台 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) 从零件到稳定行走的完整记录：遇到的问题、查出的原因、解决办法，以及调试过程中写的脚本和工具。

A complete record of one [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), from parts to steady walking: the problems met, the causes found, the fixes, and the scripts and tools written along the way.

硬件 / Hardware：树莓派 Zero 2W + 官方运行时（[Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) v2）+ 黑马（icheima）打印件和外壳，Waveshare 舵机板，14 个 STS3215（7.4 V）舵机，BNO055 IMU。
Raspberry Pi Zero 2W + the official runtime + Heima (icheima) printed parts and shell, Waveshare servo board, 14 STS3215 (7.4 V) servos, BNO055 IMU.

## 记录 / Notes

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| 组装与调试问题 / Assembly and debugging | [组装与调试问题记录](docs/组装与调试问题记录.md) | [Assembly and debugging notes](docs/assembly_and_debugging_notes.md) |
| 改用 Microduck 训练方法 / Microduck training method | [改造为MicroDuck模型记录](docs/改造为MicroDuck模型记录.md) | [Microduck training port](docs/microduck_training_port.md) |
| 蹲下 / 站起 / Squat and stand | [蹲下站起记录](docs/蹲下站起记录.md) | [Squat / stand notes](docs/squat_stand_notes.md) |
| 横移 / 后退步态训练 / Side-step and backward gait | [横移后退步态训练记录](docs/横移后退步态训练记录.md) | [Side-step / backward gait training](docs/side_step_backward_gait_training.md) |
| 超声波避障 / Ultrasonic obstacle avoidance | [超声波避障记录](docs/超声波避障记录.md) | [Ultrasonic obstacle avoidance](docs/ultrasonic_obstacle_avoidance.md) |
| 语音与音响（小智）/ Voice and speaker (XiaoZhi) | [语音音响改造记录](docs/语音音响改造记录.md) | [Voice and speaker retrofit](docs/voice_speaker_retrofit.md) |
| 视频接入方案 / Camera plan | [视频接入方案](docs/视频接入方案.md) | [Camera plan](docs/video_plan.md) |

## 最重要的三个发现 / The three most important findings

1. **舵机寄存器 85（最大加速度）出厂值 50 会让舵机比仿真慢 3 倍**（跟踪滞后 200 ms，仿真 60 ms），鸭子总是向后倒。14 个舵机都改成 0 后滞后降到 80 ms。工具：`tools/servo_regs.py`（先备份再改）。
   **Servo register 85 (max acceleration) at its factory value 50 makes the servos 3 times slower than the simulation** (tracking lag 200 ms vs 60 ms) and the duck keeps falling backward. Set to 0 on all 14 servos, the lag drops to 80 ms. Tool: `tools/servo_regs.py` (back up first).
2. **改完后腿部刚度 P 要从 30 降到 20**，否则在地上会以 5 Hz 打颤。
   **After that, lower the leg stiffness P from 30 to 20**, or it shivers at 5 Hz on the floor.
3. **树莓派硬件 I2C 读 BNO055 会出毛刺**（约 25% 的读数首字节被翻转），改用软件 I2C（`dtoverlay=i2c-gpio`）后为 0。
   **The Pi's hardware I2C glitches with the BNO055** (about 25 % of reads have the first byte flipped); software I2C (`dtoverlay=i2c-gpio`) brings it to 0.

## 目录 / Folders

| 目录 / Folder | 内容 | Contents |
| --- | --- | --- |
| `docs/` | 各项记录（中英文） | The notes (Chinese and English) |
| `robot/` | 在树莓派上运行：键盘遥控行走（摔倒保护、陀螺仪自动保持方向、数据记录）、站立、电池保护服务 | Runs on the Pi: keyboard walking (fall guard, gyro heading hold, logging), standing, battery guard service |
| `tools/` | 舵机寄存器读写、零位校准、关节测试、IMU 检查、行走数据分析器、无界面 MuJoCo 仿真对比实验 | Servo register tools, zero calibration, joint tests, IMU check, walk log analyser, headless MuJoCo comparison |
| `mjlab_port/` | Microduck 方法移植到 Open Duck 的任务和机器人配置 | Microduck tasks and robot config ported to Open Duck |
| `robot_mjlab/` | 树莓派上的服务、手机页面、超声波、小智桥接 | Pi service, phone page, ultrasonic sensor, XiaoZhi bridge |
| `xiaozhi/` | 小智固件补丁（计时、本地指令） | XiaoZhi firmware patches (timer, local commands) |
| `training/` | 训练补丁和奖励函数（横移 / 后退要领） | Training patches and reward functions (side-step / backward technique) |
| `windows/` | Windows 一键启动行走 / 关机 | Windows one-click walk / shutdown |

## 键盘遥控行走 / Keyboard walking

把 `robot/` 里的文件复制到树莓派的 `~/Open_Duck_Mini_Runtime/scripts/`，`walk.sh` 放到家目录：
Copy the files in `robot/` to `~/Open_Duck_Mini_Runtime/scripts/` on the Pi and `walk.sh` to the home folder:

```bash
ssh -t duck@<树莓派IP / Pi IP> "~/walk.sh"
```

默认设置：腿部 P = 20、方向微调 0.1、陀螺仪保持方向、倾斜超过 40° 自动卸力、每一步记录到 `~/walk_logs/`。
Defaults: leg P = 20, heading trim 0.1, gyro heading hold, torque released beyond 40° tilt, every step logged to `~/walk_logs/`.

按键：`p` 开始/暂停 · `w`/`s` 前后（按两次 = 0.1 m/s，官方模型 0.05 不迈步）· `a`/`d` 横移 · `q`/`e` 转向 · 空格 停下但保持平衡 · `z`/`c` 方向微调 · `g` 保持方向开关 · `r` 摔倒后恢复 · `x` 退出并保存。
Keys: `p` start/pause · `w`/`s` forward/back (twice = 0.1 m/s; the official model does not step at 0.05) · `a`/`d` side step · `q`/`e` turn · space stop but keep balancing · `z`/`c` heading trim · `g` heading hold on/off · `r` recover after a fall · `x` quit and save.

分析一次记录 / Analyse a log：

```bash
python tools/analyze_walk.py walk_20261025_160109.pkl --plot walk.png
```

## 重新训练（真机版）/ Retraining (real-robot version)

在 Open_Duck_Playground 目录用它的虚拟环境依次运行：
In the Open_Duck_Playground folder, with its virtual environment:

```bash
python training/make_real_xml.py 20 8 2.0   # 舵机刚度按 P=20/头部 P=8 换算，最大力矩 2.0 N·m / stiffness from P=20, head P=8, max torque 2.0 N·m
python training/patch_real.py               # 新任务 flat_terrain_backlash_real（原任务不受影响）/ new task, the original is untouched
python training/patch_sigma.py
python playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash_real --num_timesteps 160000000 --output_dir checkpoints_real
```

改动：真实舵机刚度和力矩（并随机化）、去掉横向速度死区、更严格的速度跟踪、零速度指令 20%、"站着别动"惩罚加强、每 10 秒换一次指令。
Changes: real servo stiffness and torque (randomised), no lateral speed dead zone, stricter velocity tracking, 20 % zero-speed commands, a stronger "stand still" penalty, a new command every 10 s.

后来改用 Microduck（mjlab）方法训练，见上面的记录。
Training later moved to the Microduck (mjlab) method, see the notes above.
