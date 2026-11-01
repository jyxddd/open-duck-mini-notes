# Open Duck Mini v2 组装与调试笔记

一台 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) 从零件到稳定行走的完整记录：遇到的问题、查出的原因、解决办法，以及调试过程中写的脚本和工具。

硬件：树莓派 Zero 2W + 官方运行时（[Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) v2）+ 黑马（icheima）打印件和外壳，Waveshare 舵机板，14 个 STS3215（7.4 V）舵机，BNO055 IMU。

**完整问题记录：[docs/组装与调试问题记录.md](docs/组装与调试问题记录.md)**

**改用 Microduck 训练方法的记录：[docs/改造为MicroDuck模型记录.md](docs/改造为MicroDuck模型记录.md)**

**视频接入方案：[docs/视频接入方案.md](docs/视频接入方案.md)**

## 最重要的三个发现

1. **舵机寄存器 85（最大加速度）出厂值 50 会让舵机比仿真慢 3 倍**（跟踪滞后 200 ms，仿真 60 ms），鸭子总是向后倒。14 个舵机都改成 0 后滞后降到 80 ms。工具：`tools/servo_regs.py`（先备份再改）。
2. **改完后腿部刚度 P 要从 30 降到 20**，否则在地上会以 5 Hz 打颤。
3. **树莓派硬件 I2C 读 BNO055 会出毛刺**（约 25% 的读数首字节被翻转），改用软件 I2C（`dtoverlay=i2c-gpio`）后为 0。

## 目录

| 目录 | 内容 |
| --- | --- |
| `docs/` | 组装与调试问题记录 |
| `robot/` | 在树莓派上运行：键盘遥控行走（摔倒保护、陀螺仪自动保持方向、数据记录）、站立、电池保护服务 |
| `tools/` | 舵机寄存器读写、零位校准、关节测试、IMU 检查、行走数据分析器、无界面 MuJoCo 仿真对比实验 |
| `mjlab_port/` | Microduck 方法移植到 Open Duck 的任务和机器人配置 |
| `robot_mjlab/` | 树莓派上的服务、手机页面、超声波、小智桥接 |
| `xiaozhi/` | 小智固件补丁（计时、本地指令） |
| `training/` | 按真机参数重新训练的补丁（Open_Duck_Playground） |
| `windows/` | Windows 一键启动行走 / 关机 |

## 键盘遥控行走

把 `robot/` 里的文件复制到树莓派的 `~/Open_Duck_Mini_Runtime/scripts/`，`walk.sh` 放到家目录：

```bash
ssh -t duck@<树莓派IP> "~/walk.sh"
```

默认设置：腿部 P = 20、方向微调 0.1、陀螺仪保持方向、倾斜超过 40° 自动卸力、每一步记录到 `~/walk_logs/`。

按键：`p` 开始/暂停 · `w`/`s` 前后（按两次 = 0.1 m/s，官方模型 0.05 不迈步）· `a`/`d` 横移 · `q`/`e` 转向 · 空格 停下但保持平衡 · `z`/`c` 方向微调 · `g` 保持方向开关 · `r` 摔倒后恢复 · `x` 退出并保存。

分析一次记录：

```bash
python tools/analyze_walk.py walk_20261025_160109.pkl --plot walk.png
```

## 重新训练（真机版）

在 Open_Duck_Playground 目录用它的虚拟环境依次运行：

```bash
python training/make_real_xml.py 20 8 2.0   # 舵机刚度按 P=20/头部 P=8 换算，最大力矩 2.0 N·m
python training/patch_real.py               # 新任务 flat_terrain_backlash_real（原任务不受影响）
python training/patch_sigma.py
python playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash_real --num_timesteps 160000000 --output_dir checkpoints_real
```

改动：真实舵机刚度和力矩（并随机化）、去掉横向速度死区、更严格的速度跟踪、零速度指令 20%、"站着别动"惩罚加强、每 10 秒换一次指令。结果还在验证中。
