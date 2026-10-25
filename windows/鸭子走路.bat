@echo off
chcp 65001 >nul
echo 启动鸭子键盘行走（默认设置：kp 20，方向微调 0.1，自动保持方向，自动记录数据）
echo 鸭子先放在支架上！腿到初始姿态后放到地上，按 p 开始，w 两次 = 0.1 前进，空格停，x 退出。
ssh -t duck@192.168.50.139 "~/walk.sh"
echo.
pause
