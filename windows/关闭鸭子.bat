@echo off
chcp 65001 >nul
echo 正在让鸭子的树莓派关机...
ssh -o ConnectTimeout=8 duck@192.168.50.139 "cat ~/battery_status.txt; sudo poweroff"
echo.
echo 已发送关机命令。等树莓派绿灯熄灭（约 20 秒）后，再关电源开关。
pause
