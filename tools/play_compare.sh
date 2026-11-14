#!/bin/bash
# 11-11 knee/ankle horn layout: as built ("same" = 顺拐) vs symmetric ("mirror"), with roll play at the horn side
# (tools/knee_play_sim.py). Policy od29_22250 (on the Pi). Output ~/play_cmp/<cond>_fr<f>_s<seed>.txt
cd ~/microduck_rl || exit 1
C=/mnt/d/Claude/duck/tools/mjlab_sim_check.py
ONNX=${ONNX:-~/od29_onnx/mj_od29_22250.onnx}
BASE="--vin 7.8 --clip --imu_delay 2 --backlash 1 --gyro_noise 0.1 --knee_range 1.76 --knee_torque 2.05 --new_head --head_add 40:0"
SCH="2:0,20:0.15,10:0.15:0:0.3,10:0.15:0:-0.3,3:0"
mkdir -p ~/play_cmp; rm -f ~/play_cmp/*.txt
jobs=()
for cond in rigid same_k1.5 mirror_k1.5 same_ka1.5 mirror_ka1.5 same_ka3 mirror_ka3; do
  case $cond in
    rigid) P="";; same_k1.5) P=1.5:0.3:same:knee;; mirror_k1.5) P=1.5:0.3:mirror:knee;;
    same_ka1.5) P=1.5:0.3:same:knee+ankle;; mirror_ka1.5) P=1.5:0.3:mirror:knee+ankle;;
    same_ka3) P=3:0.3:same:knee+ankle;; mirror_ka3) P=3:0.3:mirror:knee+ankle;;
  esac
  for fr in 0.5 0.8 1.2; do for s in 1 2 3 4 5 6; do
    echo "DUCK_PLAY=$P .venv/bin/python $C --onnx $ONNX --schedule $SCH $BASE --friction $fr --seed $s --gait > ~/play_cmp/${cond}_fr${fr}_s$s.txt 2>&1"
  done; done
done | xargs -P 14 -I{} bash -c "{}"
ls ~/play_cmp | wc -l
