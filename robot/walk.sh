#!/bin/bash
# Start the keyboard walk with the default (best) settings. Extra args are passed on.
cd ~/Open_Duck_Mini_Runtime/scripts || exit 1
source ~/.virtualenvs/open-duck-mini-runtime/bin/activate
exec python v2_rl_walk_keyboard.py --onnx_model_path BEST_WALK_ONNX_2.onnx "$@"
