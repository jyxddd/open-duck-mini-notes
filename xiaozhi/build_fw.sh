#!/bin/bash
# Builds the patched XiaoZhi firmware for the ESP32-S3-AUDIO-Board in WSL. Only compiles,
# it does not touch the board. Full log in ~/esp/build.log, the last lines are printed.
cd ~/esp/xiaozhi-esp32 || exit 1
. ~/esp/esp-idf/export.sh > /dev/null
rm -f releases/v*_waveshare-esp32-s3-audio-board.zip
python scripts/release.py waveshare/esp32-s3-audio-board > ~/esp/build.log 2>&1
code=$?
grep -a -n -E "error:|FAILED|ninja: build stopped" ~/esp/build.log | head -12
tail -6 ~/esp/build.log
ls -la build/xiaozhi.bin build/merged-binary.bin 2>/dev/null
echo "build finished, exit code $code"
