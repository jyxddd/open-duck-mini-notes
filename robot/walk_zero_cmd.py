"""
Run the official walk policy with all commands = 0 (balance / step in place),
started from the air and released on the ground by a signal file.

1. Duck in the air (stand). Run:  python walk_zero_cmd.py --seconds 20
   -> RLWalk.turn_on() puts the legs in the init pose, then the policy waits PAUSED.
2. Put the duck on the ground, then (from another shell):  touch /tmp/duck_go
3. Policy runs for --seconds, then pauses again (motors keep holding).
"""
import argparse
import os
import time
from threading import Thread

from v2_rl_walk_mujoco import RLWalk

GO = "/tmp/duck_go"

p = argparse.ArgumentParser()
p.add_argument("--onnx_model_path", default="BEST_WALK_ONNX_2.onnx")
p.add_argument("--seconds", type=float, default=20)
p.add_argument("--wait_max", type=float, default=120, help="give up if no go signal")
p.add_argument("--cutoff_frequency", type=float, default=None)
a = p.parse_args()

if os.path.exists(GO):
    os.remove(GO)

rl = RLWalk(a.onnx_model_path, commands=False, cutoff_frequency=a.cutoff_frequency)
rl.last_commands = [0.0] * 7
rl.paused = True
Thread(target=rl.run, daemon=True).start()
print("READY: init pose held, waiting for", GO, flush=True)

t0 = time.time()
while not os.path.exists(GO):
    if time.time() - t0 > a.wait_max:
        print("no go signal, exiting (motors keep holding)", flush=True)
        raise SystemExit
    time.sleep(0.05)

print("GO", flush=True)
rl.paused = False
time.sleep(a.seconds)
rl.paused = True
time.sleep(0.3)
os.remove(GO)
print(f"DONE after {a.seconds:.0f} s, paused (motors keep holding)", flush=True)
