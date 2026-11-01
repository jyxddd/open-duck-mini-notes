"""
Floor test: runs a list of moves over the phone service's HTTP API.

Default "box" route for ~1.5 m2 of floor, duck in the middle: quarter circle
left to a corner, half turn in place, quarter circle right back to the middle,
then the same to the opposite corner (~1.5 min, ~0.5 m from the middle at most).
"full" is the first list, it needs 5.5 m of floor.

Before starting: duck on the stand, power_on, on the floor, hands off, press
start on the phone (the script refuses anything but state "running").

    python duck_test_suite.py                 # box route
    python duck_test_suite.py --plan full --long 0   # full list without the long walk
    python duck_test_suite.py --dry           # print the list only

Stops at once on a fall, a frozen IMU, low battery, a phone pause or Ctrl+C.
Drive commands are refreshed at 5 Hz, so if this script or the Wi-Fi dies the
server's 0.5 s dead-man stops the duck by itself. At the end the duck is left
paused, relaxing is by hand: put it on the stand first.
"""

import argparse
import json
import sys
import time
import urllib.request

REFRESH_S = 0.2
FAILS_MAX = 5  # refreshes in a row
LOW_BATTERY_S = 3  # a stall or a stomp sags it for a moment, it comes back
STOP_SETTLE_S = 3
SPIN_LEAD_DEG = 5  # stop early, it keeps turning a bit
ARC_FWD, ARC_TURN = 0.1, 0.4  # ~23 deg/s with the server turn loop, circle radius ~0.3 m
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, **cmd):
    req = urllib.request.Request(url + "/cmd", json.dumps(cmd).encode(), {"Content-Type": "application/json"})
    return json.load(opener.open(req, timeout=1.5))


def status(url):
    return json.load(opener.open(url + "/status", timeout=1.5))


def spin(deg):
    return (f"原地{'左' if deg > 0 else '右'}转 {abs(deg)}°", abs(deg) / 8 + 3, 0, ARC_TURN * (1 if deg > 0 else -1), deg)


def arc(deg):
    return (f"边走边{'左' if deg > 0 else '右'}转 {abs(deg)}°", abs(deg) / 8 + 3, ARC_FWD, ARC_TURN * (1 if deg > 0 else -1), deg)


def box_plan():
    rest = ("停 2 秒", 2, 0, 0)
    return [("站立", 3, 0, 0),
            arc(90), rest, spin(180), rest, arc(-90), rest,  # out to one corner and back to the middle
            arc(90), rest, spin(180), rest, arc(-90)]  # same to the opposite corner


def full_plan(long_s):
    rest = ("停 3 秒", 3, 0, 0)
    steps = [("站立", 5, 0, 0), ("直走 10 秒，量距离", 10, 0.15, 0), rest,
             ("原地左转", 5, 0, 0.4), rest, ("原地右转", 5, 0, -0.4), rest,
             ("边走边左转", 5, 0.1, 0.4), rest, ("边走边右转", 5, 0.1, -0.4)]
    if long_s:
        steps += [rest, (f"长走 {long_s:.0f} 秒", long_s, 0.15, 0)]
    return steps


low_since = None


def why_stop(s):
    global low_since
    if s["state"] != "running":
        return f"状态变成 {s['state']}：{s['result']}"
    if not s["battery_low"]:
        low_since = None
    elif low_since is None:
        low_since = time.time()
    elif time.time() - low_since >= LOW_BATTERY_S:
        return f"电池持续偏低：{s['battery_v']} V"
    return None


def run_step(url, sec, fwd, turn, deg=None):
    """Drive for sec seconds, or until the heading has changed by deg (sec is then the time limit)."""
    t0 = time.time()
    fails = 0
    h0 = status(url)["heading_deg"]
    while time.time() - t0 < sec:
        try:
            if fwd or turn:
                r = post(url, action="drive", fwd=fwd, turn=turn)
                if not r["ok"]:
                    return f"命令被拒绝：{r['error']}"
            s = status(url)
            why = why_stop(s)
            if why:
                return why
            if deg and abs(s["heading_deg"] - h0) >= abs(deg) - SPIN_LEAD_DEG:
                post(url, action="stop")
                return None
            fails = 0
        except OSError as e:
            fails += 1
            if fails >= FAILS_MAX:
                return f"连不上鸭子：{e}"
        time.sleep(REFRESH_S)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://<树莓派IP>:8765")
    ap.add_argument("--plan", choices=("box", "full"), default="box")
    ap.add_argument("--long", type=float, default=20.0, help="s of the last long walk, 0 = skip")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    steps = box_plan() if a.plan == "box" else full_plan(a.long)
    print(f"共 {len(steps)} 步，最长约 {sum(s[1] for s in steps):.0f} 秒")
    for i, (name, sec, fwd, turn, *_) in enumerate(steps, 1):
        print(f"  {i:2d}. {name}（最多 {sec:.1f} 秒，前进 {fwd}，转向 {turn:+}）")
    if a.dry:
        return

    s = status(a.url)
    if why_stop(s):
        print(f"\n不能开始：{why_stop(s)}")
        print("先放支架上力，放到地上松手，手机按开始，再运行本程序。")
        sys.exit(1)
    print(f"\n模型 {s['model']}，电池 {s['battery_v']} V。5 秒后开始，人守在旁边，手机可随时按暂停。")
    time.sleep(5)

    why = None
    try:
        for i, (name, sec, fwd, turn, *deg) in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {name} ...", flush=True)
            if not (fwd or turn):
                post(a.url, action="stop")
            why = run_step(a.url, sec, fwd, turn, *deg)
            s = status(a.url)
            print(f"      朝向 {s['heading_deg']:+.1f}°  电池 {s['battery_v']} V", flush=True)
            if why:
                break
    except KeyboardInterrupt:
        why = "Ctrl+C"
    finally:
        try:
            if status(a.url)["state"] == "running":
                post(a.url, action="stop")
                time.sleep(STOP_SETTLE_S)  # a pause in mid-step locks the legs there and it tips over
                post(a.url, action="pause")
            print(post(a.url, action="save_log").get("message", ""))
        except OSError as e:
            print(f"收尾失败（鸭子会因收不到命令自己停下）：{e}")

    print(f"\n中途停止：{why}" if why else "\n全部完成")
    print("下一步：放回支架，再按卸力。")


if __name__ == "__main__":
    main()
