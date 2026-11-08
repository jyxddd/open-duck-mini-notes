"""
XiaoZhi voice -> duck commands: an MCP server that dials out to the xiaozhi.me
endpoint (no port opened at home) and forwards tool calls to duck_server /cmd.

Only safe commands exist as tools. power_on, start and relax are left out on
purpose: a misheard word must not put torque on the servos or drop the duck, and
"start" needs a person to check the duck is on the floor with hands off.
duck_poweroff shuts the Pi down, only with the servos already off and only after
the user answered yes to "is it on the stand" within POWEROFF_CONFIRM_S.

The endpoint address holds a token, so it lives in a private file, never in git:
    ~/.xiaozhi_mcp_url      one line, wss://api.xiaozhi.me/mcp/?token=...  (chmod 600)

    python xiaozhi_bridge.py            # connect and serve, reconnects by itself
    python xiaozhi_bridge.py --url ws://127.0.0.1:9000 --duck http://127.0.0.1:8799   # test
"""

import argparse
import asyncio
import json
import os
import threading
import time
import urllib.request

import websockets

URL_FILE = os.path.expanduser("~/.xiaozhi_mcp_url")
WALK_MAX_S = 5.0  # a misheard number must not walk it off the table
TURN_MAX_DEG = 90.0
POWEROFF_CONFIRM_S = 30.0
POWEROFF_MIN_S = 3.0
POWEROFF_STATES = ("off", "relaxed")
RETRY_S = (1, 2, 5, 10, 30)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOLS = [
    {"name": "duck_stop",
     "description": "让鸭子立刻停止走路、站稳。用户说停、站住、别动时调用。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "duck_walk",
     "description": "让鸭子向前走一小段。只能前进，不能后退。",
     "inputSchema": {"type": "object",
                     "properties": {"seconds": {"type": "number", "description": f"走几秒，1 到 {WALK_MAX_S:.0f}"}}}},
    {"name": "duck_turn",
     "description": "让鸭子边走边转弯。左转为正，右转为负。",
     "inputSchema": {"type": "object",
                     "properties": {"degrees": {"type": "number", "description": f"角度，-{TURN_MAX_DEG:.0f} 到 {TURN_MAX_DEG:.0f}"}},
                     "required": ["degrees"]}},
    {"name": "duck_poweroff",
     "description": "用户说要关机时调用。它只询问，不会关机，把返回的问题问用户。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "duck_poweroff_yes",
     "description": "刚问过鸭子是否放在架子上，用户回答是、对、放好了时调用，这才真正关机。用户没回答或回答不是时不要调用。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "duck_head",
     "description": "让鸭子点头、摇头、歪头或抬头后仰。鸭子要已开始运行、站着不动。",
     "inputSchema": {"type": "object",
                     "properties": {"gesture": {"type": "string", "enum": ["nod", "shake", "tilt", "up"], "description": "nod 点头，shake 摇头，tilt 歪头，up 抬头后仰"}},
                     "required": ["gesture"]}},
    {"name": "duck_status",
     "description": "查询鸭子的状态、电池电压和前方障碍物距离。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "duck_squat",
     "description": "让鸭子蹲下（蹲着休息）。鸭子要已开始运行、站在地上。蹲下后走路、转弯会先让它站起来。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "duck_stand_up",
     "description": "让蹲着的鸭子站起来。",
     "inputSchema": {"type": "object", "properties": {}}},
]


def duck_get(base):
    return json.load(opener.open(base + "/status", timeout=3))


def duck_cmd(base, **cmd):
    req = urllib.request.Request(base + "/cmd", json.dumps(cmd).encode(), {"Content-Type": "application/json"})
    return json.load(opener.open(req, timeout=3))


# (pitch, yaw) in rad, turned at up to 0.5 rad/s, pitch 0.12 rad/s
GESTURES = {"nod": [(-0.15, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.0, 0.0)],
            "shake": [(0.0, 0.4, 0.0), (0.0, -0.4, 0.0), (0.0, 0.0, 0.0)],
            "tilt": [(0.0, 0.0, 0.25), (0.0, 0.0, -0.25), (0.0, 0.0, 0.0)],
            "up": [(0.75, 0.0, 0.0), (0.75, 0.0, 0.0), (0.0, 0.0, 0.0)]}
GESTURE_STEP_S = {"nod": 2.5, "shake": 1.5, "tilt": 1.5, "up": 3.5}
poweroff_asked = 0.0


def gesture(base, name):
    """Runs in its own thread: a stop must not wait for the head."""
    for pitch, yaw, roll in GESTURES[name]:
        duck_cmd(base, action="head", pitch=pitch, yaw=yaw, roll=roll)
        time.sleep(GESTURE_STEP_S[name])


def poweroff(base, yes):
    """duck_poweroff asks if it is on the stand, the yes tool then shuts the Pi down.
    POWEROFF_MIN_S: the question takes longer to say, so the model cannot chain the two."""
    global poweroff_asked
    s = duck_get(base)
    if s["state"] not in POWEROFF_STATES:
        poweroff_asked = 0.0
        return f"现在是 {s['state']} 状态，舵机还带力，不能关机。请先在手机上暂停，放回支架，再卸力。", True
    waited = time.time() - poweroff_asked
    if not yes or not POWEROFF_MIN_S <= waited <= POWEROFF_CONFIRM_S:
        poweroff_asked = time.time()
        return "鸭子放在架子上了吗？放好了请说是，我再关机。", False
    poweroff_asked = 0.0
    r = duck_cmd(base, action="shutdown")
    return ("正在关机，等绿灯灭了再断电池。", False) if r.get("ok") else (r.get("error", "关机失败"), True)


def call_tool(base, name, args):
    """Returns (text, is_error)."""
    if name == "duck_status":
        s = duck_get(base)
        if not s.get("has_us"):
            front = "没有超声波"
        elif s.get("us_silent") or s.get("distance_cm") is None:
            front = "超声波没有读数"
        else:
            front = f"前方 {s['distance_cm']} 厘米"
        return f"状态 {s['state']}，电池 {s['battery_v']} 伏，{front}，{s['result']}", False
    if name in ("duck_squat", "duck_stand_up"):
        sit = name == "duck_squat"
        r = duck_cmd(base, action="posture", sit=sit)
        if r.get("ok"):
            return ("好的，先站稳两秒再蹲下" if sit else "好的，站起来"), False
        return r.get("error", "现在不能"), True
    if name == "duck_head":
        if args.get("gesture") not in GESTURES:
            return "只会点头、摇头、歪头和抬头", True
        r = duck_cmd(base, action="head", pitch=0.0, yaw=0.0)
        if not r.get("ok"):
            return r.get("error", "现在不能动头"), True
        threading.Thread(target=gesture, args=(base, args["gesture"]), daemon=True).start()
        return "好的", False
    if name == "duck_poweroff":
        return poweroff(base, False)
    if name == "duck_poweroff_yes":
        return poweroff(base, True)
    if name == "duck_stop":
        r = duck_cmd(base, action="stop")
    elif name == "duck_walk":
        sec = min(max(float(args.get("seconds", 2)), 0.5), WALK_MAX_S)
        r = duck_cmd(base, action="walk", seconds=sec)
    elif name == "duck_turn":
        deg = min(max(float(args["degrees"]), -TURN_MAX_DEG), TURN_MAX_DEG)
        r = duck_cmd(base, action="turn", degrees=deg)
    else:
        return f"没有这个工具：{name}", True
    return (r.get("message", "好的"), False) if r.get("ok") else (r.get("error", "失败"), True)


def handle(base, msg):
    """One JSON-RPC message -> reply dict, or None for a notification."""
    method, rid = msg.get("method"), msg.get("id")
    if rid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "open-duck", "version": "1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            text, err = call_tool(base, p.get("name"), p.get("arguments") or {})
        except (OSError, ValueError, KeyError) as e:
            text, err = f"鸭子没有响应：{e}", True
        print(f"[call] {p.get('name')} {p.get('arguments')} -> {text}", flush=True)
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


async def serve(url, base):
    tries = 0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print("connected to the endpoint", flush=True)
                tries = 0
                async for raw in ws:
                    reply = await asyncio.to_thread(handle, base, json.loads(raw))
                    if reply is not None:
                        await ws.send(json.dumps(reply, ensure_ascii=False))
        except (OSError, websockets.WebSocketException, json.JSONDecodeError) as e:
            print(f"disconnected: {type(e).__name__}", flush=True)  # the message may quote the url with its token
        await asyncio.sleep(RETRY_S[min(tries, len(RETRY_S) - 1)])
        tries += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="default: the address in ~/.xiaozhi_mcp_url")
    ap.add_argument("--duck", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    url = a.url or open(URL_FILE, encoding="utf-8-sig").read().strip()
    asyncio.run(serve(url, a.duck))


if __name__ == "__main__":
    main()
