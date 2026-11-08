# Voice and speaker retrofit (XiaoZhi)

[中文版](语音音响改造记录.md)

2026-10-26 to 11-08. Giving Open Duck a voice called "Xiaodou": say "你好小豆" (ni hao xiao dou) and it can stop, walk, turn, nod, squat and report its battery by voice.

## Status (2026-11-08)

| Item | Status |
| --- | --- |
| Wake word | "你好小豆" (custom firmware) |
| Voice commands | stop, walk forward (max 5 s), turn (max 90°), nod / shake / tilt / look up, status (battery, distance ahead), squat, stand up, power off (two-step confirmation) |
| Deliberately not exposed | power on, start, relax: phone only, so the duck cannot be woken up by voice when it is not set down safely |
| Mounting | speaker and voice board both on the chest front plate, no wires through the neck |
| Power | taken after the power board's switch, through a 5 V buck module to the voice board's header |

## Hardware

- **Voice board:** Waveshare ESP32-S3-AUDIO-Board (Φ58 round PCB, two microphones MIC1/MIC2, ES8311/ES7210 audio codecs, SPK speaker socket and BAT socket on the back, 2×9 pin header). Bought as the kit with a speaker box; only the PCB and the speaker are used.
- **Speaker:** the kit's driver, about Ø40, 43 g.
- **Buck module:** a red USB car-charger board (6–24 V in, fixed 5 V, USB-A out), measured 5.03 V.
- **5 V lead:** a charge-only USB-A cable with just red and black wires, cut open: red → 5V, black → GND.

## Wiring

1. **Power tap:** the two WAGO connectors of the power board's output 2 (in parallel with the Pi's UBEC), after the power switch. Switching the duck off also switches the voice board off, so it never drains the battery unnoticed.
2. **Buck input:** the + / − pads on the back, at the end farthest from the USB socket. Never swap them.
3. **Output:** USB socket → charge cable → female Dupont → the board header's **upper row**, at the MIC1 end: the outermost pin is **5V**, the one next to it is **GND** (the speaker socket is at the other end).
4. **Measure first:** check the buck output alone (5.0–5.3 V) before plugging in the voice board.
5. **Note:** do not plug the board's USB-C in while the duck is on, the two 5 V sources would fight. To flash firmware, switch the duck off first, then plug USB in.
6. Switch power off before plugging or unplugging anything.

## Mounting (chest front plate)

- **Speaker housing:** a 5 mm "mouth coupon" was printed first: lip Ø34, cone Ø39 → Ø43 within 8 mm, outer Ø49 = the speaker hole in the cover. The driver pushed in from behind wedges in the cone just right. Once the coupon fitted, the real housing's front end copied these sizes.
- **Board mount:** one piece with the speaker housing, 3 printed snap hooks hold the board plus one M3 screw (heat-set insert) as a third point. Microphones face out of the cover; the mic holes were drilled in the cover by hand.
- **Pitfalls (first version):**
  1. the housing front came out flat, with no cone, so the driver could not be pushed in;
  2. the cover's screw posts are really 12 mm (not 10 mm), leaving a 2 mm gap between housing and cover;
  3. the hooks overlapped only 0.1 mm, so they did not latch.
  Version 2 fixed all three, but the user chose not to reprint: the yellow posts were cut 2 mm shorter and the hooks replaced by screws.
- **Lesson:** sizes estimated from photos can be 2 mm off; test key interfaces with a thin coupon first (0.6 mm, 3 layers, labels engraved 0.2 mm deep).

## Weight and centre of mass

| Part | Weight |
| --- | --- |
| Front plate itself | 43 g |
| Speaker + voice board + mount | 58 g |
| Total | 101 g |

58 g at the very front moves the whole-robot CoM about 1.9 mm forward. The new head cover fitted at the same time weighs under 80 g, about 170 g less than the old Heima top + bottom cover with bearing (247 g); together the model gets about 110 g lighter with the CoM about 1.3 mm forward, which helps walking (in simulation, falls only happened with the CoM further back). A model was later retrained with the real masses (od27).

## Firmware

- First the official Waveshare v2.4.0 (esptool, merged binary at 0x0), Wi-Fi setup and activation on xiaozhi.me (account, Wi-Fi password and activation code handled by the user).
- Own firmware 9.4.0 (patches in `xiaozhi/`):
  - only the wake word "你好小豆" (custom multinet phrase, no noise reduction, one microphone channel);
  - local "停下" (stop): no cloud, an HTTP stop to the duck over the LAN;
  - countdown timer tool: the cloud says one sentence at the end (keep the text short, the cloud refuses long ones);
  - microphone gain tool.
- While testing voice, close any serial (COM) port the PC holds open on the board; an open port disturbs it.

## Connecting it to the duck (MCP bridge)

- The Pi runs `robot_mjlab/xiaozhi_bridge.py` (systemd service `xiaozhi-bridge`, starts at boot): it dials out to the MCP endpoint of the XiaoZhi console and turns voice tool calls into `/cmd` commands of the duck service. No port has to be opened at home.
- **The endpoint address is a key**: keep it only in a private file on the Pi, never in chat, screenshots or the repository.
- Tool design:
  - each tool does one safe thing; walk and turn are capped;
  - actions that need confirmation (power off) are split into two tools: the first only asks, the second within 3–30 s executes. A boolean parameter for confirmation does not work, the language model does not fill it;
  - the status tool also reports the ultrasonic distance ahead.

## Lessons

1. **The voice "stop" is not an emergency stop**: speech to action can take over ten seconds. Use the local "停下" or the phone for a real stop.
2. **Xiaodou records the list of available tools when a conversation starts and does not refresh it during that conversation.** Wake it while the duck has just been switched on and the Pi is not connected yet, and that conversation never sees the duck's tools. After switching on, wait about 1 minute, say "拜拜" (bye) to end the old conversation, then wake it again.
3. The XiaoZhi console's conversation history shows which tool was called each time, very handy for debugging.
4. Get the whole chain (wake → cloud → Pi → servos) working on the desk before fitting it into the duck.
5. Test the wake range again after mounting it in the chest: the shell muffles sound.
