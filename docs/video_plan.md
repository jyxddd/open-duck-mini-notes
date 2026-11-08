# Camera plan

[中文版](视频接入方案.md)

Status: a plan, not built yet. Items marked "to verify" have not been tested on this duck.

## Two routes

| | A. Voice-board camera | B. Raspberry Pi camera |
| --- | --- | --- |
| Hardware | CAMERA connector of the ESP32-S3-AUDIO-Board | CSI connector of the Pi Zero 2W |
| Module | OV2640 or OV5640 | Camera Module 3, or AI Camera (IMX500) |
| Software | XiaoZhi firmware already has `self.camera.take_photo` | Write our own |
| What it sees | Single photos described by the cloud language model | Continuous video processed locally |
| Delay | Seconds to over ten seconds | Depends on the algorithm for a normal camera; the AI Camera infers inside the camera |
| Good for | "What is in front?", "Have a look" | Following, recognising, avoiding |
| Not good for | Real-time control | Questions that need the cloud model |

The firmware boot log shows it probing for OV2640 and OV5640 (errors, since no camera is attached), so both are enabled in the code. The connector pin count and the exact module must be checked against the board before buying; to verify.

## Two steps

### Step 1: photos from the voice board (cheap, smallest change)

1. Buy an OV2640 or OV5640 module that fits the board's connector and try it on the desk first.
2. Say "have a look at what is in front", check that the cloud takes a photo and describes it.
3. Decide where to mount it. The voice board is in the chest, so the camera faces forward from there too and no wire goes through the neck.

### Step 2: Raspberry Pi AI Camera (only when real-time features are wanted)

- The Pi Zero 2W has little compute and must also run 50 Hz control, so vision cannot run on its CPU. The AI Camera infers inside the camera, which is why it is the choice.
- Needs the Zero's narrow adapter cable.
- On the head the camera turns with the head; the cable goes through the neck, leave enough slack.
- Vision must not slow the control loop: run it in a separate process and only hand results to `duck_server`.

## Possible uses (most practical first)

| Use | Needs | Note |
| --- | --- | --- |
| Ask by voice "what do you see" | A | Cloud description, quickest to get |
| See the duck's view on the phone | B | The Pi streams to the phone page |
| Look up and turn the head toward a person | B | Head servos track, no walking, safe |
| Turn toward whoever says its name | B + voice | Good show effect |
| Follow a person | B | Needs reliable turning and avoidance first |
| Visual obstacle avoidance | B | Covers the ultrasonic gaps, hard |
| Find a charging dock or marker | B | Long term |

## Risks

- Privacy: route A uploads photos to a cloud service; decide clearly when it may take pictures at home.
- Compute and heat: vision on the Pi competes with the control loop.
- Walking vibration: images shake while walking; recognition has to be tested.
- Weight and CoM: measure the CoM again after adding a camera.

## References

- [Raspberry Pi AI Camera specifications](https://core-electronics.com.au/raspberry-pi-ai-camera-12mp-sony-imx500.html)
- [AI Camera introduction (CNX Software)](https://www.cnx-software.com/2024/10/01/raspberry-pi-ai-camera-with-sony-imx500-ai-sensor-and-rp2040-mcu/)
