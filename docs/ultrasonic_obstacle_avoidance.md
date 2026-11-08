# Ultrasonic obstacle avoidance

[中文版](超声波避障记录.md)

A US-100 ultrasonic range sensor keeps Open Duck from walking into things. 2026-10-27 to 11-03.

## Status (2026-11-03)

- The sensor is on the head and turns with it, so it can look left and right before starting.
- Holding the forward button, it decides by itself when to stop and look again ("look ahead + watch while walking"), instead of walking all the way.
- Real test: holding forward toward a wall 1.9 m away, it did not hit it either time. The first version stopped too early (about 40–50 cm from the wall); the evening test checks the closer version (stops at about 20 cm).

## Hardware

- US-100 in serial mode (jumper on the back fitted), 3.3 V supply and 3.3 V output, so it connects straight to the Pi without a divider.
- Wiring: VCC → 3.3 V (pin 17), Trig/Tx → GPIO15 RXD (pin 10), Echo/Rx → GPIO14 TXD (pin 8), GND → pin 25.
- On the Pi, enable the UART and remove the serial console: add `enable_uart=1` to `/boot/firmware/config.txt`, delete `console=serial0,115200` from `cmdline.txt`.
- Protocol: 9600 baud, send `0x55` → 2 bytes of distance (mm); send `0x50` → 1 byte of temperature (+45). Code: `robot_mjlab/us100.py`, 10 readings per second.

## Pitfalls

1. **VCC and GND swapped**: 0 bytes read. Swapped back and it worked; the module survived.
2. **False readings close up.** At a real 3–10 cm, false 20 cm and 26 cm readings mix in and win the median. Fix: `nearest()` takes the second-smallest reading of the last 0.8 s, so two low readings are needed.
3. **Often no echo.** Soft (sofa, cloth), slanted or thin (table legs) objects absorb or deflect the sound. In the 11-03 logs, walking often gave 2–4 s in a row without a valid reading, standing up to 50+ s. **"No reading" used to mean "nothing ahead", so a held forward button walked into things.** That is why look-ahead was added.
4. **On the head it only sees at about 0.4 m height.** Low things (slippers, door sills) are invisible.
5. **Stopping has momentum.** From the stop command to standing still takes about 0.7 s and another 5–8 cm.

## Current logic (`robot_mjlab/duck_server.py`)

**Close-range protection (always on, also while turning and walking):**

| Parameter | Value | Purpose |
| --- | --- | --- |
| Slow-down zone | 60 cm → 15 cm | Speed scales linearly from 100 % to 60 % |
| Stop line | 15 cm | Stop when close for 0.2 s in a row |
| Latch | 0.5 s | A reading flickering around the stop line does not restart walking |
| Release | 20 cm | After a stop, forward is allowed again only beyond 20 cm |

Turning in place is never blocked, so it can turn away.

**Start scan:** before walking forward the head turns left, right, then ahead, 1.3 s each, reading the distance. Less than 25 cm ahead refuses to go forward. No new scan within 10 s of the last one.

**Look ahead + watch while walking (added 11-03):** after the scan it remembers the distance ahead and works out "how far it may walk" = distance ahead − 20 cm (80 cm if there was no echo, at most 1.5 m trusted from one reading). While walking it adds up the walked distance from the commanded speed (the command is faster than the real speed, so it over-estimates and stops early, which is safe). It stops and scans again when any of these happens:

| Condition | Note |
| --- | --- |
| It has walked the distance it saw free | The only rule left when there are no echoes on the way |
| A reading under 50 cm ahead that is over 15 cm closer than expected | Something appeared; a reading that matches the expectation (the same wall) lets it walk closer |
| The heading is over 45° off the scan heading | After a turn, the earlier distance no longer counts |

Fresh readings of more free space move the end further out, so facing a long open space it keeps walking and does not stop for no reason.

## Tests

Unit tests (`_rescan_reason`, 9 cases, margin 20 cm):

| Case | Result |
| --- | --- |
| 1.0 m ahead, no echo while walking | Walks 0.80 m, stops and looks again |
| No echo in the scan, none afterwards | Walks 0.60 m, stops and looks again |
| Wall at 1.2 m, read while walking | Walks 1.0 m (20 cm from the wall), stops and looks again |
| 2 m ahead, always read as 2 m | Does not stop |
| A chair appears at 45 cm after 0.3 m | Stops at once and looks again |
| Turned 60° after the scan | Looks again at once |
| Walking and turning, no echo, 1.0 m ahead | Walks 0.80 m, stops and looks again |
| Scan saw 45 cm, the same wall read while approaching | Walks 0.25 m, stops and looks again; now under 25 cm, no more forward |
| 1.86 m ahead, one false 26 cm reading on the way | Stops early once and looks again (conservative, not dangerous) |

Real robot (11-03, od13 kp 30, wooden floor, holding forward toward a wall for 25 s, twice, no hit):

- First: scan saw 186 cm → walked about 8 s → a false 26 cm echo, stopped early and looked again (really about 60 cm) → 98 cm ahead → walked 3.5 s → stopped at 50 cm and looked again.
- Second: no echo in all three directions → started on a 50 cm look-ahead → after 2–3 s read the 1.3 m wall, look-ahead extended → stopped at 47 cm and looked again → 45 cm ahead, above the refuse line of 40 cm at the time → walking again read under 50 cm at once → **kept scanning in place**.
- Fix: only a reading over 15 cm closer than expected counts as "something appeared"; margin 30 → 20 cm, refuse line 40 → 25 cm (the user found 40 cm too far). Unit tests of the fixed version above; real test pending.

## Possible improvements

- Use the look-ahead to filter false readings: a reading suddenly much closer than the expected remaining distance and between 20–26 cm could be treated as suspect and confirmed by a few more readings.
- Wander around the room: scan → pick the most open direction → turn in place → walk forward with look-ahead → scan again when blocked. No backing up needed, turning in place gets it out.
- Low obstacles need another sensor, e.g. a small range sensor in front of the feet.
