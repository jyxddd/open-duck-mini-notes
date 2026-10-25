"""Make the 'real duck' variant of the backlash model (original files untouched).
Feetech P register -> MuJoCo kp: the BAM-identified kp 17.11 belongs to P = 32
(factory value), so kp = 17.11 * P / 32. Runtime now: legs P 20, head P 8."""
import re, sys
from pathlib import Path
X = Path.home() / "Open_Duck_Playground/playground/open_duck_mini_v2/xmls"
LEG_P, HEAD_P, FORCE = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
kp_leg, kp_head = 17.11 * LEG_P / 32, 17.11 * HEAD_P / 32
src = (X / "open_duck_mini_v2_backlash.xml").read_text()
s = src.replace('<position kp="17.11" kv="0.0" forcerange="-3.23 3.23"/>',
                f'<position kp="{kp_leg:.2f}" kv="0.0" forcerange="-{FORCE} {FORCE}"/>')
assert s != src, "leg actuator default not found"
# head class: same joint dynamics, softer actuator
head_cls = (f'    <default class="sts3215_head">\n'
            f'      <position kp="{kp_head:.2f}" kv="0.0" forcerange="-{FORCE} {FORCE}"/>\n'
            f'    </default>\n')
s = s.replace('    <default class="backlash">', head_cls + '    <default class="backlash">', 1)
n = 0
for j in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
    s, k = re.subn(rf'<position class="sts3215" name="{j}"', f'<position class="sts3215_head" name="{j}"', s)
    n += k
assert n == 4, n
(X / "open_duck_mini_v2_backlash_real.xml").write_text(s)
scene = (X / "scene_flat_terrain_backlash.xml").read_text().replace(
    "open_duck_mini_v2_backlash.xml", "open_duck_mini_v2_backlash_real.xml")
(X / "scene_flat_terrain_backlash_real.xml").write_text(scene)
print(f"legs kp {kp_leg:.2f}, head kp {kp_head:.2f}, forcerange ±{FORCE}")
