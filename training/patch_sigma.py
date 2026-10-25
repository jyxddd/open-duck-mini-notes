from pathlib import Path
j = Path.home() / "Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py"
s = j.read_text()
old1 = "        relative_lin_tracking=False,  # reward_tracking_lin_vel_rel instead of the plain one\n"
new1 = old1 + "        lin_tracking_sigma=0.0,  # 0 = use reward_config.tracking_sigma\n"
old2 = "    config.relative_lin_tracking = True\n"
new2 = (old2 + "    # sharper: standing under a 0.05 m/s command scores 0.61 instead of 0.78,\n"
               "    # so stepping is worth ~2x more for small commands\n"
               "    config.lin_tracking_sigma = 0.005\n")
old3 = '''            )(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma,
            ),'''
new3 = '''            )(
                info["command"],
                self.get_local_linvel(data),
                self._config.lin_tracking_sigma or self._config.reward_config.tracking_sigma,
            ),'''
for o, n in ((old1, new1), (old2, new2), (old3, new3)):
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n)
j.write_text(s)
print("ok")
