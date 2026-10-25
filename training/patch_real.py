"""Add the 'flat_terrain_backlash_real' task. Original tasks keep their behaviour."""
from pathlib import Path
P = Path.home() / "Open_Duck_Playground/playground"

def patch(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, f"{path.name}: expected {count}x {old[:50]!r}, found {s.count(old)}"
    path.write_text(s.replace(old, new))

# 1. task -> xml
c = P / "open_duck_mini_v2/constants.py"
patch(c, 'ROUGH_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_rough_terrain_backlash.xml"\n',
      'ROUGH_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_rough_terrain_backlash.xml"\n'
      '# our duck: servo gains of the real runtime (legs P 20, head P 8), real torque\n'
      'FLAT_TERRAIN_BACKLASH_REAL_XML = ROOT_PATH / "xmls" / "scene_flat_terrain_backlash_real.xml"\n')
patch(c, '        "rough_terrain_backlash": ROUGH_TERRAIN_BACKLASH_XML,\n',
      '        "rough_terrain_backlash": ROUGH_TERRAIN_BACKLASH_XML,\n'
      '        "flat_terrain_backlash_real": FLAT_TERRAIN_BACKLASH_REAL_XML,\n')

# 2. relative velocity tracking reward
r = P / "common/rewards.py"
patch(r, "def reward_tracking_ang_vel(",
'''def reward_tracking_lin_vel_rel(
    commands: jax.Array,
    local_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    """Velocity tracking measured against standing still.

    The plain exp(-err/sigma) pays a motionless duck almost fully for small
    commands (0.05 m/s -> 0.78) and has a 0.1 m/s lateral dead band, so the
    policy learned not to step for small commands. Here standing still always
    earns exactly 0, tracking earns up to 1 - exp(-|cmd|^2/sigma), and moving
    under a zero command is negative.
    """
    err = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    still = jp.sum(jp.square(commands[:2]))
    return jp.nan_to_num(jp.exp(-err / tracking_sigma) - jp.exp(-still / tracking_sigma))


def reward_tracking_ang_vel(''')

# 3. walk task options (defaults = previous behaviour) + real-duck preset
j = P / "open_duck_mini_v2/joystick.py"
patch(j, "    reward_tracking_heading,\n)",
      "    reward_tracking_heading,\n    reward_tracking_lin_vel_rel,\n)")
patch(j, "        head_range_factor=1.0,  # to make it easier\n    )\n",
      "        head_range_factor=1.0,  # to make it easier\n"
      "        relative_lin_tracking=False,  # reward_tracking_lin_vel_rel instead of the plain one\n"
      "        zero_command_prob=0.1,\n"
      "        command_resample_steps=1500,\n"
      "    )\n\n\n"
      "def real_duck_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:\n"
      '    """Preset for tasks ending in _real (our duck, 2026-10-25 tests):\n'
      "    the robot stepped in place at zero command, did not step for\n"
      "    0.05 m/s forward / 0.1 m/s sideways and never learned start/stop.\"\"\"\n"
      "    config = config.copy_and_resolve_references()\n"
      "    config.relative_lin_tracking = True\n"
      "    config.zero_command_prob = 0.2\n"
      "    config.command_resample_steps = 500  # new command every 10 s\n"
      "    config.reward_config.scales.stand_still = -1.0\n"
      "    return config\n")
patch(j, "    ):\n        super().__init__(\n            xml_path=constants.task_to_xml(task).as_posix(),",
      "    ):\n        if task.endswith(\"_real\"):\n            config = real_duck_config(config)\n"
      "        super().__init__(\n            xml_path=constants.task_to_xml(task).as_posix(),")
patch(j, 'state.info["step"] > 1500', 'state.info["step"] > self._config.command_resample_steps', count=2)
patch(j, "jax.random.bernoulli(rng4, p=0.1)", "jax.random.bernoulli(rng4, p=self._config.zero_command_prob)")
patch(j, '''            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],''', '''            "tracking_lin_vel": (
                reward_tracking_lin_vel_rel
                if self._config.relative_lin_tracking
                else reward_tracking_lin_vel
            )(
                info["command"],''')

# 4. torque-limit randomisation for the real task
z = P / "common/randomize.py"
z.write_text(z.read_text() + '''

def domain_randomize_real(model: mjx.Model, rng: jax.Array):
    """domain_randomize + per-servo torque limit x U(0.8, 1.1).

    The real STS3215 (7.4 V) stalls around 1.9-2.1 N.m and loses torque with
    battery sag and heat; the _real xml sets 2.0 N.m.
    """
    model, in_axes = domain_randomize(model, rng)

    @jax.vmap
    def rand_force(rng):
        _, key = jax.random.split(rng)
        factor = jax.random.uniform(key, shape=(model.nu,), minval=0.8, maxval=1.1)
        return model.actuator_forcerange * factor[:, None]

    forcerange = rand_force(rng)
    model = model.tree_replace({"actuator_forcerange": forcerange})
    in_axes = in_axes.tree_replace({"actuator_forcerange": 0})
    return model, in_axes
''')

# 5. runner picks the randomizer by task
u = P / "open_duck_mini_v2/runner.py"
patch(u, "        self.randomizer = randomize.domain_randomize\n",
      "        self.randomizer = (\n            randomize.domain_randomize_real\n"
      "            if args.task.endswith(\"_real\")\n            else randomize.domain_randomize\n        )\n")
print("patched OK")
