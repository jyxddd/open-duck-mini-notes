# Home-made v2 duck: simulation training summary (walk, turn, get up after a fall, sit / stand)

The home-made v2 is our own small biped duck design (not the Open Duck Mini v2). It follows the microduck layout scaled up 1.35 times: the knees bend the same way and every joint axis matches microduck, plus a head yaw joint. Servos are Feetech STS3215, with a stronger model at the knees.

All training is in simulation: MuJoCo / mjlab + rsl_rl (PPO), an RTX 4060 laptop GPU, 2048 parallel environments. By 2026-11-24 the four motions required before printing all passed acceptance. Next: print, assemble, then the real robot.

## Results

Every exam is **deterministic**: the policy's mean action, no exploration noise, no randomisation, no pushes. 64 ducks per exam, each counted on its own.

| Motion | Checkpoint | Exam | Result |
|---|---|---|---|
| Walk | G line, iteration 44000 | forward 0.20 m/s, 20 s | no falls in 64; median distance 2.35 m (best 3.38 m); every joint's correlation with microduck ≥ +0.27; straight-line yaw drift median +0.031 rad/s, the same size as microduck's (−0.031) |
| Turn left / right | same checkpoint | forward 0.10 m/s, yaw ±0.5 rad/s, 20 s | no falls in 64; left 0.29, right 0.30 rad/s, slow / fast ratio 0.98; microduck itself turns only 0.15 / 0.23 rad/s under the same command; no joint moving opposite |
| Get up after a fall | T line get-up, iteration 2000 | fall poses never seen in training: 22 on the side, 21 face down, 21 face up; success = standing steady for 2 s | 64/64 stood up, median 0.71 s; retest with randomisation 128/128, standing 95 % of the following 20 s |
| Sit down and stand up | T line sit / stand, iteration 3000 | alternating sit / stand commands, 128 sits and 128 stands | sat 108/128 (84 %), stood 128/128; the 13 ducks that fell all got up by themselves, 64/64 standing at the end |

## Two training lines

### G line: gait method (imitation + velocity tracking)
- **Reference motion**: all 13 joints recorded from microduck's official walking policy and converted to the v2 size (length × 1.35, gait cycle 0.45 s).
- **Rewards**: imitate the joints, imitate the alternating foot loading, track the commanded velocity, stay upright, smooth actions, a penalty for falling.
- **Stages**: stand → resist pushes → step in place → walk forward, then turning.
- **Fixes, one change at a time, each driven by an exam**:
  - Knees swung too little (17–19°, microduck 24–29°): a separate knee-only imitation term brought them to about 22–26°.
  - Too slow (0.10 m/s for a 0.15 command): velocity tracking tolerance 0.20 → 0.10. At 0.20, being 0.05 m/s short still earned 94 % of the term, so there was no pull to go faster.
  - After the speed-up the right hip yaw moved opposite to microduck: a hip-yaw imitation term corrected it.
  - Feet lifted too low: a foot-lift term.
  - Would not turn: the term that penalised any turning was changed to penalise only the difference between the actual and the commanded turn rate.
- About 44000 iterations in total.

### T line: free experiments (as few rewards as possible, let it find motions)
- **First task**: walk from the centre out to a 5 m circle and back, with speed as the only reward.
  - The rolling and lying-down shuffling it learned turned out to exploit simulator gaps, see [Simulation loopholes](sim_loopholes.md).
- **Getting up**: each episode starts from a fallen pose; the reward is standing up and staying up.
  - Already 64/64 after 1000 iterations.
  - Then a penalty for real self-collisions was added: the simulator lets the thigh pass through the trunk without a collision, so a knee / hip collision table was built from the CAD.
- **Sit / stand**: continued from the checkpoint that could already get up, with a sit / stand flag added to the command.
  - The sitting pose is a deep squat on the feet, trunk 0.07 m above the ground.

## Lessons
1. **Trust only the deterministic exam.** Training-time numbers (fall rate, speed) include exploration noise and pushes and can mislead. Once G was thought to be stuck and a setting was changed, while its exam scores were in fact still improving.
2. **One change at a time.** Write down the reason and the result every time, so a bad change can be undone at once.
3. **Set the entropy coefficient explicitly.** In a task whose only reward is one small term, entropy 0.01 made the action noise run away (std 1 → 47), 0.003 still crept up, and 0.001 was stable. A watcher script now stops a run when std goes above 1.5.
4. **After a reward change, start fresh if you can.** Resuming from late checkpoints broke the policy right away three times. The same checkpoint frozen inside the training environment walked fine, so the environment was not the problem; the training itself was.
5. **Replay and check for floor penetration before believing a motion.** Impressive-looking motions often exploit gaps in the simulator.
6. **Watcher scripts must wait for the new run folder before taking its path.** Once a watcher pointed at an old folder and three exams were missed.
7. **Once it walks, move on.** G walks a little slower than commanded, but it is steady and every joint moves the right way. So the next priority was sitting and getting up, the two motions that let the duck run without a person watching it.

## Before the real robot
- Getting up from face-up is a fast roll and fairly violent. Now and then the thigh brushes the trunk for an instant. Try it on a soft mat first.
- The "sit" in sit / stand is a deep squat, not sitting on the bottom. Standing up from it sometimes tips backward, but the duck gets up again by itself.

## Related
- [Simulation loopholes](sim_loopholes.md): loopholes in the simulator found during training, how each was found and how it was closed.
- 中文：[v2 仿真训练总结](v2仿真训练总结.md)
