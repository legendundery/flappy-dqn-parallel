# Flappy DQN Parallel

This is a separate experimental repo for the homework part 3 PyTorch version.

## What changed

- keeps both `baseline` and `target` variants
- keeps reward logs in the same CSV format
- uses headless single-process training to avoid window rendering overhead
- removes FPS limiting during training

## Train

```bash
python dqn.py train baseline --iterations 50000 --log-interval 1000
python dqn.py train target --iterations 50000 --log-interval 1000
```

## Test

```bash
python dqn.py test baseline 50000
python dqn.py test target 50000
```

## Plot rewards

```bash
python plot_rewards.py
```
