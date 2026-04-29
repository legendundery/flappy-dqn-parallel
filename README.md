# Flappy DQN Parallel

This is a separate experimental repo for the homework part 3 PyTorch version.

## What changed

- keeps both `baseline` and `target` variants
- adds multi-process parallel data collection
- keeps reward logs in the same CSV format
- supports headless training workers to avoid window rendering overhead

## Train

```bash
python dqn.py train baseline --num-workers 8 --iterations 50000
python dqn.py train target --num-workers 8 --iterations 50000
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
