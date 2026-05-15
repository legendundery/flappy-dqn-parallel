import csv
import os

import matplotlib.pyplot as plt


def read_reward_log(path):
    episodes = []
    iterations = []
    rewards = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            iterations.append(int(row["iteration"]))
            rewards.append(float(row["episode_reward"]))
    return episodes, iterations, rewards


def moving_average(values, window):
    if window <= 1:
        return values
    averaged = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        averaged.append(total / len(queue))
    return averaged


def plot_rewards(x1, y1, x2, y2, xlabel, title, filename, smooth_window=None):
    plt.figure(figsize=(10, 6))
    raw_alpha = 0.25 if smooth_window and smooth_window > 1 else 1.0
    plt.plot(x1, y1, alpha=raw_alpha, label="Baseline DQN Raw")
    plt.plot(x2, y2, alpha=raw_alpha, label="Target Network DQN Raw")
    if smooth_window and smooth_window > 1:
        plt.plot(x1, moving_average(y1, smooth_window), linewidth=2, label=f"Baseline DQN MA({smooth_window})")
        plt.plot(x2, moving_average(y2, smooth_window), linewidth=2, label=f"Target Network DQN MA({smooth_window})")
    plt.xlabel(xlabel)
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    baseline_path = os.path.join("training_logs", "baseline_reward_log.csv")
    target_path = os.path.join("training_logs", "target_reward_log.csv")

    baseline_episodes, baseline_iterations, baseline_rewards = read_reward_log(baseline_path)
    target_episodes, target_iterations, target_rewards = read_reward_log(target_path)

    plot_rewards(
        baseline_episodes,
        baseline_rewards,
        target_episodes,
        target_rewards,
        "Episode",
        "Reward Comparison (Episode)",
        "reward_compare_episode.png",
        smooth_window=None,
    )
    plot_rewards(
        baseline_iterations,
        baseline_rewards,
        target_iterations,
        target_rewards,
        "Iteration",
        "Reward Comparison (Iteration)",
        "reward_compare_iteration.png",
        smooth_window=None,
    )
    plot_rewards(
        baseline_episodes,
        baseline_rewards,
        target_episodes,
        target_rewards,
        "Episode",
        "Reward Comparison (Episode, MA50)",
        "reward_compare_episode_ma50.png",
        smooth_window=50,
    )
    plot_rewards(
        baseline_iterations,
        baseline_rewards,
        target_iterations,
        target_rewards,
        "Iteration",
        "Reward Comparison (Iteration, MA50)",
        "reward_compare_iteration_ma50.png",
        smooth_window=50,
    )
    plt.show()


if __name__ == "__main__":
    main()
