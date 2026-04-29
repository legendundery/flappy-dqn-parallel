import csv
import os

import matplotlib.pyplot as plt


def read_reward_log(path):
    episodes = []
    rewards = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["episode_reward"]))
    return episodes, rewards


def main():
    baseline_path = os.path.join("training_logs", "baseline_reward_log.csv")
    target_path = os.path.join("training_logs", "target_reward_log.csv")

    baseline_episodes, baseline_rewards = read_reward_log(baseline_path)
    target_episodes, target_rewards = read_reward_log(target_path)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_episodes, baseline_rewards, label="Baseline DQN")
    plt.plot(target_episodes, target_rewards, label="Target Network DQN")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Reward Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_compare.png")
    plt.show()


if __name__ == "__main__":
    main()
