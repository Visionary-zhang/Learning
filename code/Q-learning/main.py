from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from q_learning import QLearningConfig, evaluate, save_policy, save_training_log, train


def parse_args() -> QLearningConfig:
    parser = argparse.ArgumentParser(description="Tabular Q-learning practice.")
    parser.add_argument("--env-id", default="FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--slippery", action="store_true")
    args = parser.parse_args()

    return QLearningConfig(
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        is_slippery=args.slippery,
        eval_episodes=args.eval_episodes,
    )


def main() -> None:
    config = parse_args()
    output_dir = Path("outputs") / config.env_id

    q_table, rewards, lengths, mean_squared_td_errors = train(config)
    metrics = evaluate(config, q_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "q_table.npy", q_table)
    save_training_log(output_dir / "training_log.csv", rewards, lengths, mean_squared_td_errors)
    save_policy(output_dir / "policy.txt", q_table, config.env_id)

    print("Q-learning finished")
    print(f"env_id: {config.env_id}")
    print(f"episodes: {config.episodes}")
    print(f"mean_train_return_last_100: {float(np.mean(rewards[-100:])):.4f}")
    print(f"mean_td_mse_last_100: {float(np.mean(mean_squared_td_errors[-100:])):.8f}")
    print(f"eval_mean_return: {metrics['mean_return']:.4f}")
    print(f"eval_success_rate: {metrics['success_rate']:.4f}")
    print(f"eval_mean_length: {metrics['mean_length']:.2f}")
    print(f"saved_q_table: {output_dir / 'q_table.npy'}")
    print(f"saved_training_log: {output_dir / 'training_log.csv'}")
    print(f"saved_policy: {output_dir / 'policy.txt'}")


if __name__ == "__main__":
    main()
