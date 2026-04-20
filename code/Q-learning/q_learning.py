from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np


@dataclass
class QLearningConfig:
    env_id: str = "FrozenLake-v1"
    episodes: int = 5000
    max_steps: int = 200
    learning_rate: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42
    is_slippery: bool = False
    eval_episodes: int = 100

# ʹgym.make()ʱ֮ǰ֪env_idΪFrozenLake-v1, ȥis_sl
def make_env(config: QLearningConfig):
    if config.env_id == "FrozenLake-v1":
        return gym.make(config.env_id, is_slippery=config.is_slippery)
    return gym.make(config.env_id)

# ϵ-greedy action selection
def epsilon_greedy_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    action_space: gym.Space,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(action_space.sample())
    return int(np.argmax(q_table[state]))


def train(config: QLearningConfig):
    env = make_env(config)
    rng = np.random.default_rng(config.seed)
    env.action_space.seed(config.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards = []
    lengths = []
    mean_squared_td_errors = []
    epsilon = config.epsilon_start

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        episode_squared_td_errors = []

        for step in range(1, config.max_steps + 1):
            action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            best_next_q = np.max(q_table[next_state])
            td_target = reward + config.gamma * best_next_q * (not terminated)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += config.learning_rate * td_error

            episode_squared_td_errors.append(float(td_error**2))
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards.append(total_reward)
        lengths.append(step)
        mean_squared_td_errors.append(float(np.mean(episode_squared_td_errors)))

    env.close()
    return (
        q_table,
        np.asarray(rewards),
        np.asarray(lengths),
        np.asarray(mean_squared_td_errors),
    )


def evaluate(config: QLearningConfig, q_table: np.ndarray):
    env = make_env(config)
    wins = 0
    returns = []
    lengths = []

    for episode in range(config.eval_episodes):
        state, _ = env.reset(seed=config.seed + 100_000 + episode)
        total_reward = 0.0

        for step in range(1, config.max_steps + 1):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        wins += int(total_reward > 0)
        returns.append(total_reward)
        lengths.append(step)

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(wins / config.eval_episodes),
        "mean_length": float(np.mean(lengths)),
    }


def save_training_log(
    path: Path,
    rewards: Iterable[float],
    lengths: Iterable[int],
    mean_squared_td_errors: Iterable[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("episode,reward,length,moving_avg_100,mean_squared_td_error\n")
        history = []
        for episode, (reward, length, td_mse) in enumerate(
            zip(rewards, lengths, mean_squared_td_errors),
            start=1,
        ):
            history.append(float(reward))
            moving_avg = float(np.mean(history[-100:]))
            f.write(
                f"{episode},{float(reward):.6f},{int(length)},"
                f"{moving_avg:.6f},{float(td_mse):.8f}\n"
            )


def save_policy(path: Path, q_table: np.ndarray, env_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_actions = np.argmax(q_table, axis=1)

    if env_id == "FrozenLake-v1" and q_table.shape[0] == 16:
        symbols = {0: "L", 1: "D", 2: "R", 3: "U"}
        rows = []
        for row in range(4):
            actions = [symbols[int(best_actions[row * 4 + col])] for col in range(4)]
            rows.append(" ".join(actions))
        text = "\n".join(rows) + "\n"
    else:
        text = "\n".join(f"state {s}: action {int(a)}" for s, a in enumerate(best_actions)) + "\n"

    path.write_text(text, encoding="utf-8")
