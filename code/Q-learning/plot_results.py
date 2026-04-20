from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_training_log(path: Path):
    episodes = []
    rewards = []
    moving_avg = []
    td_mse = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            moving_avg.append(float(row["moving_avg_100"]))
            td_mse.append(float(row.get("mean_squared_td_error", 0.0)))

    return (
        np.asarray(episodes),
        np.asarray(rewards),
        np.asarray(moving_avg),
        np.asarray(td_mse),
    )


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        result[i] = np.mean(values[max(0, i - window + 1) : i + 1])
    return result


def plot_training_curve(output_dir: Path) -> None:
    episodes, rewards, moving_avg, _ = read_training_log(output_dir / "training_log.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#9ca3af", linewidth=0.8, alpha=0.45, label="episode reward")
    plt.plot(episodes, moving_avg, color="#2563eb", linewidth=2.2, label="moving avg 100")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-learning Training Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close()


def plot_td_loss(output_dir: Path) -> None:
    episodes, _, _, td_mse = read_training_log(output_dir / "training_log.csv")
    td_mse_ma = moving_average(td_mse, window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, td_mse, color="#f97316", linewidth=0.8, alpha=0.35, label="episode mean squared TD error")
    plt.plot(episodes, td_mse_ma, color="#dc2626", linewidth=2.2, label="moving avg 100")
    plt.xlabel("Episode")
    plt.ylabel("Mean squared TD error")
    plt.title("Q-learning TD Error Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "td_error_curve.png", dpi=160)
    plt.close()


def plot_q_table(output_dir: Path) -> None:
    q_table = np.load(output_dir / "q_table.npy")

    plt.figure(figsize=(8, 6))
    image = plt.imshow(q_table, aspect="auto", cmap="viridis")
    plt.colorbar(image, label="Q value")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.title("Learned Q-table Heatmap")
    plt.xticks(np.arange(q_table.shape[1]))
    plt.yticks(np.arange(q_table.shape[0]))
    plt.tight_layout()
    plt.savefig(output_dir / "q_table_heatmap.png", dpi=160)
    plt.close()


def plot_policy_grid(output_dir: Path) -> None:
    q_table = np.load(output_dir / "q_table.npy")
    if q_table.shape != (16, 4):
        return

    action_symbols = {0: "<", 1: "v", 2: ">", 3: "^"}
    frozen_lake_cells = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]
    best_actions = np.argmax(q_table, axis=1).reshape(4, 4)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Greedy Policy on FrozenLake")

    for row in range(4):
        for col in range(4):
            y = 3 - row
            cell = frozen_lake_cells[row][col]
            if cell == "H":
                facecolor = "#fca5a5"
            elif cell == "G":
                facecolor = "#86efac"
            elif cell == "S":
                facecolor = "#bfdbfe"
            else:
                facecolor = "#f8fafc"

            rect = plt.Rectangle((col, y), 1, 1, facecolor=facecolor, edgecolor="#334155", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(col + 0.12, y + 0.82, cell, fontsize=11, color="#334155", ha="left", va="top")

            if cell not in {"H", "G"}:
                action = int(best_actions[row, col])
                ax.text(
                    col + 0.5,
                    y + 0.45,
                    action_symbols[action],
                    fontsize=26,
                    color="#111827",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output_dir / "policy_grid.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Q-learning training artifacts.")
    parser.add_argument("--output-dir", default="outputs/FrozenLake-v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_training_curve(output_dir)
    plot_td_loss(output_dir)
    plot_q_table(output_dir)
    plot_policy_grid(output_dir)

    print(f"saved: {output_dir / 'training_curve.png'}")
    print(f"saved: {output_dir / 'td_error_curve.png'}")
    print(f"saved: {output_dir / 'q_table_heatmap.png'}")
    if (output_dir / "policy_grid.png").exists():
        print(f"saved: {output_dir / 'policy_grid.png'}")


if __name__ == "__main__":
    main()
