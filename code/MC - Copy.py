import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class GridWorld:
    """


    状态编号:
        0 1 2
        3 4 5
        6 7 8

    动作:
        0: up
        1: right
        2: down
        3: left
        4: stay

    设定:
    - target_state = 8
    - forbidden_state = 5
    - 撞边界: reward = -1
    - 进 forbidden: reward = -1
    - 到 target: reward = +1
    - 普通走一步: reward = 0
    """

    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.n_states = self.rows * self.cols
        self.n_actions = 5

        self.target_state = 8
        self.forbidden_state = 5

    def state_to_pos(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.cols)

    def pos_to_state(self, r: int, c: int) -> int:
        return r * self.cols + c

    def is_terminal(self, s: int) -> bool:
        return s == self.target_state

    def all_states(self) -> List[int]:
        return list(range(self.n_states))

    def all_actions(self) -> List[int]:
        return list(range(self.n_actions))

    def step(self, s: int, a: int) -> Tuple[int, float]:
        if self.is_terminal(s):
            # 到终点了就别乱动了
            return s, 0.0

        r, c = self.state_to_pos(s)
        nr, nc = r, c

        if a == 0:      # up
            nr -= 1
        elif a == 1:    # right
            nc += 1
        elif a == 2:    # down
            nr += 1
        elif a == 3:    # left
            nc -= 1
        elif a == 4:    # stay
            pass
        else:
            raise ValueError(f"动作不对: {a}")

        # 撞边界：弹回去，还要扣分
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return s, -1.0

        ns = self.pos_to_state(nr, nc)

        # 走进 forbidden，也给你罚一下
        if ns == self.forbidden_state:
            return ns, -1.0

        # 走到 target，奖励 +1
        if ns == self.target_state:
            return ns, 1.0

        # 普通移动，先给 0
        return ns, 0.0

    def expected_reward(self, s: int, a: int) -> float:

        _, r = self.step(s, a)
        return r


def argmax_with_tie_break(values: List[float]) -> int:

    best_idx = 0
    best_val = values[0]
    for i in range(1, len(values)):
        if values[i] > best_val:
            best_val = values[i]
            best_idx = i
    return best_idx


def create_initial_policy(env: GridWorld) -> Dict[int, int]:
    policy = {}
    for s in env.all_states():
        policy[s] = 4
    return policy


def policy_evaluation_truncated(
    env: GridWorld,
    policy: Dict[int, int],
    v_prev: List[float],
    gamma: float,
    N_eval: int
) -> List[float]:
    """
    这一步就是 TPI 的核心：
    policy evaluation 不做到底，只做 N_eval 次

    重点：
    - N_eval = 1： Value Iteration
    - N_eval 很大：更像 Policy Iteration
    """
    # 这一轮评估的起点，直接拿上一轮的 v 来接着推
    v = v_prev[:]

    for _ in range(N_eval):
        v_new = v[:]

        for s in env.all_states():
            if env.is_terminal(s):
                # 终点 value 直接设 0，别继续滚了
                v_new[s] = 0.0
                continue

            a = policy[s]
            r = env.expected_reward(s, a)
            ns, _ = env.step(s, a)

            # Bellman expectation backup
            v_new[s] = r + gamma * v[ns]

        v = v_new

    return v


def compute_q_from_v(
    env: GridWorld,
    v: List[float],
    gamma: float
) -> Dict[Tuple[int, int], float]:
    """
    有了当前 v，就把每个 state 下每个 action 的 q 全算出来

    q(s,a) = r(s,a) + gamma * v(next_state)
    """
    q = {}

    for s in env.all_states():
        for a in env.all_actions():
            if env.is_terminal(s):
                q[(s, a)] = 0.0
            else:
                r = env.expected_reward(s, a)
                ns, _ = env.step(s, a)
                q[(s, a)] = r + gamma * v[ns]

    return q


def policy_improvement(
    env: GridWorld,
    q: Dict[Tuple[int, int], float]
) -> Dict[int, int]:
    """
    这步就是贪心策略更新：
    当前哪个 action 的 q 最大，我就选哪个
    """
    new_policy = {}

    for s in env.all_states():
        if env.is_terminal(s):
            new_policy[s] = 4
            continue

        q_values = [q[(s, a)] for a in env.all_actions()]
        best_action = argmax_with_tie_break(q_values)
        new_policy[s] = best_action

    return new_policy


def truncated_policy_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    N_eval: int = 3,
    max_iterations: int = 100,
    tol: float = 1e-8,
    verbose: bool = True
):
    """
    Truncated Policy Iteration 
    """

    policy = create_initial_policy(env)
    v = [0.0 for _ in env.all_states()]
    history = []

    for k in range(max_iterations):
        old_v = v[:]
        old_policy = policy.copy()

        # Step 1: 截断版 policy evaluation
        v = policy_evaluation_truncated(
            env=env,
            policy=policy,
            v_prev=v,
            gamma=gamma,
            N_eval=N_eval
        )

        # Step 2: 根据当前 v 反推出 q
        q = compute_q_from_v(env, v, gamma)

        # Step 3: 贪心改策略
        policy = policy_improvement(env, q)

        # 看这一轮到底变了多少
        delta = max(abs(v[s] - old_v[s]) for s in env.all_states())

        # 看策略是不是已经不动了
        policy_stable = all(policy[s] == old_policy[s] for s in env.all_states())

        history.append({
            "iteration": k + 1,
            "v": v[:],
            "policy": policy.copy(),
            "delta": delta,
            "policy_stable": policy_stable
        })

        if verbose:
            print(f"[TPI] iter={k+1:02d}, delta={delta:.10f}, policy_stable={policy_stable}")

        # 值也差不多不变了，策略也不变了，那就收工
        if delta < tol and policy_stable:
            break

    return policy, v, q, history




def policy_symbol(a: int) -> str:
    symbols = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
        4: "·"
    }
    return symbols[a]


def draw_value_heatmap(env: GridWorld, v: List[float], title: str = "State Value Heatmap"):
    """
    画 value 热力图
    一眼看哪块高，哪块低
    """
    grid = np.zeros((env.rows, env.cols), dtype=float)

    for s in env.all_states():
        r, c = env.state_to_pos(s)
        if s == env.forbidden_state:
            grid[r, c] = np.nan
        elif s == env.target_state:
            grid[r, c] = 0.0
        else:
            grid[r, c] = v[s]

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, interpolation="nearest")
    plt.colorbar(label="Value")
    plt.title(title)

    for s in env.all_states():
        r, c = env.state_to_pos(s)
        if s == env.forbidden_state:
            plt.text(c, r, "F", ha="center", va="center", fontsize=16, fontweight="bold")
        elif s == env.target_state:
            plt.text(c, r, "T", ha="center", va="center", fontsize=16, fontweight="bold")
        else:
            plt.text(c, r, f"{v[s]:.2f}", ha="center", va="center", fontsize=12)

    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def draw_policy_map(env: GridWorld, policy: Dict[int, int], title: str = "Policy Map"):
    """
    画策略图
    直接看每个格子往哪走
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)

    # 画网格
    for x in range(env.cols + 1):
        ax.axvline(x - 0.5)
    for y in range(env.rows + 1):
        ax.axhline(y - 0.5)

    for s in env.all_states():
        r, c = env.state_to_pos(s)

        if s == env.forbidden_state:
            ax.text(c, r, "F", ha="center", va="center", fontsize=18, fontweight="bold")
        elif s == env.target_state:
            ax.text(c, r, "T", ha="center", va="center", fontsize=18, fontweight="bold")
        else:
            ax.text(c, r, policy_symbol(policy[s]), ha="center", va="center", fontsize=20)

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def draw_delta_curve(history: List[dict], title: str = "Convergence Curve"):
    """
    看 delta 怎么掉下去
    这个图特别适合拿不同 N_eval 对比
    """
    iterations = [h["iteration"] for h in history]
    deltas = [h["delta"] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(iterations, deltas, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("delta = max |v_k - v_{k-1}|")
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def print_policy(env: GridWorld, policy: Dict[int, int]):
    print("\nPolicy:")
    for r in range(env.rows):
        row_syms = []
        for c in range(env.cols):
            s = env.pos_to_state(r, c)

            if s == env.forbidden_state:
                row_syms.append("F")
            elif s == env.target_state:
                row_syms.append("T")
            else:
                row_syms.append(policy_symbol(policy[s]))
        print(" ".join(row_syms))


def print_values(env: GridWorld, v: List[float]):
    print("\nState Values:")
    for r in range(env.rows):
        row_vals = []
        for c in range(env.cols):
            s = env.pos_to_state(r, c)

            if s == env.forbidden_state:
                row_vals.append("  F   ")
            elif s == env.target_state:
                row_vals.append("  T   ")
            else:
                row_vals.append(f"{v[s]:5.2f}")
        print(" ".join(row_vals))


if __name__ == "__main__":
    env = GridWorld()

#内部迭代
    N_eval = 3

    policy, v, q, history = truncated_policy_iteration(
        env=env,
        gamma=0.9,
        N_eval=N_eval,
        max_iterations=100,
        tol=1e-8,
        verbose=True
    )

    print_values(env, v)
    print_policy(env, policy)

    # ====== 可视化 ======
    draw_value_heatmap(env, v, title=f"TPI Value Heatmap (N_eval={N_eval})")
    draw_policy_map(env, policy, title=f"TPI Policy Map (N_eval={N_eval})")
    draw_delta_curve(history, title=f"TPI Convergence Curve (N_eval={N_eval})")