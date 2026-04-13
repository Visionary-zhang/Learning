import random
from collections import defaultdict


class GridWorld:
    """
    一个简单的 3x3 GridWorld：
    - 状态: 0~8
    - 动作:
        0: up
        1: right
        2: down
        3: left
        4: stay
    - 终点: state 8
    - forbidden: state 5
    - 撞边界: reward = -1
    - 进入 forbidden: reward = -1
    - 到达 target: reward = +1
    - 其他普通移动: reward = 0
    """

    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.n_states = self.rows * self.cols
        self.n_actions = 5

        self.target_state = 8
        self.forbidden_state = 5

    def state_to_pos(self, state: int) -> tuple[int, int]:
        return divmod(state, self.cols)

    def pos_to_state(self, row: int, col: int) -> int:
        return row * self.cols + col

    def is_terminal(self, state: int) -> bool:
        # 这里我们把 target 作为终止状态
        return state == self.target_state

    def step(self, state: int, action: int) -> tuple[int, float]:
        """
        给定当前 state 和 action，返回:
        next_state, reward
        """
        if self.is_terminal(state):
            # 终止状态就停在原地
            return state, 0.0

        row, col = self.state_to_pos(state)
        next_row, next_col = row, col

        if action == 0:      # up
            next_row -= 1
        elif action == 1:    # right
            next_col += 1
        elif action == 2:    # down
            next_row += 1
        elif action == 3:    # left
            next_col -= 1
        elif action == 4:    # stay
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        # 撞边界：弹回原地
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return state, -1.0

        next_state = self.pos_to_state(next_row, next_col)

        # 进入 forbidden
        if next_state == self.forbidden_state:
            return next_state, -1.0

        # 到达 target
        if next_state == self.target_state:
            return next_state, 1.0

        return next_state, 0.0

    def all_states(self) -> list[int]:
        return list(range(self.n_states))

    def all_actions(self) -> list[int]:
        return list(range(self.n_actions))


def argmax_random_tie(values: list[float]) -> int:
    """
    从多个最大值中随机选一个，避免总是偏向前面的动作
    """
    max_value = max(values)
    candidates = [i for i, v in enumerate(values) if v == max_value]
    return random.choice(candidates)


def create_random_deterministic_policy(env: GridWorld) -> dict[int, int]:
    """
    初始化一个确定性策略:
    policy[state] = action
    """
    policy = {}
    for s in env.all_states():
        if env.is_terminal(s):
            policy[s] = 4  # terminal 默认 stay
        else:
            policy[s] = random.choice(env.all_actions())
    return policy


def generate_episode_with_exploring_starts(
    env: GridWorld,
    policy: dict[int, int],
    max_steps: int = 50
) -> list[tuple[int, int, float]]:
    """
    生成一个 episode，形式为:
    [(s0, a0, r1), (s1, a1, r2), ..., (s_t, a_t, r_{t+1})]

    Exploring Starts:
    - 随机选起始 state
    - 随机选起始 action
    """
    # 随机选一个非终止起点 state
    candidate_states = [s for s in env.all_states() if not env.is_terminal(s)]
    state = random.choice(candidate_states)

    # 随机选起始 action
    action = random.choice(env.all_actions())

    episode = []

    for step_idx in range(max_steps):
        next_state, reward = env.step(state, action)
        episode.append((state, action, reward))

        if env.is_terminal(next_state):
            break

        state = next_state
        action = policy[state]  # 后续动作按当前策略走

    return episode


def mc_exploring_starts(
    env: GridWorld,
    num_episodes: int = 5000,
    gamma: float = 0.9,
    max_steps_per_episode: int = 50
):
    """
    MC Exploring Starts (First-Visit) 完整实现
    """
    # 1. 初始化策略
    policy = create_random_deterministic_policy(env)

    # 2. 初始化 Q 表
    Q = defaultdict(lambda: 0.0)

    # 3. 记录每个 (s,a) 历史上所有回报，用于取平均
    returns = defaultdict(list)

    for episode_idx in range(num_episodes):
        # === Episode generation ===
        episode = generate_episode_with_exploring_starts(
            env=env,
            policy=policy,
            max_steps=max_steps_per_episode
        )

        # === Policy evaluation + policy improvement ===
        G = 0.0
        visited_pairs_in_this_episode = set()

        # 从后往前扫描
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t_plus_1 = episode[t]

            # 递推 return: G <- gamma * G + r_{t+1}
            G = gamma * G + r_t_plus_1

            # first-visit 判断：
            # 只有当 (s_t, a_t) 是“这一回合第一次出现”的时候才更新
            # 因为我们现在是从后往前扫，所以：
            # 如果这个 pair 还没在 visited_pairs_in_this_episode 中，
            # 说明这是从前往后看时第一次访问到它的位置。
            if (s_t, a_t) not in visited_pairs_in_this_episode:
                visited_pairs_in_this_episode.add((s_t, a_t))

                # 记录 return
                returns[(s_t, a_t)].append(G)

                # Q(s,a) = average(Returns(s,a))
                Q[(s_t, a_t)] = sum(returns[(s_t, a_t)]) / len(returns[(s_t, a_t)])

                # 策略改进：对当前 state 贪心
                q_values = [Q[(s_t, a)] for a in env.all_actions()]
                best_action = argmax_random_tie(q_values)
                policy[s_t] = best_action

        if (episode_idx + 1) % 1000 == 0:
            print(f"Episode {episode_idx + 1}/{num_episodes} completed.")

    return policy, Q, returns


def print_policy(env: GridWorld, policy: dict[int, int]):
    """
    以箭头形式打印策略
    """
    action_symbols = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
        4: "·"
    }

    print("\nLearned Policy:")
    for r in range(env.rows):
        row_symbols = []
        for c in range(env.cols):
            s = env.pos_to_state(r, c)

            if s == env.target_state:
                row_symbols.append("T")
            elif s == env.forbidden_state:
                row_symbols.append("F")
            else:
                row_symbols.append(action_symbols[policy[s]])
        print(" ".join(row_symbols))


def print_q_table(env: GridWorld, Q):
    """
    打印每个 state 的各动作 Q 值
    """
    action_names = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
        4: "stay"
    }

    print("\nQ-table:")
    for s in env.all_states():
        values = []
        for a in env.all_actions():
            values.append(f"{action_names[a]}={Q[(s, a)]:.2f}")
        print(f"State {s}: " + ", ".join(values))


if __name__ == "__main__":
    random.seed(42)

    env = GridWorld()

    policy, Q, returns = mc_exploring_starts(
        env=env,
        num_episodes=5000,
        gamma=0.9,
        max_steps_per_episode=50
    )

    print_policy(env, policy)
    print_q_table(env, Q)