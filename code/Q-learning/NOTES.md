# Notes

## Task background
Q-learning is used here as the bridge from Bellman optimality equations to DQN.
Before using a neural network, we first learn a Q-table directly.

## Problem

Dynamic programming needs the transition model of the environment. In practice,
we usually do not know the exact transition probability. Q-learning uses samples
from interaction instead.

## Method

For each transition:

```text
(state, action, reward, next_state)
```

the algorithm constructs a TD target:

```text
reward + gamma * max_next_action Q[next_state, next_action]
```

Then it moves the current estimate toward that target.


## Limitation

Q-learning 是一个递增是的算法，但是递增式的算法无法对数据进行充分利用，且当状态是连续的或非常大时，它不适用。因此引入我们后面的DQN
