# Q-learning Practice

理论公式，与书上的书写有点不同
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

- `s`: current state
- `a`: action taken in the current state
- `r`: immediate reward
- `s'`: next state
- `alpha`: learning rate
- `gamma`: discount factor
- `max_a' Q(s', a')`: best estimated future value from the next state

## Run

Activate the project environment first:

```bash
cd ~/reinforcement
source .venv-rl/bin/activate
```

Then run:

```bash
cd ~/reinforcement/reinforcement/Q-learning
python main.py
```

Try the stochastic version:

```bash
python main.py --slippery
```

Try CliffWalking:

```bash
python main.py --env-id CliffWalking-v0 --episodes 10000
```

## Outputs
`q_table.npy`: learned Q table
`training_log.csv`: reward and episode length for each episode
`policy.txt`: greedy policy after training
