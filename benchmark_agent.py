import time
import torch
from main import agent
from kaggle_environments import make

env = make('orbit_wars', debug=False)
steps = 20
total_time = 0

# Run a match and measure agent time
env.reset()
for i in range(steps):
    obs = env.state[0].observation
    start = time.perf_counter()
    with torch.no_grad():
        move = agent(obs)
    end = time.perf_counter()
    total_time += (end - start)
    env.step([move, []]) # dummy second player

avg_time = total_time / steps
print(f"Average agent time per turn over {steps} steps: {avg_time:.4f}s")
