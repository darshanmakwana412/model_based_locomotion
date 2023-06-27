import os
import torch
import pybullet
import matplotlib.pyplot as plt
from locomotion.envs.gym_envs import A1GymEnv
from scripts.dynamics import *

if not os.path.exists("./logs"):
    os.mkdir("./logs")

# Init the NN dynamics model
dynamics_function = Dynamics(
    n_in=31,
    n_hidden=500,
    n_out=19,
    depth=2
)

# Create the robot environment
env = A1GymEnv(action_limit=(0.75, 0.75, 0.75), render=False, on_rack=False)
# Set lateral friction to 0.7
pybullet.changeDynamics(env._world_dict["ground"], -1, lateralFriction=0.7)

# Train the nn dynamics model
optimizer = torch.optim.Adam(dynamics_function.parameters(), lr = 1e-4)
loss_history = train_dynamics(
    dynamics=dynamics_function,
    env=env,
    optimizer=optimizer,
    num_episodes=100,
    len_episode=100,
    device="cpu",
    pred_delta=True,      # True for predicting the change in state(pred_st_1 - st_1), False for predicting next state(pred_st_1),
    output_dir="./logs"  # directory for saving model after every 10 episodes
)

plt.plot(loss_history)
plt.ylabel("Equation Error")
plt.xlabel("Episodes")
plt.savefig("./logs/loss.png")