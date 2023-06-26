import torch
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Dynamics(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, depth, activation=None, drop=None):
        super(Dynamics, self).__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.depth = depth

        self.activation = activation
        if self.activation == None:
            self.activation = torch.nn.ReLU()

        self.drop = drop

        layers = []

        # Input Layer
        layers.append(('input_lin', torch.nn.Linear(self.n_in, self.n_hidden)))
        layers.append(('input_act', self.activation))
        if self.drop is not None:
            layers.append(torch.nn.Dropout(p=self.drop))

        # Hidden Layer
        for d in range(self.depth):
            layers.append(('hidden_lin_' + str(d), torch.nn.Linear(self.n_hidden, self.n_hidden)))
            layers.append(('hidden_act_' + str(d), self.activation))
            if self.drop is not None:
                layers.append(torch.nn.Dropout(p=self.drop))

        # Output Layer
        layers.append(('out_lin', torch.nn.Linear(self.n_hidden, self.n_out)))

        self.model = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, X):
        return self.model(X)

# Define the training loop
def train_dynamics(dynamics, env, optimizer, num_episodes, len_episode, device, pred_delta, output_dir):
    dynamics.train()
    dynamics.to(device)
    loss_function = torch.nn.MSELoss(reduction='mean')
    losses = []

    torch.save(dynamics.state_dict(), f"{output_dir}/dynamics_{0}.pt")

    for episode_idx in range(num_episodes):
        st = env.reset()
        st = torch.from_numpy(st).to(device).float()
        train_loss = 0.0
        loss = 0
        for step_idx in range(len_episode):
            # Sample a random action at any given state
            at = env.action_space.sample()
            # Step the environment using the action
            st_1, reward, done, info = env.step(at)

            # Concatenate the state and action pairs to form the input
            X = np.concatenate((st, at), axis=0)
            X = torch.from_numpy(X).to(device).float()
            st_1 = torch.from_numpy(st_1).to(device).float()

            # Perform the forward pass
            pred_st_1 = dynamics(X)
            loss += ( loss_function(pred_st_1 - st, st_1 - st) if pred_delta else loss_function(pred_st_1, st_1) )

            if (step_idx+1)%16==0:
                train_loss += loss.item()
                # Perform the backward pass and update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = 0

            st = st_1
        losses.append(train_loss/len_episode)

        if (episode_idx + 1) % 10==0:
            torch.save(dynamics.state_dict(), f"{output_dir}/dynamics_{episode_idx + 1}.pt")

        print('Train Episode {}: Average Loss: {:.6f}'.format(episode_idx + 1, train_loss/len_episode))

    return losses

# Reward function for walking along the x axis
def reward_fn(st, st_1, at):
    return st_1[:, 0] - st[:, 0]

def eval_dynamics(dynamics, env, horizon, num_samples, device, pred_delta):
    dynamics.eval()
    dynamics = dynamics.to(device)

    st = env.reset()
    at = np.array([env.action_space.sample() for _ in range(num_samples)])
    rt = torch.zeros(num_samples)

    for i in range(horizon):
        if i != 0:
            at = np.array([env.action_space.sample() for _ in range(num_samples)])

        X = np.concatenate((st, at), axis=1)
        X = torch.from_numpy(X).to(device).float()

        st_1 = dynamics(X)
        rt += reward_fn(st, st_1, at)
        st = st_1