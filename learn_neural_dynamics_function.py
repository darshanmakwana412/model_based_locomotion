'''
Pseudocode:

for each episode in num_episodes
    for each st, at, st_1 in episode

        pred_st_1 = model(st, at)
        loss = loss_fn(pred_st_1, st_1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
'''

import torch
from collections import OrderedDict

class Dynamic(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, depth, activation=None, drop=None, **kwargs) -> None:
        super().__init__(**kwargs)

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

Dynamic(2, 4, 3, 3)