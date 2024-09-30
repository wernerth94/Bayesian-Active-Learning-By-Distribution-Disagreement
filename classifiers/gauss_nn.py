import torch
import torch.nn as nn
from classifiers.seeded_layers import SeededLinear

class GaussNN(nn.Module):
    def __init__(self, model_rng, n_inputs:int, hidden_sizes:list, dropout:float=0.0):
        super().__init__()

        self.features = 1
        # Shared parameters
        self.inpt = SeededLinear(model_rng, n_inputs, hidden_sizes[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.hidden.append(SeededLinear(model_rng, hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
            self.hidden.append(nn.Dropout(p=dropout))
        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.ReLU(),
            SeededLinear(model_rng, hidden_sizes[-1], 1),
        )
        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.ReLU(),
            SeededLinear(model_rng, hidden_sizes[-1], 1),
            nn.Softplus(),  # enforces positivity
        )

    def _encode(self, x:torch.Tensor):
        x = self.inpt(x)
        for layer in self.hidden:
            x = layer(x)
        return x

    def forward(self, x):
        x = self._encode(x)
        mu = self.mean_layer(x)
        sigma = self.std_layer(x)
        return torch.distributions.Normal(mu, sigma)

    def predict(self, x:torch.Tensor, num_samples=64):
        cond_dist = self.forward(x)
        y_hat = cond_dist.sample((num_samples,))
        argmax = cond_dist.log_prob(y_hat).argmax(dim=0)
        y_hat_max = torch.zeros((len(x), self.features)).to(x.device)
        for i in range(len(argmax)):
            y_hat_max[i] = y_hat[argmax[i], i, :]
        return y_hat_max
