import torch.nn

from src.models.models_base import NodeRegressionModel
import torch.nn as nn
from torchdiffeq import odeint
from torch.optim import Adam,AdamW,RMSprop
import numpy as np
class ControlledODE(nn.Module):
    """Neural ODE with control input"""
    def __init__(self, dim_x, dim_z, hidden_size,n_layers_hidden=1):
        super().__init__()
        if n_layers_hidden>1:
            self.ODE = torch.nn.Sequential(
                torch.nn.Linear(dim_x + dim_z, hidden_size),
                torch.nn.Tanh(),
                *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size),
                                      torch.nn.Tanh()) for _ in range(n_layers_hidden-1)],
                torch.nn.Linear(hidden_size, dim_z))
        else:

            self.ODE = torch.nn.Sequential(
                torch.nn.Linear(dim_x + dim_z, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, dim_z))

    def set_x(self, t, x):
        self.x = x
        self.t = t.squeeze(0)

    def get_current_x(self, t):
        #     #check which index is closest to t
        if len(self.t.shape) > 1:
            differences = torch.abs(self.t - t.unsqueeze(1))
            indices = differences.argmin(dim=1)
            return torch.stack([self.x[b, indices[b], :] for b in range(self.x.size(0))])
        else:
            index = (torch.abs(self.t - t)).argmin().item()
            return self.x[:, index, :]


    def forward(self,t,z,x=None):
        if x is None:
            x=self.get_current_x(t)
        z_aug=torch.cat([x,z],dim=-1)
        return self.ODE(z_aug)


class BlackBoxODEModel(NodeRegressionModel):
    """Black box ODE model."""

    def __init__(self, config=None):
        super(BlackBoxODEModel, self).__init__()
        assert config['training']['optimizer'] in ['Adam', 'AdamW', 'RMSprop']
        assert config['model']['hidden_size_ODE'] > 0


        self.ODE = ControlledODE(dim_x=config['data']['dim_x'], dim_z=config['data']['dim_y'],
                                 hidden_size=config['model']['hidden_size_ODE'],
                                 n_layers_hidden=config['model']['layers_hidden_ODE'])
        self.prepare_training(config)

    def _initialize_optimizer(self):
        """Initialize the Adam optimizer with the learning rate from the config"""
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        if self.config.get("training", {}).get("optimizer", 'Adam') == 'Adam':
            return Adam(self.trainable_parameters(), lr=lr)
        elif self.config.get("training", {}).get("optimizer", 'Adam') == 'AdamW':
            return AdamW(self.trainable_parameters(), lr=lr)
        elif self.config.get("training", {}).get("optimizer", 'Adam') == 'RMSprop':
            return RMSprop(self.trainable_parameters(), lr=lr, momentum=0.1)

    def forward(self, t, x, y_0, return_z=False):
        dt_min = t.diff(dim=1).min()
        dt_max = t.diff(dim=1).max()
        # get relative times in batch
        t_rel = (t[:, :] - t[:, 0].unsqueeze(1))
        # check if data is regulary sampled
        if (dt_max / dt_min) > 1.08:
            raise ValueError("Data is not regularly sampled")
        self.ODE.set_x(t_rel[0, :], x)
        y = odeint(self.ODE,
                   y_0,
                   t_rel[0, :],
                   method='rk4',
                   options={'step_size': dt_min / 5}
                   ).permute(1, 0, 2)

        if not return_z:
            return y
        else:
            return y, y


