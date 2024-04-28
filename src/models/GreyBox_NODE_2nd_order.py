import torch.nn

from src.models.models_base import NodeRegressionModel
import torch.nn as nn
from torchdiffeq import odeint
from torch.optim import Adam,AdamW,RMSprop
import numpy as np
class GreyboxODE1(nn.Module):
    """
    Greybox ODE model used in the GreyboxModel class.
    """
    def __init__(self, dim_x, dim_z, hidden_size,L=1*1e-3,R=10*1e-2,Rp=1*1e9):
        super().__init__()
        self.C = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1))
        self.L_param = nn.Parameter(torch.tensor(np.log(L), dtype=torch.float32))
        self.R_param = nn.Parameter(torch.tensor(np.log(R), dtype=torch.float32))
        self.Rp_param = nn.Parameter(torch.tensor(np.log(Rp), dtype=torch.float32))

    def print_params(self):
        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)
        Rp = torch.exp(self.Rp_param)
        print(f" L: {L}, R: {R}, Rp: {Rp}")
    def set_x(self, t, x):
        self.x = x
        self.t = t.squeeze(0)

    def get_current_x(self, t):
        """ Get the current x values at
        :param t: current time"""
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
            # given (controlled= values
        v_in = x[..., 0]
        i_out = x[..., 1]
        v_out = z[..., 0]
        i_in = z[..., 1]

        C = torch.exp(self.C(torch.cat([x, z], dim=-1))).squeeze()
        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)
        Rp = torch.exp(self.Rp_param)

        #nonlinearity is directly captured by the the NN that models C
        c_act = C

        d_v_out_dt = (i_in \
                      - i_out \
                      - v_out / (Rp)) / c_act
        d_i_in_dt = (v_in \
                     - R * i_in \
                     - v_out) / L

        z_dot = torch.stack([d_v_out_dt, d_i_in_dt], dim=-1)
        return z_dot

class GreyboxODE2(nn.Module):
    """
    Greybox ODE model used in the GreyboxModel class.

    """
    def __init__(self, dim_x, dim_z, hidden_size,L=1*1e-3,R=10*1e-2):
        super().__init__()


        self.term_NN = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1))

        self.L_param = nn.Parameter(torch.tensor(np.log(L), dtype=torch.float32))
        self.R_param = nn.Parameter(torch.tensor(np.log(R), dtype=torch.float32))

    def print_params(self):
        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)
        print(f" L: {L}, R: {R}")
    def set_x(self, t, x):
        self.x = x
        self.t = t.squeeze(0)

    def get_current_x(self, t):
        """ Get the current x values at
        :param t: current time"""
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
            # given (controlled= values
        v_in = x[..., 0]
        i_out = x[..., 1]
        v_out = z[..., 0]
        i_in = z[..., 1]


        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)


        d_v_out_dt = self.term_NN(torch.cat([x, z], dim=-1)).squeeze()
        d_i_in_dt = (v_in \
                     - R * i_in \
                     - v_out) / L

        z_dot = torch.stack([d_v_out_dt, d_i_in_dt], dim=-1)
        return z_dot


class GreyboxODE3(nn.Module):

    """
    Greybox ODE model used in the GreyboxModel class.
    """
    def __init__(self, dim_x, dim_z, hidden_size,L=1*1e-3,R=10*1e-2,C=100*1e-3):
        super().__init__()


        self.term_NN = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, dim_z))

        self.L_param = nn.Parameter(torch.tensor(np.log(L)))
        self.R_param = nn.Parameter(torch.tensor(np.log(R)))
        self.C_param = nn.Parameter(torch.tensor(np.log(C)))

    def print_params(self):
        C = torch.exp(self.C_param)
        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)
        print(f" L: {L}, R: {R}", f"C: {C}")
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
            # given (controlled= values
        v_in = x[..., 0]
        i_out = x[..., 1]
        v_out = z[..., 0]
        i_in = z[..., 1]


        L = torch.exp(self.L_param)
        R = torch.exp(self.R_param)
        C = torch.exp(self.C_param)

        d_v_out_dt = ((i_in / C)
                      - (i_out / C))

        d_i_in_dt = (v_in / L
                     - (R / L) * i_in
                     - v_out / L)

        z_dot = torch.stack([d_v_out_dt, d_i_in_dt], dim=-1)

        #NN to model nonlinearitys
        z_dot=z_dot+self.term_NN(torch.cat([x, z], dim=-1))

        return z_dot

class GreyBoxODEModel(NodeRegressionModel):
    """Partially known ODE model."""

    def __init__(self, config=None):
        super(GreyBoxODEModel, self).__init__()
        assert config['training']['optimizer'] in ['Adam', 'AdamW', 'RMSprop']
        assert config['model']['Greybox_type'] is not None

        if config['model']['Greybox_type'] == 'GreyboxODE1':
            assert config['model']['hidden_size_C_NN'] > 0
            assert config['model']['init_params']['L'] > 0
            assert config['model']['init_params']['R'] > 0
            assert config['model']['init_params']['Rp'] > 0
            self.ODE = GreyboxODE1(dim_x=config['data']['dim_x'],
                                   dim_z=config['data']['dim_y'],
                                   R=config['model']['init_params']['R'],
                                   L=config['model']['init_params']['L'],
                                   Rp=config['model']['init_params']['Rp'],
                                  hidden_size=config['model']['hidden_size_C_NN'])

        elif config['model']['Greybox_type'] == 'GreyboxODE2':
            assert config['model']['hidden_size_term_NN'] > 0
            assert config['model']['init_params']['L'] > 0
            assert config['model']['init_params']['R'] > 0
            self.ODE = GreyboxODE2(dim_x=config['data']['dim_x'],
                                   dim_z=config['data']['dim_y'],
                                    R=config['model']['init_params']['R'],
                                    L=config['model']['init_params']['L'],

                                  hidden_size=config['model']['hidden_size_term_NN'])


        elif config['model']['Greybox_type'] == 'GreyboxODE3':
            assert config['model']['hidden_size_term_NN'] > 0
            assert config['model']['init_params']['L'] > 0
            assert config['model']['init_params']['R'] > 0
            assert config['model']['init_params']['C'] > 0
            self.ODE = GreyboxODE3(dim_x=config['data']['dim_x'],
                                   dim_z=config['data']['dim_y'],
                                    R=config['model']['init_params']['R'],
                                    L=config['model']['init_params']['L'],
                                    C=config['model']['init_params']['C'],
                                  hidden_size=config['model']['hidden_size_term_NN'])

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


