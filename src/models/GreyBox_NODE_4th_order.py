import torch.nn

from src.models.models_base import NodeRegressionModelLatEncoder
import torch.nn as nn
from torchdiffeq import odeint
from torch.optim import Adam, AdamW, RMSprop
import numpy as np
torch.set_default_dtype(torch.float64)

class GreyboxODE1_4th_order(torch.nn.Module):
    """
    Greybox 1 ODE model used in the GreyBoxModel class.
    """
    def __init__(self,
                 dim_x,
                 dim_z,
                 hidden_size,
                 R1=50 * 1e-3,  #50 milli ohm
                 R2=10 * 1e-6,  #10 micro ohm
                 #Rp1=1*1e9, #1 giga ohm
                 #Rp2=1*1e9, #1 giga ohm
                 ):
        super(GreyboxODE1_4th_order, self).__init__()

        self.C1 = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
            )

        self.C2 = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1))

        self.L1 = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1))

        self.L2 = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1))



        self.R1 = nn.Parameter(self.inverse_softplus(torch.tensor(R1, dtype=torch.float64)))
        self.R2 = nn.Parameter(self.inverse_softplus(torch.tensor(R2, dtype=torch.float64)))

    @staticmethod
    def inverse_softplus(y):
        return torch.log(torch.exp(y) - 1)

    def get_parameters(self):
        if self.learn_params:
            softplus = nn.Softplus()
            return {
                'R1': softplus(self.R1).item(),
                'R2': softplus(self.R2).item(),
                #'Rp1': softplus(self.Rp1).item(),
                #'Rp2': softplus(self.Rp2).item(),
            }
        else:
            return {
                'R1': self.R1.item(),
                'R2': self.R2.item(),
                #'Rp1': self.Rp1.item(),
                #'Rp2': self.Rp2.item(),
            }

    def print_params(self):
        softplus = nn.Softplus()
        R1 = softplus(self.R1).item()
        R2 = softplus(self.R2).item()
        print(f" R1: {R1}, R2: {R2}")

    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def _batch_linear_interpolation(self, t_query):
        """
        Linear interpolation of the x values
        """

        # Get indices for interpolation
        idx_left = (torch.abs(self.t - t_query)).argmin().item()
        if idx_left + 1 >= self.t.shape[0]:
            return self.x[:, idx_left, :]
        idx_right = idx_left + 1

        # Gather the x values
        x_left = self.x[:, idx_left, :]
        x_right = self.x[:, idx_right, :]

        # Gather the t values
        t_left = self.t[idx_left]
        t_right = self.t[idx_right]

        # Calculate interpolated x values
        interpolated_x = x_left + (x_right - x_left) * ((t_query - t_left) / (t_right - t_left))

        return interpolated_x.squeeze(1)

    def forward(self, t, z, x=None):
        if x is None:
            x = self.get_current_x(t)

        v_IN = x[..., 0]
        i_OUT = x[..., 1]
        v_OUT = z[..., 0]
        i_IN = z[..., 1]
        v_INT = z[..., 2]
        i_INT = z[..., 3]
        softplus = nn.Softplus()

        R1 = softplus(self.R1)
        R2 = softplus(self.R2)

        C1_eff = softplus(self.C1(torch.cat([x, z], dim=-1)).squeeze())
        C2_eff = softplus(self.C2(torch.cat([x, z], dim=-1)).squeeze())

        L1_eff = softplus(self.L1(torch.cat([x, z], dim=-1)).squeeze())
        L2_eff = softplus(self.L2(torch.cat([x, z], dim=-1)).squeeze())

        # System equations
        d_v_INT_dt = (i_IN - i_INT) / C1_eff  #- (v_INT / (Rp1 * C1_eff))
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)  #- (v_OUT / (Rp2 * C2_eff))
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)

        return torch.stack([d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt], dim=-1)


class GreyboxODE2_4th_order(torch.nn.Module):
    """ Greybox 2 ODE model used in the GreyBoxModel class."""
    def __init__(self,
                 C2=100 * 1e-6,  # 100 micro farad
                 R1=50 * 1e-3,  # 50 milli ohm
                 # Rp1=1*1e9, #1 giga ohm
                 # Rp2=1*1e9, #1 giga ohm
                 L1=1 * 1e-6,  # 1 micro henry
                 dim_x=2,
                 dim_z=4,
                 hidden_size=100,
                 ):
        super(GreyboxODE2_4th_order, self).__init__()

        self.C2 = nn.Parameter(self.inverse_softplus(torch.tensor(C2, dtype=torch.float64)))
        self.R1 = nn.Parameter(self.inverse_softplus(torch.tensor(R1, dtype=torch.float64)))
        # self.Rp1 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp1, dtype=torch.float64)))
        # self.Rp2 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp2, dtype=torch.float64)))
        self.L1 = nn.Parameter(self.inverse_softplus(torch.tensor(L1, dtype=torch.float64)))

        self.NN_Unknown = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 2))

    @staticmethod
    def inverse_softplus(y):
        return torch.log(torch.exp(y) - 1)

    def get_parameters(self):
        if self.learn_params:
            softplus = nn.Softplus()
            return {
                'C2': softplus(self.C2).item(),
                'R1': softplus(self.R1).item(),

                # 'Rp1': softplus(self.Rp1).item(),
                # 'Rp2': softplus(self.Rp2).item(),
                'L1': softplus(self.L1).item(),
            }

        else:
            return {
                'C2': self.C2.item(),
                'R1': self.R1.item(),
                # 'Rp1': self.Rp1.item(),
                # 'Rp2': self.Rp2.item(),
                'L1': self.L1.item()}

    def print_params(self):

        softplus = nn.Softplus()
        C2 = softplus(self.C2).item()
        R1 = softplus(self.R1).item()
        L1 = softplus(self.L1).item()

        print(f",C2={C2},R1={R1},L1={L1}")

    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def _batch_linear_interpolation(self, t_query):
        """
        Linear interpolation of the x values
        """

        # Get indices for interpolation
        idx_left = (torch.abs(self.t - t_query)).argmin().item()
        if idx_left + 1 >= self.t.shape[0]:
            return self.x[:, idx_left, :]
        idx_right = idx_left + 1

        # Gather the x values
        x_left = self.x[:, idx_left, :]
        x_right = self.x[:, idx_right, :]

        # Gather the t values
        t_left = self.t[idx_left]
        t_right = self.t[idx_right]

        # Calculate interpolated x values
        interpolated_x = x_left + (x_right - x_left) * ((t_query - t_left) / (t_right - t_left))

        return interpolated_x.squeeze(1)

    def forward(self, t, z, x=None):
        if x is None:
            x = self.get_current_x(t)

        v_IN = x[..., 0]
        i_OUT = x[..., 1]
        v_OUT = z[..., 0]
        i_IN = z[..., 1]
        v_INT = z[..., 2]
        i_INT = z[..., 3]
        softplus = nn.Softplus()
        C2 = softplus(self.C2)
        R1 = softplus(self.R1)
        L1 = softplus(self.L1)

        C2_eff = C2
        L1_eff = L1

        # System equations
        # System equations

        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)  # - (v_OUT / (Rp2 * C2_eff))

        z_known = torch.stack([d_v_OUT_dt, d_i_IN_dt], dim=-1)
        z_unknown = self.NN_Unknown(torch.cat([x, z], dim=-1))*2500

        return torch.cat([z_known, z_unknown], dim=-1)


class GreyboxODE3_4th_order(torch.nn.Module):
    """Greybox 3 ODE model used in the GreyBoxModel class."""
    def __init__(self,
                 C1=500 * 1e-6,  #500 micro farad
                 C2=100 * 1e-6,  #100 micro farad
                 R1=50 * 1e-3,  #50 milli ohm
                 R2=10 * 1e-6,  #10 micro ohm
                 #Rp1=1*1e9, #1 giga ohm
                 #Rp2=1*1e9, #1 giga ohm
                 L1=1 * 1e-6,  #1 micro henry
                 L2=100 * 1e-9,  #100 nano henry
                 dim_x=2,
                 dim_z=4,
                 hidden_size=100,
                 ):
        super(GreyboxODE3_4th_order, self).__init__()

        self.C1 = nn.Parameter(self.inverse_softplus(torch.tensor(C1, dtype=torch.float64)))
        self.C2 = nn.Parameter(self.inverse_softplus(torch.tensor(C2, dtype=torch.float64)))
        self.R1 = nn.Parameter(self.inverse_softplus(torch.tensor(R1, dtype=torch.float64)))
        self.R2 = nn.Parameter(self.inverse_softplus(torch.tensor(R2, dtype=torch.float64)))
        #self.Rp1 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp1, dtype=torch.float64)))
        #self.Rp2 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp2, dtype=torch.float64)))
        self.L1 = nn.Parameter(self.inverse_softplus(torch.tensor(L1, dtype=torch.float64)))
        self.L2 = nn.Parameter(self.inverse_softplus(torch.tensor(L2, dtype=torch.float64)))

        self.NN_Nonlinearitys = torch.nn.Sequential(
            torch.nn.Linear(dim_x + dim_z, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, dim_z))

    @staticmethod
    def inverse_softplus(y):
        return torch.log(torch.exp(y) - 1)

    def get_parameters(self):
        if self.learn_params:
            softplus = nn.Softplus()
            return {'C1': softplus(self.C1).item(),
                    'C2': softplus(self.C2).item(),
                    'R1': softplus(self.R1).item(),
                    'R2': softplus(self.R2).item(),
                    #'Rp1': softplus(self.Rp1).item(),
                    #'Rp2': softplus(self.Rp2).item(),
                    'L1': softplus(self.L1).item(),
                    'L2': softplus(self.L2).item()}

        else:
            return {'C1': self.C1.item(),
                    'C2': self.C2.item(),
                    'R1': self.R1.item(),
                    'R2': self.R2.item(),
                    #'Rp1': self.Rp1.item(),
                    #'Rp2': self.Rp2.item(),
                    'L1': self.L1.item(),
                    'L2': self.L2.item()}

    def print_params(self):
        if self.learn_params:
            softplus = nn.Softplus()
            C1 = softplus(self.C1).item()
            C2 = softplus(self.C2).item()
            R1 = softplus(self.R1).item()
            R2 = softplus(self.R2).item()
            L1 = softplus(self.L1).item()
            L2 = softplus(self.L2).item()

            print(f"C1={C1},C2={C2},R1={R1},R2={R2},L1={L1},L2={L2}")
        else:
            print(f"C1={self.C1},\nC2={self.C2},R1={self.R1},R2={self.R2},L1={self.L1},L2={self.L2}")

    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def _batch_linear_interpolation(self, t_query):
        """
        Linear interpolation of the x values
        """

        # Get indices for interpolation
        idx_left = (torch.abs(self.t - t_query)).argmin().item()
        if idx_left + 1 >= self.t.shape[0]:
            return self.x[:, idx_left, :]
        idx_right = idx_left + 1

        # Gather the x values
        x_left = self.x[:, idx_left, :]
        x_right = self.x[:, idx_right, :]

        # Gather the t values
        t_left = self.t[idx_left]
        t_right = self.t[idx_right]

        # Calculate interpolated x values
        interpolated_x = x_left + (x_right - x_left) * ((t_query - t_left) / (t_right - t_left))

        return interpolated_x.squeeze(1)

    def forward(self, t, z, x=None):
        if x is None:
            x = self.get_current_x(t)

        v_IN = x[..., 0]
        i_OUT = x[..., 1]
        v_OUT = z[..., 0]
        i_IN = z[..., 1]
        v_INT = z[..., 2]
        i_INT = z[..., 3]
        softplus = nn.Softplus()
        C1 = softplus(self.C1)
        C2 = softplus(self.C2)
        R1 = softplus(self.R1)
        R2 = softplus(self.R2)
        L1 = softplus(self.L1)
        L2 = softplus(self.L2)

        C1_eff = C1
        C2_eff = C2
        L1_eff = L1
        L2_eff = L2

        # System equations
        d_v_INT_dt = (i_IN - i_INT) / C1_eff  #- (v_INT / (Rp1 * C1_eff))
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)  #- (v_OUT / (Rp2 * C2_eff))
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)

        z = torch.stack([d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt], dim=-1)

        z_nonlinear = self.NN_Nonlinearitys(torch.cat([x, z], dim=-1)).squeeze()*2500
        return z + z_nonlinear


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(
            input_size=config['data']['dim_x'] + config['data']['dim_z'],
            hidden_size=config['model']['n_hidden_encode'],
            num_layers=config['model']['n_layers_encode'],
            batch_first=True
        )
        self.fc = nn.Linear(config['model']['n_hidden_encode'], config['data']['dim_z'])

    def forward(self, x):
        x = torch.flip(x, [1])  # we want to go backwards in time
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


class GreyBoxODEModel(NodeRegressionModelLatEncoder):
    """Class that defines the GreyBoxODEModel model and executes the forward pass. The type of GreyBoxODEModel is
    defined in the config in the 'Greybox_type' key. The model is trained with the Adam optimizer. The model can
    encode the latent space if 'encode_latent_space' is set to True in the config. The model can be of type
    GreyboxODE1, GreyboxODE2, or GreyboxODE3.
    """

    def __init__(self, config=None):
        super(GreyBoxODEModel, self).__init__()
        assert config['training']['optimizer'] in ['Adam', 'AdamW', 'RMSprop']
        assert config['model']['Greybox_type'] is not None

        self.encode_latent_space = config['model']['encode_latent_space']
        if self.encode_latent_space:
            self.len_encode = config['model']['len_encode']
            assert config['model']['n_hidden_encode'] is not None
            assert config['model']['n_hidden_encode'] > 0
            assert config['model']['n_layers_encode'] is not None
            assert config['model']['n_layers_encode'] > 0
            self.encoder = Encoder(config)

        if config['model']['Greybox_type'] == 'GreyboxODE1':
            assert config['model']['hidden_size_NN'] > 0
            assert config['model']['init_params']['R1'] > 0
            assert config['model']['init_params']['R2'] > 0

            self.ODE = GreyboxODE1_4th_order(dim_x=config['data']['dim_x'],
                                             dim_z=config['data']['dim_y']+config['data']['dim_z'],
                                             R1=config['model']['init_params']['R1'],
                                             R2=config['model']['init_params']['R2'],
                                             hidden_size=config['model']['hidden_size_NN'])

        elif config['model']['Greybox_type'] == 'GreyboxODE2':
            assert config['model']['hidden_size_NN'] > 0
            assert config['model']['init_params']['L1'] > 0
            assert config['model']['init_params']['R1'] > 0
            assert config['model']['init_params']['C2'] > 0

            self.ODE = GreyboxODE2_4th_order(dim_x=config['data']['dim_x'],
                                             dim_z=config['data']['dim_y']+config['data']['dim_z'],
                                             L1=config['model']['init_params']['L1'],
                                             R1=config['model']['init_params']['R1'],
                                             C2=config['model']['init_params']['C2'],
                                             hidden_size=config['model']['hidden_size_NN'])

        elif config['model']['Greybox_type'] == 'GreyboxODE3':
            assert config['model']['hidden_size_NN'] > 0
            assert config['model']['init_params']['L1'] > 0
            assert config['model']['init_params']['R1'] > 0
            assert config['model']['init_params']['C1'] > 0
            assert config['model']['init_params']['C2'] > 0
            assert config['model']['init_params']['L2'] > 0
            assert config['model']['init_params']['R2'] > 0
            self.ODE = GreyboxODE3_4th_order(dim_x=config['data']['dim_x'],
                                             dim_z=config['data']['dim_y']+config['data']['dim_z'],
                                             R1=config['model']['init_params']['R1'],
                                             R2=config['model']['init_params']['R2'],
                                             C1=config['model']['init_params']['C1'],
                                             C2=config['model']['init_params']['C2'],
                                             L1=config['model']['init_params']['L1'],
                                             L2=config['model']['init_params']['L2'],
                                             hidden_size=config['model']['hidden_size_NN'])

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
        z = odeint(self.ODE,
                   y_0,
                   t_rel[0, :],
                   method='rk4',
                   options={'step_size': dt_min / 5}
                   ).permute(1, 0, 2)
        y=z[...,0:2]
        if not return_z:
            return y
        else:
            return y, z
