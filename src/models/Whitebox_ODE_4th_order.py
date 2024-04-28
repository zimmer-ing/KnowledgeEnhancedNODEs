import torch
import torch.nn as nn
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from src.models.models_base import NodeRegressionModelLatEncoder,ensure_sequential_dataloader,concatenate_batches
import Constants as const
from torch.optim import Adam,AdamW,RMSprop,SGD
from tqdm import tqdm
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class Whitebox_ODE_4th_order(torch.nn.Module):
    """
    Known ODE model of 4th order for the white-box experiment.
    """
    def __init__(self, learn_params=False,
                 C1=500*1e-6, #500 micro farad
                 C2=100*1e-6, #100 micro farad
                 R1=50*1e-3, #50 milli ohm
                 R2=10*1e-6, #10 micro ohm
                 #Rp1=1*1e9, #1 giga ohm
                 #Rp2=1*1e9, #1 giga ohm
                 L1=1*1e-6, #1 micro henry
                 L2=100*1e-9,#100 nano henry
                Nonlinear_C1=False,
                Nonlinear_C2=True,
                Nonlinear_L1=True,
                Nonlinear_L2=False,
                 Slope_C1=0.1,
                 Slope_C2=0.1,
                 Slope_L1=3,
                 Slope_L2=0.1):
        super(Whitebox_ODE_4th_order, self).__init__()
        self.nonlinear_C1 = Nonlinear_C1
        self.nonlinear_C2 = Nonlinear_C2
        self.nonlinear_L1 = Nonlinear_L1
        self.nonlinear_L2 = Nonlinear_L2
        self.learn_params = learn_params





        if learn_params:
            self.C1 = nn.Parameter(self.inverse_softplus(torch.tensor(C1, dtype=torch.float64)))
            self.C2 = nn.Parameter(self.inverse_softplus(torch.tensor(C2, dtype=torch.float64)))
            self.R1 = nn.Parameter(self.inverse_softplus(torch.tensor(R1, dtype=torch.float64)))
            self.R2 = nn.Parameter(self.inverse_softplus(torch.tensor(R2, dtype=torch.float64)))
            #self.Rp1 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp1, dtype=torch.float64)))
            #self.Rp2 = nn.Parameter(self.inverse_softplus(torch.tensor(Rp2, dtype=torch.float64)))
            self.L1 = nn.Parameter(self.inverse_softplus(torch.tensor(L1, dtype=torch.float64)))
            self.L2 = nn.Parameter(self.inverse_softplus(torch.tensor(L2, dtype=torch.float64)))
            self.Slope_C1 = nn.Parameter(self.inverse_softplus(torch.tensor(Slope_C1, dtype=torch.float64)))
            self.Slope_C2 = nn.Parameter(self.inverse_softplus(torch.tensor(Slope_C2, dtype=torch.float64)))
            self.Slope_L1 = nn.Parameter(self.inverse_softplus(torch.tensor(Slope_L1, dtype=torch.float64)))
            self.Slope_L2 = nn.Parameter(self.inverse_softplus(torch.tensor(Slope_L2, dtype=torch.float64)))

        else:
            self.C1 = torch.tensor(C1)
            self.C2 = torch.tensor(C2)
            self.R1 = torch.tensor(R1)
            self.R2 = torch.tensor(R2)
            #self.Rp1 = torch.tensor(Rp1)
            #self.Rp2 = torch.tensor(Rp2)
            self.L1 = torch.tensor(L1)
            self.L2 = torch.tensor(L2)
            self.Slope_C1 = torch.tensor(Slope_C1)
            self.Slope_C2 = torch.tensor(Slope_C2)
            self.Slope_L1 = torch.tensor(Slope_L1)
            self.Slope_L2 = torch.tensor(Slope_L2)
            self.scale = nn.Parameter(torch.tensor(1.0))  # dummy parameter to satisfy torch
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
                    'L2': softplus(self.L2).item(),
                    'Slope_C1': softplus(self.Slope_C1).item(),
                    'Slope_C2': softplus(self.Slope_C2).item(),
                    'Slope_L1': softplus(self.Slope_L1).item(),
                    'Slope_L2': softplus(self.Slope_L2).item()}
        else:
            return {'C1': self.C1.item(),
                    'C2': self.C2.item(),
                    'R1': self.R1.item(),
                    'R2': self.R2.item(),
                    #'Rp1': self.Rp1.item(),
                    #'Rp2': self.Rp2.item(),
                    'L1': self.L1.item(),
                    'L2': self.L2.item(),
                    'Slope_C1': self.Slope_C1.item(),
                    'Slope_C2': self.Slope_C2.item(),
                    'Slope_L1': self.Slope_L1.item(),
                    'Slope_L2': self.Slope_L2.item()}

    def print_params(self):
        if self.learn_params:
            softplus = nn.Softplus()
            C1 = softplus(self.C1).item()
            C2 = softplus(self.C2).item()
            R1 = softplus(self.R1).item()
            R2 = softplus(self.R2).item()
            L1 = softplus(self.L1).item()
            L2 = softplus(self.L2).item()
            Slope_C1 = softplus(self.Slope_C1).item()
            Slope_C2 = softplus(self.Slope_C2).item()
            Slope_L1 = softplus(self.Slope_L1).item()
            Slope_L2 = softplus(self.Slope_L2).item()
            print(f"C1={C1},C2={C2},R1={R1},R2={R2},L1={L1},L2={L2},Slope_C1={Slope_C1},Slope_C2={Slope_C2},Slope_L1={Slope_L1},Slope_L2={Slope_L2}")
        else:
            print(f"C1={self.C1},\nC2={self.C2},R1={self.R1},R2={self.R2},L1={self.L1},L2={self.L2},Slope_C1={self.Slope_C1},Slope_C2={self.Slope_C2},Slope_L1={self.Slope_L1},Slope_L2={self.Slope_L2}")

    def nonlinear_capacitor(self, voltage, C, Slope=0.1):

        c_act = torch.where(voltage.abs() < 0.1,
                            C,  # When voltage is very small, use C
                            torch.where(voltage >= 0,
                                        C * torch.exp(-Slope * voltage),
                                        C * torch.exp(Slope * voltage)
                                        )
        )
        return c_act

    def nonlinear_inductor(self, current, L, Slope_l=3):
        # Calculate the factor for nonlinear inductance effect
        tanh_term = torch.tanh(Slope_l * current)
        # Use torch.where to handle small current values near zero
        l_act = torch.where(current.abs() < 0.1,
                            torch.tensor([L],device=current.device),  # When current is very small, use L
                            L * tanh_term / (Slope_l * current))  # General case
        return l_act


    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def set_nonlinearities(self, Nonlinear_C1, Nonlinear_C2, Nonlinear_L1, Nonlinear_L2):
        self.nonlinear_C1 = Nonlinear_C1
        self.nonlinear_C2 = Nonlinear_C2
        self.nonlinear_L1 = Nonlinear_L1
        self.nonlinear_L2 = Nonlinear_L2

    def _batch_linear_interpolation(self, t_query):



        # Get indices for interpolation
        idx_left = (torch.abs(self.t - t_query)).argmin().item()
        if idx_left +1 >= self.t.shape[0]:
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

    def forward(self, t, z,x=None):
        if x is None:
            x=self.get_current_x(t)

        v_IN = x[...,0]
        i_OUT = x[...,1]
        v_OUT = z[..., 0]
        i_IN = z[...,1]
        v_INT = z[...,2]
        i_INT = z[...,3]
        softplus = nn.Softplus()
        C1 = softplus(self.C1) if self.learn_params else self.C1
        C2 = softplus(self.C2) if self.learn_params else self.C2
        R1 = softplus(self.R1) if self.learn_params else self.R1
        R2 = softplus(self.R2) if self.learn_params else self.R2
        L1 = softplus(self.L1) if self.learn_params else self.L1
        L2 = softplus(self.L2) if self.learn_params else self.L2
        Slope_C1 = softplus(self.Slope_C1) if self.learn_params else self.Slope_C1
        Slope_C2 = softplus(self.Slope_C2) if self.learn_params else self.Slope_C2
        Slope_L1 = softplus(self.Slope_L1) if self.learn_params else self.Slope_L1
        Slope_L2 = softplus(self.Slope_L2) if self.learn_params else self.Slope_L2

        C1_eff = self.nonlinear_capacitor(v_INT, C1, Slope_C1) if self.nonlinear_C1 else C1
        C2_eff = self.nonlinear_capacitor(v_OUT, C2, Slope_C2) if self.nonlinear_C2 else C2

        L1_eff = self.nonlinear_inductor(i_IN, L1, Slope_L1) if self.nonlinear_L1 else L1
        L2_eff = self.nonlinear_inductor(i_INT, L2, Slope_L2) if self.nonlinear_L2 else L2

        # System equations
        d_v_INT_dt = (i_IN  - i_INT)/ C1_eff #- (v_INT / (Rp1 * C1_eff))
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) -(i_OUT/C2_eff)#- (v_OUT / (Rp2 * C2_eff))
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)


        return torch.stack( [d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt], dim=-1)




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
        x=torch.flip(x,[1]) # we want to go backwards in time
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])






class Whitebox_ODE_4th_order_Model(NodeRegressionModelLatEncoder):
    """Known ODE model for the white-box experiment."""

    def __init__(self, config=None,):
        super(Whitebox_ODE_4th_order_Model, self).__init__()
        assert config['model']['learn_params'] in [True,False]
        assert config['model']['encode_latent_space'] in [True,False]
        if config['model']['encode_latent_space']:
            assert config['model']['len_encode'] is not None
        assert config['training']['optimizer'] in ['Adam','AdamW','RMSprop','SGD']
        assert config['model']['init_params'] is not None
        assert config['model']['init_params']['C1'] is not None
        assert config['model']['init_params']['C2'] is not None
        assert config['model']['init_params']['R1'] is not None
        #assert config['model']['init_params']['Rp1'] is not None
        #assert config['model']['init_params']['Rp2'] is not None
        assert config['model']['init_params']['L1'] is not None
        assert config['model']['init_params']['L2'] is not None
        assert config['model']['init_params']['Nonlinear_C1'] is not None
        assert config['model']['init_params']['Nonlinear_C2'] is not None
        assert config['model']['init_params']['Nonlinear_L1'] is not None
        assert config['model']['init_params']['Nonlinear_L2'] is not None
        assert config['model']['init_params']['Slope_C1'] is not None
        assert config['model']['init_params']['Slope_C2'] is not None
        assert config['model']['init_params']['Slope_L1'] is not None
        assert config['model']['init_params']['Slope_L2'] is not None


        C1=config['model']['init_params']['C1']
        C2=config['model']['init_params']['C2']
        R1=config['model']['init_params']['R1']
        R2=config['model']['init_params']['R2']
        #Rp1=config['model']['init_params']['Rp1']
        #Rp2=config['model']['init_params']['Rp2']
        L1=config['model']['init_params']['L1']
        L2=config['model']['init_params']['L2']
        nonlinear_C1=config['model']['init_params']['Nonlinear_C1']
        nonlinear_C2=config['model']['init_params']['Nonlinear_C2']
        nonlinear_L1=config['model']['init_params']['Nonlinear_L1']
        nonlinear_L2=config['model']['init_params']['Nonlinear_L2']
        Slope_C1=config['model']['init_params']['Slope_C1']
        Slope_C2=config['model']['init_params']['Slope_C2']
        Slope_L1=config['model']['init_params']['Slope_L1']
        Slope_L2=config['model']['init_params']['Slope_L2']

        self.encode_latent_space=config['model']['encode_latent_space']
        if self.encode_latent_space:
            self.len_encode=config['model']['len_encode']
            assert config['model']['n_hidden_encode'] is not None
            assert config['model']['n_hidden_encode'] > 0
            assert config['model']['n_layers_encode'] is not None
            assert config['model']['n_layers_encode'] > 0
            self.encoder = Encoder(config)


        self.ODE=Whitebox_ODE_4th_order(learn_params=config['model']['learn_params'],
                                         C1=C1,
                                         C2=C2,
                                         R1=R1,
                                         R2=R2,
                                         #Rp1=Rp1,
                                         #Rp2=Rp2,
                                         L1=L1,
                                         L2=L2,
                                         Nonlinear_C1=nonlinear_C1,
                                         Nonlinear_C2=nonlinear_C2,
                                         Nonlinear_L1=nonlinear_L1,
                                         Nonlinear_L2=nonlinear_L2,
                                         Slope_C1=Slope_C1,
                                         Slope_C2=Slope_C2,
                                         Slope_L1=Slope_L1,
                                         Slope_L2=Slope_L2
                                         )
        self.prepare_training(config)

    def get_parameters(self):
        return self.ODE.get_parameters()

    def _initialize_optimizer(self):
        """Initialize the Adam optimizer with the learning rate from the config"""
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        if self.config.get("training", {}).get("optimizer", 'Adam') == 'Adam':
            return Adam(self.trainable_parameters(), lr=lr)
        elif self.config.get("training", {}).get("optimizer", 'Adam') == 'AdamW':
            return AdamW(self.trainable_parameters(), lr=lr)
        elif self.config.get("training", {}).get("optimizer", 'Adam') == 'RMSprop':
            return RMSprop(self.trainable_parameters(), lr=lr,momentum=0.1)
        elif self .config.get("training", {}).get("optimizer", 'Adam') == 'SGD':
            return SGD(self.trainable_parameters(), lr=lr)
    def forward(self, t,x,z_0,return_z=False):
        dt_min=t.diff(dim=1).min()
        dt_max=t.diff(dim=1).max()
        # get relative times in batch
        t_rel = (t[:, :] - t[:, 0].unsqueeze(1))
        #check if data is regulary sampled
        if (dt_max/dt_min)>1.08:
            raise ValueError("Data is not regularly sampled")
        #make everything double
        t_rel=t_rel.double()
        x=x.double()
        z_0=z_0.double()
        self.ODE.set_x(t_rel[0,:], x)
        z=odeint(self.ODE,
                 z_0,
                 t_rel[0,:],
                 method='rk4',
                 options={'step_size': dt_min/10}
                 ).permute(1,0,2)
        y=z[...,0:2]
        if not return_z:
            return y
        else:
            return y,z[...,2:]

    def set_nonlinearities(self, Nonlinear_C1, Nonlinear_C2, Nonlinear_L1, Nonlinear_L2):
        self.ODE.set_nonlinearities(Nonlinear_C1, Nonlinear_C2, Nonlinear_L1, Nonlinear_L2)

    def train_step(self, data_loader, return_loss=False):
        # Example training logic using the optimizer
        losses = []
        self.train()
        for batch in tqdm(data_loader, desc="Iteration Training Set", disable=not const.VERBOSE):
            inputs = batch['x']
            time = batch['time']
            targets = batch['y']
            latent = batch['z']
            z_out = targets[:, 0, :]
            self.optimizer.zero_grad()
            if self.encode_latent_space:
                enc_input = torch.cat([inputs[:, 0:self.len_encode, :], targets[:, 0:self.len_encode, :]],
                                      dim=-1).double()
                z_lat = self.encoder(enc_input)
            else:
                z_lat = latent[:, 0, :]
            z_0 = torch.cat([z_out, z_lat], dim=-1)

            predictions = self(time, inputs, z_0)
            targets = targets.to(dtype=predictions.dtype)
            loss = self.calculate_loss(predictions, targets)
            loss.backward()
            if return_loss:
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
        if return_loss:
            # return mean loss
            return sum(losses) / len(losses)