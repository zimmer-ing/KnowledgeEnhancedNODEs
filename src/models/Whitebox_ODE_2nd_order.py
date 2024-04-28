import torch.nn

from src.models.models_base import NodeRegressionModel
import torch.nn as nn
#from torchdiffeq import odeint
from torchdiffeq import odeint as odeint
import numpy as np
from torch.optim import Adam,AdamW,RMSprop
from tqdm import tqdm
import Constants as const

import itertools


class Whitebox_ODE_2nd_order(nn.Module):
    """
    Known ODE model for the white-box experiment.
    """
    def __init__(self,learn_params=False,C=100*1e-3,L=1*1e-3,R=10*1e-2,Rp=1*1e9,C_nonlinear=False,slope_c=0.1):
        super().__init__()

        self.c_nonlinear=C_nonlinear
        self.learn_params=learn_params

        if not learn_params:
            self.scale = nn.Parameter(torch.tensor(1.0))#dummy parameter
            self.C=C
            self.L=L
            self.R=R
            self.Rp=Rp
            self.slope_c=slope_c
        else:
            self.scale = torch.tensor(1.0)
            self.C_param = nn.Parameter(torch.tensor(np.log(C), dtype=torch.float32))
            self.L_param = nn.Parameter(torch.tensor(np.log(L), dtype=torch.float32))
            self.R_param = nn.Parameter(torch.tensor(np.log(R), dtype=torch.float32))
            self.Rp_param = nn.Parameter(torch.tensor(np.log(Rp), dtype=torch.float32))
            if self.c_nonlinear:
                self.slope_c = nn.Parameter(torch.tensor(np.log(slope_c), dtype=torch.float32))
            else:
                self.slope_c=torch.tensor(0)


    def set_x(self,t,x):
        """
        Set the x values and the corresponding times
        """
        self.x=x
        self.t=t.squeeze(0)


    def get_current_x(self, t):
        """
        Get the current x values at
        :param t: current time
        """
        #     #check which index is closest to t
        if len(self.t.shape) > 1:
            differences = torch.abs(self.t - t.unsqueeze(1))
            indices = differences.argmin(dim=1)
            return torch.stack([self.x[b, indices[b], :] for b in range(self.x.size(0))])
        else:
            index = (torch.abs(self.t - t)).argmin().item()
            return self.x[:, index, :]


    def nonlinear_capacitor(self, voltage, C,slope=0.1):
        """
        Nonlinear capacitor model
        """
        c_act = torch.where(voltage >= 0,
                            C * torch.exp(-slope * voltage),
                            C * torch.exp(slope * voltage))
        return c_act
    def print_params(self):
        if not self.learn_params:
            C=self.C
            L=self.L
            R=self.R
            Rp=self.Rp
        else:
            C = torch.exp(self.C_param)
            L = torch.exp(self.L_param)
            R = torch.exp(self.R_param)
            Rp = torch.exp(self.Rp_param)
            if self.c_nonlinear:
                slope_c=torch.exp(self.slope_c)
                print(f"C={C},L={L},R={R},Rp={Rp},slope_c={slope_c}")
            else:
                print(f"C={C},L={L},R={R},Rp={Rp}")

    def get_parameters(self):
        if not self.learn_params:
            C=self.C
            L=self.L
            R=self.R
            Rp=self.Rp
        else:
            C = torch.exp(self.C_param)
            L = torch.exp(self.L_param)
            R = torch.exp(self.R_param)
            Rp = torch.exp(self.Rp_param)
            if self.c_nonlinear:
                slope_c=torch.exp(self.slope_c)
                return {'C':C,'L':L,'R':R,'Rp':Rp,'slope_c':slope_c}
            else:
                return {'C': C, 'L': L, 'R': R, 'Rp': Rp, 'slope_c': np.nan}


    def forward(self,t,z,x=None):
        """
        Forward pass of the model
        """
        if x is None:
            x=self.get_current_x(t)
        #given (controlled) values
        v_in=x[...,0]
        i_out=x[...,1]
        v_out=z[...,0]
        i_in=z[...,1]

        if not self.learn_params:
            C=self.C
            L=self.L
            R=self.R
            Rp=self.Rp
            slope_c=self.slope_c
        else:
            C = torch.exp(self.C_param)
            L = torch.exp(self.L_param)
            R = torch.exp(self.R_param)
            Rp = torch.exp(self.Rp_param)
            slope_c = torch.exp(self.slope_c)

        if self.c_nonlinear:
            c_act=self.nonlinear_capacitor(v_out,C,slope_c)
        else:
            c_act=C

        d_v_out_dt= ((i_in/c_act)
                    -(i_out/c_act)
                    -(v_out/(Rp*c_act)))
        d_i_in_dt= (v_in/L
                    -(R/L)*i_in
                    -v_out/L)

        z_dot=torch.stack([d_v_out_dt, d_i_in_dt], dim=-1)

        return z_dot*self.scale




class Whitebox_ODE_2nd_order_Model(NodeRegressionModel):
    """Known ODE model for the white-box experiment."""

    def __init__(self, config=None):
        super(Whitebox_ODE_2nd_order_Model, self).__init__()
        assert config['model']['learn_params'] in [True,False]
        assert config['training']['optimizer'] in ['Adam','AdamW','RMSprop']
        assert config['model']['init_params'] is not None
        c=config['model']['init_params']['C']
        l=config['model']['init_params']['L']
        r=config['model']['init_params']['R']
        rp=config['model']['init_params']['Rp']
        nonlinear_C = config['model']['init_params']['Nonlinear_C']
        if nonlinear_C:
            assert config['model']['init_params']['slope_c'] is not None
            slope_c=config['model']['init_params']['slope_c']
        else:
            slope_c=0



        self.ODE=Whitebox_ODE_2nd_order(learn_params=config['model']['learn_params'], C=c, L=l, R=r, Rp=rp, C_nonlinear=nonlinear_C, slope_c=slope_c)
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

    def forward(self, t,x,y_0,return_z=False):
        """
        Forward pass of the model
        """
        dt_min=t.diff(dim=1).min()
        dt_max=t.diff(dim=1).max()
        # get relative times in batch
        t_rel = (t[:, :] - t[:, 0].unsqueeze(1))
        #check if data is regulary sampled
        if (dt_max/dt_min)>1.08:
            raise ValueError("Data is not regularly sampled")
        self.ODE.set_x(t_rel[0,:], x)
        y=odeint(self.ODE,
                 y_0,
                 t_rel[0,:],
                 method='rk4',
                 options={'step_size': dt_min/5}
                 ).permute(1,0,2)

        if not return_z:
            return y
        else:
            return y,y







