import torch
from torchdiffeq import odeint
from torch import nn


class Order_2_RLC_circuit(nn.Module):
    def __init__(self, C=100 * 1e-3, L=1 * 1e-3, R=10 * 1e-2, Rp=1 * 1e9, C_nonlinear=False):
        super().__init__()

        self.C = C
        self.L = L
        self.R = R
        self.Rp = Rp
        self.C_nonlinear = C_nonlinear

    def set_x(self, t, x):
        self.x = x
        self.t = t.squeeze(0)

    def get_current_x(self, t):
        #check which index is closest to t
        if len(self.t.shape) > 1:
            differences = torch.abs(self.t - t.unsqueeze(1))
            indices = differences.argmin(dim=1)
            return torch.stack([self.x[b, indices[b], :] for b in range(self.x.size(0))])
        else:
            index = (torch.abs(self.t - t)).argmin().item()
            return self.x[:, index, :]

    def nonlinear_capacitor(self, voltage, C):
        c_act = torch.where(voltage >= 0,
                            C * torch.exp(-0.1 * voltage),
                            C * torch.exp(0.1 * voltage))
        return c_act

    def forward(self, t, z, x=None):
        if x is None:
            x = self.get_current_x(t)
        #given (controlled= values
        v_in = x[..., 0]
        i_out = x[..., 1]
        v_out = z[..., 0]
        i_in = z[..., 1]

        C = self.C
        L = self.L
        R = self.R
        Rp = self.Rp

        if self.C_nonlinear:
            c_act = self.nonlinear_capacitor(v_out, C)
        else:
            c_act = C

        d_v_out_dt = (i_in \
                      - i_out \
                      - v_out / (Rp)) / c_act
        d_i_in_dt = (v_in \
                     - R * i_in \
                     - v_out) / L

        z_dot = torch.stack([d_v_out_dt, d_i_in_dt], dim=1)
        return z_dot


class Order_4_RLC_Circuit(torch.nn.Module):
    def __init__(self, learn_params=False,
                 C1=500 * 1e-6,  #500 micro farad
                 C2=100 * 1e-6,  #100 micro farad
                 R1=50 * 1e-3,  #50 milli ohm
                 R2=10 * 1e-6,  #10 micro ohm
                 Rp1=1 * 1e9,  #1 giga ohm
                 Rp2=1 * 1e9,  #1 giga ohm
                 L1=1 * 1e-6,  #1 micro henry
                 L2=100 * 1e-9,  #100 nano henry
                 Nonlinear_C1=False,
                 Nonlinear_C2=True,
                 Nonlinear_L1=True,
                 Nonlinear_L2=False,
                 Slope_C1=0.1,
                 Slope_C2=0.1,
                 Slope_L1=3,
                 Slope_L2=0.1):
        super(Order_4_RLC_Circuit, self).__init__()
        self.nonlinear_C1 = Nonlinear_C1
        self.nonlinear_C2 = Nonlinear_C2
        self.nonlinear_L1 = Nonlinear_L1
        self.nonlinear_L2 = Nonlinear_L2




        self.C1 = torch.tensor(C1)
        self.C2 = torch.tensor(C2)
        self.R1 = torch.tensor(R1)
        self.R2 = torch.tensor(R2)
        #Rp1 and Rp2 are not used in the model (terms are commented out)
        #self.Rp1 = torch.tensor(Rp1)
        #self.Rp2 = torch.tensor(Rp2)
        self.L1 = torch.tensor(L1)
        self.L2 = torch.tensor(L2)
        self.slope_c1 = torch.tensor(Slope_C1)
        self.slope_c2 = torch.tensor(Slope_C2)
        self.slope_l1 = torch.tensor(Slope_L1)
        self.slope_l2 = torch.tensor(Slope_L2)


    def get_parameters(self):

        return {'C1': self.C1.item(),
                'C2': self.C2.item(),
                'R1': self.R1.item(),
                'R2': self.R2.item(),
                #'Rp1': self.Rp1.item(),
                #'Rp2': self.Rp2.item(),
                'L1': self.L1.item(),
                'L2': self.L2.item(),
                'slope_c1': self.slope_c1.item(),
                'slope_c2': self.slope_c2.item(),
                'slope_l1': self.slope_l1.item(),
                'slope_l2': self.slope_l2.item()}

    def print_params(self):

        print(f"C1={self.C1},C2={self.C2},R1={self.R1},R2={self.R2},L1={self.L1},L2={self.L2},slope_c1={self.slope_c1},slope_c2={self.slope_c2},slope_l1={self.slope_l1},slope_l2={self.slope_l2}")

    def nonlinear_capacitor(self, voltage, C, slope=0.1):
        c_act = torch.where(voltage >= 0,
                            C * torch.exp(-slope * voltage),
                            C * torch.exp(slope * voltage))
        return c_act

    def nonlinear_inductor(self, current, L, slope_l=3):
        # Nonlinear inductance effect, if current is near 0, l_act is L
        if current.abs() < 1e-6:
            l_act = torch.Tensor([L])
        else:
            tanh_term = torch.tanh(slope_l * current)
            l_act = L * tanh_term / (slope_l * current)
        return l_act


    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def _batch_linear_interpolation(self, t_query):

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

        C1 = self.C1
        C2 = self.C2
        R1 = self.R1
        R2 = self.R2
        #Rp1 and Rp2 are not used in the model (terms are commented out)
        #Rp1 = self.Rp1
        #Rp2 = self.Rp2
        L1 = self.L1
        L2 = self.L2
        slope_c1 = self.slope_c1
        slope_c2 = self.slope_c2
        slope_l1 = self.slope_l1
        slope_l2 = self.slope_l2

        C1_eff = self.nonlinear_capacitor(v_INT, C1, slope_c1) if self.nonlinear_C1 else C1
        C2_eff = self.nonlinear_capacitor(v_OUT, C2, slope_c2) if self.nonlinear_C2 else C2

        L1_eff = self.nonlinear_inductor(i_IN, L1, slope_l1) if self.nonlinear_L1 else L1
        L2_eff = self.nonlinear_inductor(i_INT, L2, slope_l2) if self.nonlinear_L2 else L2

        # System equations
        d_v_INT_dt = (i_IN / C1_eff) - (i_INT / C1_eff)  #- (v_INT / (Rp1 * C1_eff))
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)  #- (v_OUT / (Rp2 * C2_eff))
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)

        return torch.stack([d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt], dim=-1)
