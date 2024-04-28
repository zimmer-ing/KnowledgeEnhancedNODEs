from src.models.models_base import BaseRegressionModel,NodeRegressionModel
import torch.nn as nn
import torch

class LSTM(NodeRegressionModel):
    """A simple LSTM model"""

    def __init__(self, config):
        super(LSTM, self).__init__()
        assert config['data']['dim_x'] is not None
        assert config['data']['dim_y'] is not None
        assert config['model']['hidden_size_LSTM'] is not None
        assert config['model']['LSTM_layers'] in [1,2,3]

        inputs=config['data']['dim_x']
        outputs=config['data']['dim_y']
        hidden_size=config['model']['hidden_size_LSTM']
        self.y_0_to_cellstate=nn.Linear(outputs,hidden_size)
        self.y_0_to_hiddenstate=nn.Linear(outputs,hidden_size)

        self.LSTM=None
        if config['model']['LSTM_layers']>1:
            self.LSTM=nn.LSTM(inputs+outputs, hidden_size, batch_first=True, num_layers=config['model']['LSTM_layers']-1)
            self.LSTM_out=nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        else:
            self.LSTM_out=nn.LSTM(inputs+outputs, hidden_size, batch_first=True, num_layers=1)

        #linear layer to map the output of the LSTM to the desired output size
        self.linear=nn.Linear(hidden_size,outputs)

        self.prepare_training(config)


    def forward(self, t,x,y_0,return_z=False):
        #repeat y_0 along the time axis and concatenate with x
        y_0_repeated = y_0.unsqueeze(1).repeat(1, x.shape[1], 1)
        c_0_out=self.y_0_to_cellstate(y_0)
        h_0_out=self.y_0_to_hiddenstate(y_0)
        out=torch.cat([x,y_0_repeated],dim=2)

        if self.LSTM is not None:
            #int hidden cells by 0
            h_0 = torch.zeros(self.config['model']['LSTM_layers']-1, x.shape[0], self.config['model']['hidden_size_LSTM']).to(x.device)
            c_0 = torch.zeros(self.config['model']['LSTM_layers']-1, x.shape[0], self.config['model']['hidden_size_LSTM']).to(x.device)
            out,z=self.LSTM(out,(h_0,c_0))

        out, z = self.LSTM_out(out,(h_0_out.unsqueeze(0),c_0_out.unsqueeze(0)))
        out=self.linear(out)




        if return_z:
            raise NotImplementedError("return_z is not implemented for LSTM")
        else:
            return out






