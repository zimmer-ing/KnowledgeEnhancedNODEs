from src.models.models_base import NodeRegressionModelLatEncoder
import torch.nn as nn
import torch
torch.set_default_dtype(torch.float64)
class Encoder(nn.Module):
    """A simple encoder model"""
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


class LSTM(NodeRegressionModelLatEncoder):
    """A simple LSTM model"""

    def __init__(self, config):
        super(LSTM, self).__init__()
        assert config['data']['dim_x'] is not None
        assert config['data']['dim_y'] is not None
        assert config['model']['hidden_size_LSTM'] is not None
        assert config['model']['LSTM_layers'] in [1,2,3]

        inputs=config['data']['dim_x']
        outputs=config['data']['dim_y']
        latent=config['data']['dim_z']
        hidden_size=config['model']['hidden_size_LSTM']
        self.y_0_to_cellstate=nn.Linear(outputs+latent,hidden_size)
        self.y_0_to_hiddenstate=nn.Linear(outputs+latent,hidden_size)
        self.encode_latent_space = config['model']['encode_latent_space']
        if self.encode_latent_space:
            self.len_encode = config['model']['len_encode']
            assert config['model']['n_hidden_encode'] is not None
            assert config['model']['n_hidden_encode'] > 0
            assert config['model']['n_layers_encode'] is not None
            assert config['model']['n_layers_encode'] > 0
            self.encoder = Encoder(config)



        self.LSTM=None
        if config['model']['LSTM_layers']>1:
            self.LSTM=nn.LSTM(inputs+latent+outputs, hidden_size, batch_first=True, num_layers=config['model']['LSTM_layers']-1)
            self.LSTM_out=nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        else:
            self.LSTM_out=nn.LSTM(inputs+latent+outputs, hidden_size, batch_first=True, num_layers=1)

        #linear layer to map the output of the LSTM to the desired output size
        self.linear=nn.Linear(hidden_size,outputs)

        self.prepare_training(config)


    def forward(self, t,x,z_0,return_z=False):
        """
        Forward pass of the model
        """
        #repeat y_0 along the time axis and concatenate with x
        z_0_repeated = z_0.unsqueeze(1).repeat(1, x.shape[1], 1)
        c_0_out=self.y_0_to_cellstate(z_0)
        h_0_out=self.y_0_to_hiddenstate(z_0)
        out=torch.cat([x,z_0_repeated],dim=-1)

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






