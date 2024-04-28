import numpy as np
import torch

import Constants as const
from src.datasets.datasets_base import TimeSeriesBase,DatasetBase
from pathlib import Path
import pandas as pd
from scipy import signal
from src.models.Simulation_ODEs.RLC_circuit import Order_2_RLC_circuit,Order_4_RLC_Circuit
from torchdiffeq import odeint
from plotly import express as px
import hashlib
import json

def generate_data(order=2,input_freqency=5,t_end=10,dt=0.001,scaling=1,flip_signal=False,params=None):
    assert order in [2,4]#only order 2 and 4 are supported
    assert params is not None
    if order==2:
        #generate data from the RLC circuit
        #create an instance of the RLC circuit

        assert "C" in params
        assert "L" in params
        assert "R" in params
        assert "Rp" in params
        assert "Nonlinear_C" in params
        if params is not None:
            rlc_circuit = Order_2_RLC_circuit(C=params["C"], L=params["L"], R=params["R"], Rp=params["Rp"], C_nonlinear=params["Nonlinear_C"])
        else:
            rlc_circuit = Order_2_RLC_circuit(C=100 * 1e-3, L=1 * 1e-3, R=10 * 1e-2, Rp=1 * 1e9, C_nonlinear=False)


    elif order==4:
        assert "C1" in params
        assert "C2" in params
        assert "L1" in params
        assert "L2" in params
        assert "R1" in params
        assert "R2" in params
        #assert "Rp1" in params
        #assert "Rp2" in params
        assert "Nonlinear_C1" in params
        assert "Nonlinear_C2" in params
        assert "Nonlinear_L1" in params
        assert "Nonlinear_L2" in params
        assert "Slope_C1" in params
        assert "Slope_C2" in params
        assert "Slope_L1" in params
        assert "Slope_L2" in params
        if params is not None:
            rlc_circuit = Order_4_RLC_Circuit(**params)
        else:
            params = {'C1': 500 * 1e-6,
                      'L1': 90 * 1e-6,
                      'R1': 350 * 1e-3,
                      'Nonlinear_C1': True,
                      'Nonlinear_L1': True,
                      'Slope_C1': 0.05,
                      'Slope_L1': 0.5,
                      'C2': 200 * 1e-6,
                      'L2': 250 * 1e-6,
                      'R2': 200 * 1e-3,
                      'Nonlinear_C2': True,
                      'Nonlinear_L2': True,
                      'Slope_C2': 0.05,
                      'Slope_L2': 0.5}
            rlc_circuit = Order_4_RLC_Circuit(**params)


    ##########generate data#########
    #generate signals to excite the RLC circuit
    t=np.linspace(0,t_end,int(t_end/dt))
    # Frequency
    f = input_freqency  # Hz

    sig = (signal.square(2.1 * np.pi * f * t, duty=0.33)+1)/2
    if flip_signal:
        input=torch.tensor(np.array([sig,np.sin(2 * np.pi * f * t)])).T.unsqueeze(0)
    else:
        input=torch.tensor(np.array([np.sin(2 * np.pi * f * t),sig])).T.unsqueeze(0)
    #scale input
    input=input*scaling

    rlc_circuit.set_x(torch.tensor(t).unsqueeze(0),input)
    y_0=torch.zeros(1,order)
    #generate the output
    with torch.no_grad():
        y=odeint(rlc_circuit, y_0, torch.tensor(t),
                 method='dopri5').permute(1,0,2)


    if order==2:
        df=pd.DataFrame(np.concatenate([t.reshape(-1,1),input.squeeze(0).numpy(),y.squeeze(0).numpy()],axis=1),columns=["time","V_in","I_out","V_out","I_in"])
    elif order==4:
        df=pd.DataFrame(np.concatenate([t.reshape(-1,1),input.squeeze(0).numpy(),y.squeeze(0).numpy()],axis=1),columns=["time","V_in","I_out","V_out","I_in","V_int","I_int"])


    #return the data
    return df





def generate_hash_from_dict(param_dict):
    # Sort the dictionary and convert it to a JSON string
    # json.dumps ensures a consistent representation
    # sort_keys=True ensures the keys are sorted
    sorted_dict_string = json.dumps(param_dict, sort_keys=True)

    # Generate the hash value of the sorted dictionary string
    # Use SHA-256 hash function
    hash_object = hashlib.sha1(sorted_dict_string.encode())

    # Hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex


class Order_2_Dataset(DatasetBase):
    """
    Dataset that generates some data from the known ODE models
    """
    def __init__(self,dataset_type='train',config={"stride":50,"sequence_length":500,"order":2},*args,**kwargs):

        params=config["params"]
        hash=generate_hash_from_dict(params)
        directory = Path(const.DATA_PATH, '2nd_order_RLC', hash, dataset_type)

        if not directory.exists():
            #save parameters to directory
            directory.parent.mkdir(parents=True,exist_ok=True)
            with open(directory.parent.joinpath("params.json"), 'w') as f:
                json.dump(params, f)
            print("Dir with hash",hash,"does not exist. Generating data")
            train_data = generate_data(2, 2, 1, 0.001, flip_signal=False,params=params)
            print("Train data generated")
            val_data = generate_data(2, 1.7, 0.25, 0.001, flip_signal=False,params=params)
            print("Val data generated")
            test_data = generate_data(2, 1.5, 1, 0.001, 0.9,flip_signal=False,params=params)
            print("Test data generated")
            #save data
            path = Path(const.DATA_PATH, '2nd_order_RLC',hash, 'train')
            path.mkdir(exist_ok=True, parents=True)
            train_data.to_csv(path.joinpath("train_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '2nd_order_RLC',hash, 'val')
            path.mkdir(exist_ok=True, parents=True)
            val_data.to_csv(path.joinpath("val_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '2nd_order_RLC',hash, 'test')
            path.mkdir(exist_ok=True, parents=True)
            test_data.to_csv(path.joinpath("test_data.csv.gz"),index=False,compression='gzip')

            # visualize data
            fig_train = px.line(train_data, x="time", y=["V_in", "I_out", "V_out", "I_in"], title="Train data")
            fig_val = px.line(val_data, x="time", y=["V_in", "I_out", "V_out", "I_in"], title="Val data")
            fig_test = px.line(test_data, x="time", y=["V_in", "I_out", "V_out", "I_in"], title="Test data")

            path = Path(directory.parent, 'viz')
            path.mkdir(exist_ok=True, parents=True)
            fig_train.write_html(str(path.joinpath("train_data.html")))
            fig_val.write_html(str(path.joinpath("val_data.html")))
            fig_test.write_html(str(path.joinpath("test_data.html")))

        file_extension = ".csv.gz"
        super().__init__(directory, config=config,file_extension=file_extension,*args,**kwargs)

    def load_file(self, file_path) -> pd.DataFrame:
        data = pd.read_csv(file_path, delimiter=',',compression='gzip')
        return data




    @staticmethod
    def get_channels():
        time_channel = "time"
        x_channels = ["V_in", "I_out"]
        y_channels = ["V_out", "I_in"]
        z_channels = []

        return time_channel, x_channels, y_channels,z_channels

class Order_4_Dataset(DatasetBase):
    """
    Dataset that generates some data from the known ODE models
    """
    def __init__(self,dataset_type='train',config={"stride":50,"sequence_length":500,"order":2},*args,**kwargs):

        params=config["params"]
        hash=generate_hash_from_dict(params)
        directory = Path(const.DATA_PATH, '4th_order_RLC', hash, dataset_type)

        if not directory.exists():
            #save parameters to directory
            directory.parent.mkdir(parents=True,exist_ok=True)
            with open(directory.parent.joinpath("params.json"), 'w') as f:
                json.dump(params, f)
            print("Dir with hash",hash,"does not exist. Generating data")
            train_data = generate_data(4, 300, 0.5, 0.00005, flip_signal=False,params=params)
            print("Train data generated")
            val_data = generate_data(4, 270, 0.1, 0.00005, flip_signal=False,params=params)
            print("Val data generated")
            test_data = generate_data(4, 250, 0.5, 0.00005, 0.9,flip_signal=False,params=params)
            print("Test data generated")
            #save data
            path = Path(const.DATA_PATH, '4th_order_RLC',hash, 'train')
            path.mkdir(exist_ok=True, parents=True)
            train_data.to_csv(path.joinpath("train_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '4th_order_RLC',hash, 'val')
            path.mkdir(exist_ok=True, parents=True)
            val_data.to_csv(path.joinpath("val_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '4th_order_RLC',hash, 'test')
            path.mkdir(exist_ok=True, parents=True)
            test_data.to_csv(path.joinpath("test_data.csv.gz"),index=False,compression='gzip')

            # visualize data
            fig_train = px.line(train_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                                title="Train data")
            fig_val = px.line(val_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                              title="Val data")
            fig_test = px.line(test_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                               title="Test data")

            path = Path(directory.parent, 'viz')
            path.mkdir(exist_ok=True, parents=True)
            fig_train.write_html(str(path.joinpath("train_data.html")))
            fig_val.write_html(str(path.joinpath("val_data.html")))
            fig_test.write_html(str(path.joinpath("test_data.html")))

        file_extension = ".csv.gz"
        super().__init__(directory, config=config,file_extension=file_extension,*args,**kwargs)

    def load_file(self, file_path) -> pd.DataFrame:
        data = pd.read_csv(file_path, delimiter=',',compression='gzip')
        return data




    @staticmethod
    def get_channels():
        time_channel = "time"
        x_channels = ["V_in", "I_out"]
        y_channels = ["V_out", "I_in"]
        z_channels = ["V_int", "I_int"]

        return time_channel, x_channels, y_channels,z_channels




if __name__ == '__main__':

    #params
    config={'data': {'params': {}}, 'model': {'init_params': {}}, 'training': {}}#create empty config
    config['data']['params']['Nonlinear_C'] = True
    config['data']['params']['C'] = 100 * 1e-3
    config['data']['params']['L'] = 1 * 1e-3
    config['data']['params']['R'] = 10 * 1e-2
    config['data']['params']['Rp'] = 1 * 1e9



    params = {'C1': 500 * 1e-6,
              'L1': 90 * 1e-6,
              'R1': 350 * 1e-3,
              'Nonlinear_C1': True,
              'Nonlinear_L1': True,
              'Slope_C1': 0.05,
              'Slope_L1': 0.5,
              'C2': 200 * 1e-6,
              'L2': 250 * 1e-6,
              'R2': 200 * 1e-3,
              'Nonlinear_C2': True,
              'Nonlinear_L2': True,
              'Slope_C2': 0.05,
              'Slope_L2': 0.5}
    config['data']['params']=params

    print("Generating data for 4th order RLC circuit")
    train_data = generate_data(4, 200, 0.1, 0.00001, flip_signal=False, params=config['data']['params'])
    print("Train data generated")
    #val_data = generate_data(4, 1.7, 2, 0.001, flip_signal=False, params=config['data']['params'])
    print("Val data generated")
    #test_data = generate_data(4, 1.5, 5, 0.001, 0.9, flip_signal=False, params=config['data']['params'])
    print("Test data generated")
    print("Data generated. Plot ant saving data")
    #visualize data
    fig_train=px.line(train_data,x="time",y=["V_in","I_out","V_out","I_in","V_int","I_int"],title="Train data")
    #fig_val=px.line(val_data,x="time",y=["V_in","I_out","V_out","I_in","V_int","I_int"],title="Val data")
    #fig_test=px.line(test_data,x="time",y=["V_in","I_out","V_out","I_in","V_int","I_int"],title="Test data")

    path=Path(const.PROJECT_PATH,"data",'4th_Order_ds','viz')
    path.mkdir(exist_ok=True,parents=True)
    fig_train.write_html(str(path.joinpath("train_data.html")))
    #fig_val.write_html(str(path.joinpath("val_data.html")))
    #fig_test.write_html(str(path.joinpath("test_data.html")))






