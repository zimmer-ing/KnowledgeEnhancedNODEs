from pathlib import Path
import os
import sys

PROJECT_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(Path(PROJECT_PATH)))

import torch
import numpy as np
import pandas as pd
import json

from torch.utils.data import DataLoader
from torch.optim import Adam
from src.datasets.datasets_base import TimeSeriesBase
from src.datasets.Simulated_dataset import Order_4_Dataset
from src.models.Whitebox_ODE_4th_order import Whitebox_ODE_4th_order_Model
from src.models.GreyBox_NODE_4th_order import GreyBoxODEModel
from src.models.BlackBox_NODE_4th_order import BlackBoxODEModel
from src.models.Inital_state_LSTM_4th_order import LSTM
import Constants as const
from ray import tune,train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
torch.set_default_dtype(torch.float64)
import ray
#seeds etc for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pd.options.mode.chained_assignment = None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






# Seed setup for reproducibility
def setup_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration management
def create_config(model_type='KnownODE',init_parameters=None,FLAG_TEST=False):
    """
    Create configuration for the experiment
    :param model_type: Type of the model
    """
    # Base configuration setup
    config = {
        'data': {'dim_x': 2, 'dim_y': 2, 'dim_z':2,'sequence_length': 50, 'stride': 25,'params': {}},
        'training': {'learning_rate': 0.01,
                     'batch_size': 50,
                     'epochs': 200,
                     'epochs_hparam_tuning': 100, #number of epochs for hyperparameter tuning
                     'epochs_pretraining': 100,
                     'optimizer': 'Adam',
                     'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    }

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

    config['data']['params'] = params



    if init_parameters is None:
        init_parameters =  {
         'C1': 500 * 1e-3,
         'L1': 90 * 1e-3,
         'R1': 300 * 1e-3,
         'Nonlinear_C1': True,
         'Nonlinear_L1': True,
         'Slope_C1': 0.01,
         'Slope_L1': 0.1,
         'C2': 200 * 1e-3,
         'L2': 250 * 1e-3,
         'R2': 150 * 1e-3,
         'Nonlinear_C2': True,
         'Nonlinear_L2': True,
         'Slope_C2': 0.01,
         'Slope_L2': 0.1}

    if FLAG_TEST:
        config['training']['epochs']=1
        config['training']['epochs_pretraining']=1


    # Adding model-specific configurations
    if model_type == 'WhiteBoxODE':
        config['model'] = {'learn_params': True, 'init_params': init_parameters}
    elif model_type == 'GreyBoxODE1':
        config['model'] = {'hidden_size_NN': 10, 'init_params': init_parameters}
        config['model']['Greybox_type'] = 'GreyboxODE1'
    elif model_type == 'GreyBoxODE2':
        config['model'] = {'hidden_size_NN': 10, 'init_params': init_parameters}
        config['model']['Greybox_type'] = 'GreyboxODE2'
    elif model_type == 'GreyBoxODE3':
        config['model'] = {'hidden_size_NN': 10, 'hidden_size_C_NN': 10, 'init_params': init_parameters}
        config['model']['Greybox_type'] = 'GreyboxODE3'
    elif model_type == 'BlackBoxODE':
        config['model'] = {'hidden_size_ODE': 10}
    elif model_type == 'BlackBoxLSTM':
        config['model'] = {'hidden_size_LSTM': 10}
        config['model'] = {'LSTM_layers': 1}
        config['training']['epochs_pretraining'] = 0 #no pretraining for LSTM

    config['model']['encode_latent_space'] = False
    #config['model']['n_hidden_encode'] = 10
    #config['model']['n_layers_encode'] = 1
    #config['model']['len_encode'] = 25

    return config

# Data loading
def setup_data_loaders(device, config):
    datasets = {x: Order_4_Dataset(dataset_type=x, device=device, config=config['data']) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(datasets[x],
                                 batch_size=config['training']['batch_size'],
                                 shuffle=(x == 'train'),
                                 collate_fn=TimeSeriesBase.collate_fn) for x in ['train', 'val', 'test']}
    return dataloaders

# Model setup
def setup_model(model_type, config):
    assert model_type in ['WhiteBoxODE', 'GreyBoxODE1', 'GreyBoxODE2', 'GreyBoxODE3','BlackBoxODE','BlackBoxLSTM']
    if model_type == 'WhiteBoxODE':
        model= Whitebox_ODE_4th_order_Model(config)
    elif model_type == 'GreyBoxODE1':
        model= GreyBoxODEModel(config)
    elif model_type == 'GreyBoxODE2':
        model= GreyBoxODEModel(config)
    elif model_type == 'GreyBoxODE3':
        model= GreyBoxODEModel(config)
    elif model_type == 'BlackBoxODE':
        model= BlackBoxODEModel(config)
    elif model_type == 'BlackBoxLSTM':
        model=LSTM(config)
    return model


def setup_searchspace(model_type):
    search_space = {'training':
                        {'learning_rate': tune.loguniform(1e-4, 5e-1),
                        'batch_size': tune.choice([50,  150, 200,250]),
                         },
                    'model':{}
                    }

    if model_type == 'GreyBoxODE1' or model_type == 'GreyBoxODE2' or model_type == 'GreyBoxODE3':
        search_space['model']['hidden_size_NN'] = tune.choice([5, 15, 25, 35,45])
    elif model_type == 'BlackBoxODE':
        search_space['model']['hidden_size_ODE'] = tune.choice([5, 15, 25, 35,45])
        search_space['model']['layers_hidden_ODE'] = tune.choice([1, 2, 3])
    elif model_type == 'BlackBoxLSTM':
        search_space['model']['hidden_size_LSTM'] = tune.choice([5, 15, 25, 35,45])
        search_space['model']['LSTM_layers'] = tune.choice([1, 2, 3])


    return search_space
def pretrain_model(model, dataloader, device, optimizer, epochs_pretraining):
    """
    Pretraining the model with the initial parameter guesses
    :param model: The model to pretrain
    :param dataloader: Data for training
    :param device: Device to use
    :param optimizer: Optimizer for training
    :param epochs_pretraining: Number of pretraining epochs
    """
    model.train()
    for epoch in range(epochs_pretraining):
        loss_list = []
        for batch in dataloader:
            inputs, time, y,z = batch['x'], batch['time'], batch['y'], batch['z']
            #use double
            inputs = inputs.double().to(device)
            time = time.double().to(device)
            y = y.double().to(device)
            z = z.double().to(device)
            targets =torch.cat((y, z), dim=-1)
            d_targets = targets.diff(axis=1)
            d_time = time.diff()
            dt_targets = d_targets / d_time.unsqueeze(2)
            optimizer.zero_grad()
            dt_predictions = model.ODE(time[:, 1:], targets[:, 1:, :], inputs[:, 1:, :])
            loss = torch.nn.functional.mse_loss(dt_predictions, dt_targets)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if epoch % 100 == 0:
                print(f"Pretraining loss at epoch {epoch}: {np.mean(loss_list)}")
# Training function
def train_model(model, dataloaders, device, config):
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    if config['training']['epochs_pretraining'] > 0:
        pretrain_model(model, dataloaders['train'],device, optimizer, config['training']['epochs_pretraining'])

    losses_df = pd.DataFrame(columns=['epochs','train', 'val','test'])
    #check losses after pretraining
    model.eval()
    loss_train = model.train_step(dataloaders['train'], return_loss=True)
    loss_val = model.validate_step(dataloaders['val'])
    loss_test = model.validate_step(dataloaders['test'])
    losses_df.loc[0] = [0,loss_train,loss_val,loss_test]
    for epoch in range(1,config['training']['epochs']):
        model.train()
        train_loss = model.train_step(dataloaders['train'], return_loss=True)
        print(f"Epoch {epoch} - Training loss: {train_loss}")
        if train_loss == np.nan or train_loss == np.inf or train_loss == 'nan':
            print('Training loss is NaN, stopping training')
            break
        model.eval()
        val_loss = model.validate_step(dataloaders['val'])
        test_loss = model.validate_step(dataloaders['test'])
        losses_df.loc[epoch] = [epoch,train_loss,val_loss,test_loss]
    return losses_df
def recursive_update(base_dict, new_dict):
    """
    Recursively updates base_dict with values from new_dict. If both base_dict and new_dict
    contain a dictionary at a given key, then it merges those dictionaries via recursive call.
    Otherwise, it updates the value in base_dict with the value in new_dict.
    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value

    return base_dict
def train_hparams(config, base_config,model_type, device):
    # Merge hyperparameter tuning configurations with base configurations

    new_config = recursive_update(base_config, config)
    dataloaders =setup_data_loaders(device, new_config)
    model = setup_model(model_type, new_config)
    model.to(device).double()

    optimizer = Adam(model.parameters(), lr=new_config['training']['learning_rate'])
    if new_config['training']['epochs_pretraining'] > 0:
        pretrain_model(model, dataloaders['train'], device, optimizer, new_config['training']['epochs_pretraining'])

    for epoch in range(new_config['training']['epochs_hparam_tuning']):
        model.train()
        loss_train = model.train_step(dataloaders['train'])
        if loss_train == np.nan or loss_train== np.inf or loss_train == 'nan':
            print('Training loss is NaN, stopping training')
            break

        model.eval()
        loss_val = model.validate_step(dataloaders['val'])
        # Report metrics to Ray Tune
        train.report(metrics={"loss": loss_val, "train_loss": loss_train,'epoch':epoch})



    return {"loss": loss_val}




# Hyperparameter tuning
def tune_hyperparameters(base_config, model_type, device, num_samples=25, seed=42):
    #check if we want to debug or using the debugger
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        debugger = True
        print('Ray enters debugging mode using a single local instance')
    else:
        debugger = False

    ray.init(local_mode=(debugger or FLAG_TEST))


    search_space = setup_searchspace(model_type)

    if FLAG_TEST:
        num_samples=1

    if device.type == 'cuda':
        num_gpus = 0.5
    else:
        num_gpus = 0

    # Configure the scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=base_config['training']['epochs_hparam_tuning']+1,  # Maximum number of training iterations (epochs)
        grace_period=50,  # Minimum number of epochs to run before stopping poorly performing trials
        reduction_factor=4)
    search_alg = OptunaSearch(seed=seed)
    trainable = tune.with_resources(train_hparams, {"cpu": 4, "gpu": num_gpus})
    trainable = tune.with_parameters(trainable, base_config=base_config, model_type=model_type, device=device)

    # Create the tuner and start the tuning process
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=num_samples,
            scheduler=scheduler,
            metric="loss",
            mode="min",
        ),
        param_space=search_space,
    )

    analysis = tuner.fit()
    best_trial = analysis.get_best_result("loss", "min", "all")
    print('Hyperparameter tuning for model {} with seed {} has finished.'.format(model_type,seed))
    print(analysis.num_terminated, "have been completed.",analysis.num_errors , " trails have errored out.")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}  after {} epochs".format(best_trial.metrics['loss'],best_trial.metrics['training_iteration']-1))
    aux_results = analysis.get_dataframe()
    ray.shutdown()

    best_hyperparameters = recursive_update(base_config, best_trial.config)
    return best_hyperparameters, best_trial.metrics, aux_results


# Run experiments
def run_experiments(name_models,device,seeds,path):

    results={model : {} for model in name_models}
    for name_model in name_models:
        results[name_model]={seed : {} for seed in seeds}
        for seed in seeds:
            setup_seeds(seed)
            config=create_config(model_type=name_model,FLAG_TEST=FLAG_TEST)
            dataloaders=setup_data_loaders(device,config) #this is done again inside the hyperparameter tuning as ray tune does not 'like' big objects to be passed
            hyperparameters,metrics,aux_results=tune_hyperparameters(config,name_model,device, num_samples=12, seed=seed)
            #save the hyperparameters and results
            path_hparams=Path(path,f'{name_model}_seed_{seed}')
            path_hparams.mkdir(parents=True,exist_ok=True)
            with open(Path(path_hparams,'hyperparameters.json'),'w') as f:
                json.dump(hyperparameters,f)
            with open(Path(path_hparams,'metrics.json'),'w') as f:
                json.dump(metrics,f)
            aux_results.to_csv(Path(path_hparams,'aux_results.csv'))
            #load the best hyperparameters
            #with open(Path(path_hparams,'hyperparameters.json'),'r') as f:
                #hyperparameters=json.load(f)

            #train the model with the best hyperparameters
            model=setup_model(name_model,hyperparameters)
            model.to(device)
            res_df=train_model(model, dataloaders, device, hyperparameters)
            results[name_model][seed]=res_df
            #save the results
            res_df.to_csv(Path(path_hparams,'results_training.csv'))
    return results

# Main entry point
if __name__ == "__main__":
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    FLAG_TEST = False
    seeds = [0, 42, 123, 205]
    name_models=['BlackBoxODE','BlackBoxLSTM','GreyBoxODE1','GreyBoxODE2','GreyBoxODE3','WhiteBoxODE']
    name_experiment = Path(__file__).stem
    path = Path(const.RESULTS_PATH, name_experiment)
    results = run_experiments(name_models,device, seeds, path)

    # go through all results and mean them over the seeds, return the mean and std in dataframes
    results_mean_std = {}
    for name_model in name_models:
        results_mean_std[name_model] = {}
        #aggregate the all dataframes for each model
        for key in results[name_model][seeds[0]].keys():
            results_mean_std[name_model][key] = pd.concat([results[name_model][seed][key] for seed in seeds]).groupby(level=0).agg(['mean', 'std'])


    # save the results
    for name_model in name_models:
        for key in results_mean_std[name_model].keys():
            results_mean_std[name_model][key].to_csv(Path(path, f'{name_model}_{key}_mean_std.csv'))







