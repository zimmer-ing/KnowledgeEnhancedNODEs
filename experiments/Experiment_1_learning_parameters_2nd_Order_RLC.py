from pathlib import Path
import os
import sys

PROJECT_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(Path(PROJECT_PATH)))
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import Adam
from src.datasets.datasets_base import TimeSeriesBase
from src.datasets.Simulated_dataset import Order_2_Dataset
from src.models.Whitebox_ODE_2nd_order import Whitebox_ODE_2nd_order_Model
import Constants as const
#seeds etc for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pd.options.mode.chained_assignment = None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






def create_config(nonlinear_C=True, initial_params=None):
    """
    Create configuration for the experiment
    :param nonlinear_C: Whether to use a nonlinear capacitor
    :param initial_params: Initial parameters for the model
    """
    config = {'data': {'params': {}}, 'model': {'init_params': {}}, 'training': {}}
    config['data'].update({'dim_x': 2, 'dim_y': 2, 'sequence_length': 25, 'stride': 25,
                           'params': {'Nonlinear_C': nonlinear_C, 'C': 100e-3, 'L': 1e-3, 'R': 0.1, 'Rp': 1e9, 'slope_c': 0.1}})
    config['model'].update({'learn_params': True, 'init_params': initial_params})
    config['training'].update({'learning_rate': 0.1, 'batch_size': 50, 'epochs': 300, 'epochs_pretraining': 25, 'optimizer': 'Adam'})
    return config

def setup_data_loaders(device, config):
    """
    Setup data loaders for the experiment
    :param device: Device to use
    :param config: Configuration
    """
    datasets = {x: Order_2_Dataset(dataset_type=x, device=device, config=config['data']) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=config['training']['batch_size'],
                                 shuffle=(x == 'train'), collate_fn=TimeSeriesBase.collate_fn) for x in ['train', 'val']}
    return dataloaders

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
            inputs, time, targets = batch['x'], batch['time'], batch['y']
            d_targets = targets.diff(axis=1)
            d_time = time.diff()
            dt_targets = d_targets / d_time.unsqueeze(2)
            optimizer.zero_grad()
            dt_predictions = model.ODE(time[:, 1:], targets[:, 1:, :], inputs[:, 1:, :])
            loss = torch.nn.functional.mse_loss(dt_predictions, dt_targets)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        if epoch % 5 == 0:
            print(f"Pretraining loss at epoch {epoch}: {np.mean(loss_list)}")

def train_and_evaluate_model(model, dataloaders, device, config, capture_epochs=[30,300]):
    optimizer = Adam(model.parameters(), lr=0.01)
    captured_params = {}
    # Capture configuration


    # Pretraining phase
    pretrain_model(model, dataloaders['train'], device, optimizer, config['training']['epochs_pretraining'])
    captured_params['post_pretrain'] = model.get_parameters()  # Capture parameters after pretraining

    losses = {'train': [], 'val': []}
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = model.train_step(dataloaders['train'], return_loss=True)
        losses['train'].append(train_loss)
        #ensure print in new line for better readability
        print("\n")
        print(f"Training loss at epoch {epoch}: {train_loss}")


        # Capture parameters at specific epochs
        if epoch + 1 in capture_epochs:
            captured_params[f'epoch_{epoch + 1}'] = model.get_parameters()

    return model, losses, captured_params


def convert_tensors_to_numpy(value):
    #check if value is a tensor
    if isinstance(value, torch.Tensor):
        return value.cpu().detach().numpy()
    return value

def organize_results(initial_params, captured_params):
    results = {'R': {}, 'L': {}, 'C': {}, 'Rp': {}, 'slope_c': {}}
    param_keys = ['R', 'L', 'C', 'Rp', 'slope_c']


    # Initialize with initial values
    for key in param_keys:
        results[key]['Initial'] = convert_tensors_to_numpy(initial_params[key])

    # Map captured parameters to results dictionary
    for stage in captured_params:
        for key in param_keys:
            results[key][stage] = convert_tensors_to_numpy(captured_params[stage][key])

    return results

def run_experiments(device):

    initial_params = [
        {'C': 1, 'L': 1, 'R': 1, 'Rp': 1, 'Nonlinear_C': True, 'slope_c': 1},
        {'C': 80e-3, 'L': 0.9e-3, 'R': 0.08, 'Rp': 0.9e9, 'Nonlinear_C': True, 'slope_c': 0.001},
        {'C': 1, 'L': 1.1e-3, 'R': 0.09, 'Rp': 1.1e7, 'Nonlinear_C': True, 'slope_c': 0.5},
    ]

    results_table = []

    for initial in initial_params:
        config = create_config(nonlinear_C=initial['Nonlinear_C'], initial_params=initial)
        dataloaders = setup_data_loaders(device, config)

        # Create, train, and evaluate the model
        model = Whitebox_ODE_2nd_order_Model(config).to(device)
        model, losses, captured_params = train_and_evaluate_model(model, dataloaders, device, config)

        # Organize results for table
        table_entry = organize_results(initial, captured_params)
        results_table.append(table_entry)



    # Output results to fill the table
    return results_table,config


def create_latex_table(results,config):
    # Function to format numbers in scientific notation
    def format_sci(number):
        """
        Formats a number into scientific notation with exponents rounded to the nearest multiple of 3.

        Parameters:
        - number (float): The number to format.

        Returns:
        - str: The formatted number in scientific notation.
        """
        # Return "0.0" if the number is zero
        if number == 0:
            return "0.0"

        #handle nans
        if np.isnan(number):
            return "NaN"

        # Calculate the exponent from the log base 10 of the absolute number
        exponent = np.log10(abs(number))
        exponent = int(np.floor(exponent))

        # Round the exponent down to the nearest multiple of 3
        rounded_exponent = 3 * (exponent // 3)

        # Calculate the base by dividing the original number by 10 to the power of the rounded exponent
        base = number / (10 ** rounded_exponent)

        # Format the base to display up to three decimal places
        base_formatted = f"{base:.1f}"

        # Correct the formatting if the base is close to 1000, which is not an ideal representation
        if abs(base - 1000) < 0.001:
            base = 1
            rounded_exponent += 3
            base_formatted = f"{base:.3f}"

        # Compile the final string, avoiding the exponent part if it's zero for cleaner display
        if rounded_exponent == 0:
            return f"{base_formatted}"
        else:
            return f"{base_formatted}e{rounded_exponent}"

    true_params = config['data']['params']

    # Adjusted LaTeX table generation to include a separate column for true parameters
    latex_table = r"\begin{table}[htbp]" + "\n" + \
                  r"\caption{Parameter Identification Results}" + "\n" + \
                  r"\label{tab:experiment1}" + "\n" + \
                  r"\centering" + "\n" + \
                  r"\begin{tabular}{l|llll|l}" + "\n" + \
                  r"\hline" + "\n" + \
                  r"Parameter & Initial & Post-Pretrain & 30 Epochs & 300 Epochs & True Parameter \\" + "\n" + \
                  r"\hline" + "\n"
    pram_name_map = {'C': r"$C$", 'L': r"$L$", 'R': r"$R$", 'Rp': r"$R_p$", 'slope_c': r"$\tau$"}

    for param in ['R', 'L', 'C', 'Rp', 'slope_c']:
        name_param = pram_name_map[param]
        latex_table += f"{name_param} & " + \
                       " & ".join(format_sci(results[param][key]) for key in
                                  ['Initial', 'post_pretrain', 'epoch_30', 'epoch_300']) + \
                       " & " + format_sci(true_params[param]) + r"\\" + "\n"


    latex_table += r"\hline" + "\n" + \
                   r"\end{tabular}" + "\n" + \
                   r"\end{table}"

    return latex_table


if __name__ == "__main__":

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results,config = run_experiments(device)

    latex_tables=[]

    for i in range(len(results)):
        latex_table=create_latex_table(results[i],config)
        latex_tables.append(latex_table)

    #concatenate the latex tables with a new line in between
    latex_tables="\n".join(latex_tables)

    print(latex_tables)
    # Save the LaTeX table to a file
    results_path = Path(const.RESULTS_PATH, "experiment1_2n_order_results.tex")
    with open(results_path, "w") as file:
        file.write(latex_tables)

    print(f"Results saved to {results_path}")






