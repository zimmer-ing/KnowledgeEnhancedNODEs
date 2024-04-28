# Enhancing Nonlinear Electrical Circuit Modeling with Prior Knowledge-Infused Neural ODEs

This repository contains the code for the paper titled *"Enhancing Nonlinear Electrical Circuit Modeling with Prior Knowledge-Infused Neural ODEs"* by Bernd Zimmering, Jan-Philipp Roche, and Oliver Niggemann. You can access the paper [here on ResearchGate](https://www.researchgate.net/publication/383902918_Enhancing_Nonlinear_Electrical_Circuit_Modeling_with_Prior_Knowledge-Infused_Neural_ODEs).
## Abstract
This study investigates the integration of formal and informal prior knowledge into Neural Ordinary Differential Equations (NODEs) for enhancing the modeling of nonlinear electrical circuits, using a state-of-the-art LSTM as a baseline for comparison. By applying a systematic methodology to incorporate diverse knowledge types, our experiments with 2nd and 4th-order RLC circuits demonstrate improved model accuracy and efficiency. Results highlight that knowledge-infused NODEs outperform traditional LSTM models in handling complexity. Furthermore, the ability of NODEs to identify physical parameters is demonstrated.

## Table of Contents
- [Abstract](#abstract)
- [Python Version & Required Packages](#python-version--required-packages)
- [Code Structure](#code-structure)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Usage](#usage)
- [Questions and Issues](#questions-and-issues)
- [Citation](#citation)
- [License](#license)

## Python Version & Required Packages
- Python 3.11.5
- Used Packages:
```
pytorch 2.2.2
pandas 2.1.1
numpy 1.26.0
scikit-learn 1.2.2
tqdm 4.66.2
plotly 5.20.0
matplotlib 3.8.4
torchdiffeq 0.2.3
ray[default,tune] 2.10.0
optuna 3.6.1
```

We also provide a requirements.txt file for the required packages. 
```
pip install -r requirements.txt
```

## Code Structure
Our code is stored in the `src` folder. The experiments are executable `.py` files that are stored in the `experiments` folder. The `models` folder contains the code for the models. The `utils` folder contains the code for the utility functions like plots. The `data` folder contains the data. The `Constants.py` file contains all the constants used in the code (such as data path, project root, etc.). All results are stored in the `results` folder.

## Dataset
The datasets are all generated and stored in the `data` folder. Each dataset contains a `train` and `test` folder. The combination of parameters generates a hash code, which is the dataset name. The parameters are stored as `params.json` and the data is visualized as `.html` file in the `viz` folder.

## Experiments
As explained in the paper, we have two types of experiments:
- Experiment 1: This experiment identifies physical parameters of the RLC circuits using the WhiteBoxODE Model (similar to the inverse mode of PINNs).
- Experiment 2: Searches for hyperparameters based on validation loss for all models. The best hyperparameter set is then used to train the model. We repeat this procedure multiple (4) times with different seeds. Plots can be created by using the `Experiment2_postprocessing.py` file. Just make sure you use the correct paths in the file.

Experiment 1 and 2 are available in the `experiments` folder for the second and fourth-order RLC circuits.

## Usage  
To run the experiments, navigate to the `experiments` folder and execute the desired script. For example, to run the experiment for the second-order RLC circuit, execute the following command:
```
python Experiment_1_learning_parameters_2nd_Order_RLC.py
```
(Optional) Add the `src` folder to your Python path:  
In some cases, you may need to add the `src` folder to your Python path to ensure all modules are correctly recognized. You can do this as follows:

- **Windows:**
  Open a Command Prompt or PowerShell window and run:
  ```
  set PYTHONPATH=%PYTHONPATH%;%CD%
  ```
- **Linux/macOS:**
    Open a terminal window and run:
    ```
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```
## Questions and Issues
If you have any questions or issues, please open an issue in the repository. You can also write an email to [bernd.zimmering@hsu-hh.de](mailto:bernd.zimmering@hsu-hh.de)


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@INPROCEEDINGS{zimmering2024neuralodes,
  author={Zimmering, Bernd and Roche, Jan-Philipp and Niggemann, Oliver},
  booktitle={2024 IEEE International Conference on Emerging Technologies and Factory Automation (ETFA)}, 
  title={Enhancing Nonlinear Electrical Circuit Modeling with Prior Knowledge-Infused Neural ODEs}, 
  year={2024},
  volume={},
  number={},
  pages={},
  keywords={Neural ODEs;Prior Knowledge;Nonlinear Electrical Circuits;Machine Learning;Automation;Modeling},
  doi={}
}

```  

## License
This code is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). You are free to use, share, and adapt the material for any purpose, provided that you give appropriate credit, provide a link to the license, and indicate if changes were made.

Please use the [Citation](#citation) provided above when referencing this work. For more details, please refer to the [LICENSE](LICENSE) file.




