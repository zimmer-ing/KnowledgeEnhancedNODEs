import torch
import torch.nn as nn
from torch.optim import Adam
from abc import ABC, abstractmethod
import json
import Constants as const
from tqdm import tqdm
from src.utils.helpers import ensure_sequential_dataloader,concatenate_batches

class BaseRegressionModel(nn.Module, ABC):
    """
    Base class for regression models
    """
    def __init__(self):
        super(BaseRegressionModel, self).__init__()

    def prepare_training(self, config):
        """Prepare the model for training"""
        self.config = config
        self.optimizer = self._initialize_optimizer()
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    def forward(self,t, x,return_z=False):
        """Forward pass logic"""
        pass

    def trainable_parameters(self):
        """Return the list of trainable parameters"""
        return list(self.parameters())


    def calculate_loss(self, predictions, targets):
        """Calculate the loss from the predictions and targets"""
        return self.loss_fn(predictions, targets)

    def _initialize_optimizer(self):
        """Initialize the Adam optimizer with the learning rate from the config"""
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        return Adam(self.trainable_parameters(), lr=lr)

    def train_step(self, data_loader,return_loss=False):
        """Train the model for one epoch on the given data_loader
        Returns:
            mean loss if return_loss is True
        """

        losses = []
        self.train()
        for batch in tqdm(data_loader,desc="Iteration Training Set",disable=not const.VERBOSE):
            inputs= batch['x']
            time=batch['time']
            targets = batch['y']
            self.optimizer.zero_grad()
            predictions = self(time,inputs)
            targets=targets.to(dtype=predictions.dtype)
            loss = self.calculate_loss(predictions, targets)
            loss.backward()
            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            #return mean loss
            return sum(losses)/len(losses)


    def predict(self, data_loader,samples_only=False):
        """Predict on the given data_loader
        Parameters:
            samples_only: if True, return only the samples
            data_loader: dataloader to predict on
        Returns:
            predictions: list of predictions
            truth: list of ground truth
            inputs: list of inputs
            ts: list of time
            z: list of latent variables"""
        inputs= []
        predictions = []
        truth= []
        ts=[]
        z=[]
        #check if dataloader is not in shuffle mode
        data_loader=ensure_sequential_dataloader(data_loader)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,desc="Iteration Prediction Set",disable=not const.VERBOSE):
                x= batch['x']
                time=batch['time']
                targets = batch['y']
                y_hat,z_batch=self(time,x,return_z=True)
                inputs.append(x)
                predictions.append(y_hat)
                truth.append(targets)
                ts.append(time)
                z.append(z_batch)
            if samples_only:
                return predictions, truth, inputs, ts, z
            predictions=concatenate_batches(predictions)
            truth=concatenate_batches(truth)
            inputs=concatenate_batches(inputs)
            ts=concatenate_batches(ts)
            z=concatenate_batches(z)
            return predictions,truth,inputs,ts,z


    def validate_step(self, data_loader):
        """Validate the model on the given data_loader
        Returns:
            mean loss
        """
        losses = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,desc="Iteration Validation Set",disable=not const.VERBOSE):
                inputs= batch['x']
                time = batch['time']
                targets = batch['y']
                predictions = self(time,inputs)
                loss = self.calculate_loss(predictions, targets)
                losses.append(loss.item())
            #return mean loss
            return sum(losses)/len(losses)

    def save_model(self, path):
        """ Save the model to the given path """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """ Load the model from the given path """
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode

    @staticmethod
    def save_config(config, path):
        """ Save the configuration to the given path (as JSON) """
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_config(path):
        """ Load the configuration from the given path (assumes JSON format) """
        with open(path, 'r') as f:
            config = json.load(f)
        return config


class NodeRegressionModel(BaseRegressionModel):
    """Base class for NODE regression models"""
    def __init__(self):
        super(NodeRegressionModel, self).__init__()


    def train_step(self, data_loader, return_loss=False):
        """Train the model for one epoch on the given data_loader
        Returns:
            mean loss if return_loss is True
        """
        losses = []
        self.train()
        for batch in tqdm(data_loader, desc="Iteration Training Set", disable=not const.VERBOSE):
            inputs = batch['x']
            time = batch['time']
            targets = batch['y']
            y_0 = targets[:, 0, :]
            self.optimizer.zero_grad()
            predictions = self(time, inputs,y_0)
            targets = targets.to(dtype=predictions.dtype)
            loss = self.calculate_loss(predictions, targets)
            loss.backward()
            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            # return mean loss
            return sum(losses) / len(losses)

    def predict(self, data_loader,samples_only=False,return_z=False):
        """Predict on the given data_loader
        Returns:
            predictions: list of predictions
            truth: list of ground truth
            inputs: list of inputs
            ts: list of time
            z: list of latent variables"""
        inputs = []
        predictions = []
        truth = []
        ts = []
        z = []
        #ensure dataloader is not in shuffle mode
        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE):
                x = batch['x']
                time = batch['time']
                targets = batch['y']
                y_0 = targets[:, 0, :]
                y_hat, z_batch = self(time, x,y_0, return_z=True)
                inputs.append(x)
                predictions.append(y_hat)
                truth.append(targets)
                ts.append(time.unsqueeze(2))
                z.append(z_batch)
            if samples_only:
                if return_z:
                    return predictions, truth, inputs, ts, z
                return predictions, truth, inputs, ts
            predictions=concatenate_batches(predictions)
            truth=concatenate_batches(truth)
            inputs=concatenate_batches(inputs)
            ts=concatenate_batches(ts)
            z=concatenate_batches(z)
            if return_z:
                return predictions, truth, inputs, ts, z

            return predictions, truth, inputs, ts

    def validate_step(self, data_loader):
        """Validate the model on the given data_loader
        Returns:
            mean loss
        """
        losses = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Validation Set", disable=not const.VERBOSE):
                inputs = batch['x']
                time = batch['time']
                targets = batch['y']
                y_0 = targets[:, 0, :]
                predictions = self(time, inputs, y_0)
                loss = self.calculate_loss(predictions, targets)
                losses.append(loss.item())
            # return mean loss
            return sum(losses) / len(losses)

class NodeRegressionModelLatEncoder(BaseRegressionModel):
    """Base class for NODE regression models with latent encoder"""
    def __init__(self):
        super(NodeRegressionModelLatEncoder, self).__init__()

    def train_step(self, data_loader, return_loss=False):
        # Example training logic using the optimizer
        losses = []
        self.train()
        for batch in tqdm(data_loader, desc="Iteration Training Set", disable=not const.VERBOSE):
            inputs = batch['x'].double()
            time = batch['time'].double()
            targets = batch['y'].double()
            latent = batch['z'].double()
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
            self.optimizer.step()
        if return_loss:
            # return mean loss
            return sum(losses) / len(losses)

    def validate_step(self, data_loader):
        """Validate the model on the given data_loader
        Returns:
            mean loss
        """
        losses = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Validation Set", disable=not const.VERBOSE):
                inputs = batch['x'].double()
                time = batch['time'].double()
                targets = batch['y'].double()
                latent = batch['z'].double()
                z_out = targets[:, 0, :]
                if self.encode_latent_space:
                    enc_input = torch.cat([inputs[:, 0:self.len_encode, :], targets[:, 0:self.len_encode, :]], dim=-1).double()
                    z_lat = self.encoder(enc_input)
                else:
                    z_lat = latent[:, 0, :]
                z_0 = torch.cat([z_out, z_lat], dim=-1)
                predictions = self(time, inputs, z_0)
                loss = self.calculate_loss(predictions, targets)
                losses.append(loss.item())
            # return mean loss
            return sum(losses) / len(losses)

    def predict(self, data_loader, samples_only=False, return_z=False):
        """Predict on the given data_loader
               Returns:
                   predictions: list of predictions
                   truth: list of ground truth
                   inputs: list of inputs
                   ts: list of time
                   z: list of latent variables"""
        inputs = []
        predictions = []
        truth = []
        ts = []
        z_hat = []
        z_true = []
        # ensure dataloader is not in shuffle mode
        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE):
                x = batch['x'].double()
                time = batch['time'].double()
                targets = batch['y'].double()
                latent = batch['z'].double()
                z_out = targets[:, 0, :]
                if self.encode_latent_space:
                    enc_input = torch.cat([inputs[:, 0:self.len_encode, :], targets[:, 0:self.len_encode, :]], dim=-1).double()
                    z_lat = self.encoder(enc_input)
                else:
                    z_lat = latent[:, 0, :]
                z_0 = torch.cat([z_out, z_lat], dim=-1)
                y_hat, z_hat_batch = self(time, x, z_0, return_z=True)
                inputs.append(x)
                predictions.append(y_hat)
                truth.append(targets)
                ts.append(time.unsqueeze(2))
                z_hat.append(z_hat_batch)
                z_true.append(latent)
            if samples_only:
                if return_z:
                    return predictions, truth, inputs, ts, z_hat, z_true
                return predictions, truth, inputs, ts
            predictions = concatenate_batches(predictions)
            truth = concatenate_batches(truth)
            inputs = concatenate_batches(inputs)
            ts = concatenate_batches(ts)
            z_hat = concatenate_batches(z_hat)
            z_true = concatenate_batches(z_true)
            if return_z:
                return predictions, truth, inputs, ts, z_hat, z_true

            return predictions, truth, inputs, ts

