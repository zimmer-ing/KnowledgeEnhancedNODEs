from abc import abstractmethod
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from pathlib import Path
import numpy as np



class TimeSeriesBase(Dataset):
    """Abstract class for time series datasets of a single file"""

    def __init__(self, data, config,channels,device=None):
        self.config = config
        assert config is not None
        assert data is not None
        self.device = device or 'cpu'
        self.data_sequences = self.create_subsequences(data, config,channels,device=self.device)

    def to(self, device):
        """
        Move all tensors in the dataset to the specified device.
        """
        self.data_sequences = [(time_seq.to(device), x_seq.to(device),
                                y_seq.to(device), z_seq.to(device))
                               for time_seq, x_seq, y_seq, z_seq in self.data_sequences]

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, index):
        return self.data_sequences[index]

    @staticmethod
    def collate_fn(batch):
        """Collates batch of data."""
        time, x, y, z = zip(*batch)
        return {"time": torch.stack(time),
                "x": torch.stack(x),
                "y": torch.stack(y),
                "z": torch.stack(z)}
    @staticmethod
    def create_subsequences(data, config,channels,device=None):
        """Creates subsequences from the data"""
        stride = config['stride']
        sequence_length = config['sequence_length']

        if sequence_length== -1:
            sequence_length = len(data)

        if stride == -1:
            stride = sequence_length
        #get names of channels
        time_channel, x_channels, y_channels,z_channels = channels
        #get data from channels
        time_data = data[time_channel].values
        x_data = data[x_channels].values
        y_data = data[y_channels].values
        if len(z_channels)>0:
            Z_data = data[z_channels].values
        else:
            Z_data = np.zeros((len(data),1))
        # Create rolling windows
        time_windows = np.lib.stride_tricks.sliding_window_view(time_data, window_shape=sequence_length)[::stride]
        x_windows = np.lib.stride_tricks.sliding_window_view(x_data, window_shape=(sequence_length, x_data.shape[1]))[
                    ::stride]
        y_windows = np.lib.stride_tricks.sliding_window_view(y_data, window_shape=(sequence_length, y_data.shape[1]))[
                    ::stride]
        z_windows = np.lib.stride_tricks.sliding_window_view(Z_data, window_shape=(sequence_length, Z_data.shape[1]))[
                    ::stride]
        # Convert to PyTorch tensors, mode to device an squeeze to remove the extra dimension created by sliding_window_view
        time_seqs = torch.from_numpy(time_windows.copy()).float().to(device).squeeze(1)
        x_seqs = torch.from_numpy(x_windows.copy()).float().to(device).squeeze(1)
        y_seqs = torch.from_numpy(y_windows.copy()).float().to(device).squeeze(1)
        z_seqs = torch.from_numpy(z_windows.copy()).float().to(device).squeeze(1)
        #return list of tuples of subsequences
        return list(zip(time_seqs, x_seqs, y_seqs,z_seqs))


class DatasetBase(Dataset):
    """Abstract class to create a dataset from a directory of files using the TimeSeriesBase class"""

    def __init__(self, directory,file_extension=".csv", config=None,device=None):
        if config is None:
            config = {}
        self.directory = directory
        self.file_extension = file_extension
        self.config = config
        self.device = device or 'cpu'


        self.timeseries_datasets = self.load_dataset()

        self.len_subsets = [len(dataset) for dataset in self.timeseries_datasets]
        self.cumulative_len_subsets = np.cumsum(self.len_subsets)

    def to(self, device):
        """
        Move all sub-datasets to the specified device.
        """
        for ts_dataset in self.timeseries_datasets:
            ts_dataset.to(device)

    def __len__(self):
        return sum(self.len_subsets)


    def __getitem__(self, index):
        #get the index of the dataset
        dataset_index = np.argmax(self.cumulative_len_subsets>index)
        if dataset_index == 0:
            sample_index = index  # No offset subtraction needed for the first subset
        else:
            # Subtract the cumulative length of all previous datasets
            sample_index = index - self.cumulative_len_subsets[dataset_index - 1]
        #get the sample
        sample = self.timeseries_datasets[dataset_index][sample_index]
        return sample

    def load_raw_dataset(self):
        """Loads the dataset from the directory"""
        channels=self.get_channels()
        directory_path = Path(self.directory)
        all_files = list(directory_path.glob(f'*{self.file_extension}'))
        time=[]
        x=[]
        y=[]
        z=[]
        filename=[]

        for file_path in all_files:
            data= self.load_file(file_path)
            #get desired channels
            filename.append(file_path.name)
            time.append(data[channels[0]])
            x.append(data[channels[1]])
            y.append(data[channels[2]])
            z.append(data[channels[3]])
        return {"time": time, "x": x, "y": y, "z": z, "filename": filename}


    def load_dataset(self):
        """Loads the dataset from the directory"""
        channels=self.get_channels()
        directory_path = Path(self.directory)
        all_files = list(directory_path.glob(f'*{self.file_extension}'))
        timeseries_datasets = []

        for file_path in all_files:
            data= self.load_file(file_path)
            # use the TimeSeriesBase class to create a dataset from the file
            dataset = TimeSeriesBase(data, self.config,channels,device=self.device)
            timeseries_datasets.append(dataset)
        return timeseries_datasets
    @staticmethod
    def collate_fn(batch):
        """Collates batch of data."""
        return TimeSeriesBase.collate_fn(batch)


    @abstractmethod
    def load_file(self, file_path) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def get_channels() -> Tuple[str, List[str], List[str], List[str]]:
        """Returns the channel names for the time, X, Y ,Z channels"""
        pass





