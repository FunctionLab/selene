import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import  sys

class SamplerDataset(data.Dataset):
    def __init__(self, sampler, size=sys.maxsize):
        super(SamplerDataset, self).__init__()
        self.sampler = sampler
        self.size = size
    
    def __getitem__(self, index):
        sequences, targets = self.sampler.sample(batch_size=1 if isinstance(index, int) else len(index))
        if sequences.shape[0]==1:
            sequences = sequences[0,:]
            targets = targets[0,:]
        return torch.from_numpy(sequences.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))
    
    def __len__(self):
        return self.size
    

class SamplerDataLoader(DataLoader):
    def __init__(self,
                 sampler,
                 num_workers=1,
                 batch_size=1,
                 size=sys.maxsize):
         args = {
             "batch_size": batch_size,
             "num_workers": num_workers,
             "pin_memory": True,
             }
         super(SamplerDataLoader, self).__init__(SamplerDataset(sampler, size=size),**args)
    
    def get_data_and_targets(self, batch_size, n_samples=None):
       return self.dataset.get_data_and_targets( batch_size, n_samples=n_samples)
    
    
