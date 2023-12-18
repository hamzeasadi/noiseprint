"""
docs
"""

import sys
sys.path.append("../../..")
import os
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data import DataLoader

from noiseprint.noiseprint.Utils.gutils import Paths, load_pickle
from noiseprint.noiseprint.Utils.gutils import load_json




class SocDataset(Dataset):
    """docs"""
    def __init__(self, paths:Paths, dataset_name:str, pack_size:int) -> None:
        super().__init__()
        self.pack_size = pack_size
        self.paths = paths
        self.dataset_name = dataset_name
        self.sample_dict = self.get_samples_list()
        self.dataset_size = len(list(self.sample_dict.keys()))


    def get_samples_list(self)->Dict:
        Samples = dict()
        dataset_path = os.path.join(self.paths.dataset, self.dataset_name)
        crop_names = [f for f in os.listdir(dataset_path) if f.startswith("crop")]
        sample_counter = 0
        for crop_name in crop_names:
            crop_path = os.path.join(dataset_path, crop_name)
            crop_patches = [f for f in os.listdir(crop_path) if f.startswith("patch")]
            num_patches = len(crop_patches)
            for i in range(0, num_patches, self.pack_size):
                sample = []
                for j in range(self.pack_size):
                    sample_path = os.path.join(crop_path, crop_patches[i+j])
                    sample.append(sample_path)
                
                Samples[sample_counter] = sample
                sample_counter += 1

        return Samples
    


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        sample_info = self.sample_dict[index]
        xx = []
        yy = []
        for sample_path in sample_info:
            data = load_pickle(file_path=sample_path)
            xx.append(torch.from_numpy(data['crop']).unsqueeze(dim=0))
            yy.append(torch.tensor(data['label']))
        
        return torch.cat(xx, dim=0), torch.tensor(yy)
    
        


def custome_collate_0(data):
    X = data[0][0]
    Y = data[0][1]
    for i in range(1, len(data)):
        x, y = data[i][0], data[i][1]
        X = torch.cat((X, x), dim=0)
        Y = torch.cat((Y, y), dim=0)
    
    return X, Y



def custom_collate_1(batch):
    transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
    return torch.cat(transposed[0], dim=0), torch.cat(transposed[1], dim=0)


def create_loader(config_name:str="loader_config.json", paths:Paths=Paths()):
    config = load_json(os.path.join(paths.config, config_name))
    dataset = SocDataset(paths=paths, dataset_name=config['dataset_name'], pack_size=config["pack_size"])
    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], collate_fn=custom_collate_1,
                        shuffle=True, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    
    return loader








if __name__ == "__main__":
    print(__file__)

    paths = Paths()

    # dataset = SocDataset(paths=paths, dataset_name="socraties", pack_size=5)

    # print(len(dataset))
    # x, y = dataset[0]
    # print(x.shape)
    # print(y)


    # x = [(torch.randn(size=(3,4,4)), torch.tensor([1,2])), (torch.randn(size=(3,4,4)), torch.tensor([3,4])), 
    #      (torch.randn(size=(3,4,4)), torch.tensor([5, 6])), (torch.randn(size=(3,4,4)), torch.tensor([7,8]))]
    # transposed = list(zip(*x))  # It may be accessed twice, so we use a list.
    # X = torch.cat(transposed[0], dim=0)
    # y = torch.cat(transposed[1], dim=0)
    # # out =  [default_collate(samples) for samples in transposed]  # Backwards compatibility.

    # print(X.shape)
    # print(y)

    loader = create_loader()

    for X, y in loader:
        print(X.shape)
        print(y.shape)
        print(y)
        break