"""
docs
"""

import sys
sys.path.append("../../..")
import os
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from noiseprint.noiseprint.Utils.gutils import Paths, load_pickle





class Cam_Dataset(Dataset):

    def __init__(self, cam_name:str|int, seq_size:int, paths:Paths) -> None:
        super().__init__()
        self.cam_name = cam_name
        self.seq_size = seq_size
        self.paths = paths
        self.all_samples = self.get_dataset_meta()
        self.dataset_size = len(self.all_samples)


    def get_dataset_meta(self)->List:
        Samples = []
        cam_imgs_path = os.path.join(self.paths.dataset, self.cam_name, "imgs")
        cam_prnu_path = os.path.join(self.paths.dataset, self.cam_name, "prnu")
        crops_names = [f for f in os.listdir(cam_imgs_path) if f.startswith("crop")]
        for crop_name in crops_names:
            crop_path = os.path.join(cam_imgs_path, crop_name)
            prnu_patch_path = os.path.join(cam_prnu_path, crop_name, "crop_0.pkl")
            patch_names = [f for f in os.listdir(crop_path) if f.endswith(".pkl")]
            random.shuffle(patch_names)
            num_patches = len(patch_names)
            for i in range(0, num_patches-self.seq_size, self.seq_size):
                patches = []
                for j in range(self.seq_size):
                    patch_path = os.path.join(crop_path, patch_names[i+j])
                    patches.append(patch_path)
                
                sample = dict(crops=patches, prnu=prnu_patch_path, label=self.cam_name)
                Samples.append(sample)
        
        return Samples


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index) -> Dict:
        sample_info = self.all_samples[index]
        crops = []
        labels = []
        for crop_path in sample_info['crops']:
            data = load_pickle(file_path=crop_path)
            crops.append(torch.from_numpy(data['crop']).unsqueeze(dim=0))
            labels.append(data['label'])
        
        X = torch.cat(crops, dim=0)
        y = torch.tensor(labels)

        prnu_crop = load_pickle(sample_info['prnu'])
        pcrop = torch.from_numpy(prnu_crop['crop']).unsqueeze(dim=0)

        return (X, y, pcrop)
    



def create_loaders(dataset_name:str, paths:Paths, batch_size:int=1, seq_size:int=7):

    dataset_root = os.path.join(paths.dataset, dataset_name)
    loaders = []
    for cam_name in os.listdir(dataset_root):
        dataset = Cam_Dataset(cam_name=cam_name, seq_size=seq_size, paths=paths)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    
    return loaders






def create_batch(batch:Tuple):
    all_X = 0
    all_Y = 0
    all_prnu = 0
    for i, sbatch in enumerate(batch):
        if i==0:
            all_X = sbatch[0].squeeze()
            all_Y = sbatch[1]
            all_prnu = sbatch[2]
        else:
            all_X = torch.cat((all_X, sbatch[0].squeeze()), dim=0)
            all_y = torch.cat((all_Y, sbatch[1]), dim=0)
            all_prnu = torch.cat((all_prnu, sbatch[2]), dim=0)

    
    return all_X, all_Y, all_prnu






if __name__ == "__main__":
    print(__file__)

    paths = Paths()

    loaders = create_loaders(dataset_name="vision", paths=paths, seq_size=10)
    for bbatch in zip(*loaders):
        print(bbatch)
        break