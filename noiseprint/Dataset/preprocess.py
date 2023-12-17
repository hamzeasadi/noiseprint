"""
general preprocessing for the dataset
"""

import os
import sys
sys.path.append("../../../")
from typing import List
from typing import Dict

import numpy as np
import torch
from torchvision.io import read_video

from noiseprint.noiseprint.Utils.gutils import save_as_pickle
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Dataset.utils import get_video_frames
from noiseprint.noiseprint.Prnu.utils import rgb2gray
from noiseprint.noiseprint.Prnu.prnu import get_prnu_np

def rgb2gray_pack(pack:torch.Tensor, num_frame_per_pack:int=3):
    intensity_list = []
    for sample_idx in range(num_frame_per_pack):
        sample = pack[sample_idx]
        intensity = rgb2gray(im=sample.numpy())
        intensity_list.append(np.expand_dims(intensity, axis=0))
    
    pack_intensity = np.concatenate(intensity_list, axis=0)
    return pack_intensity



def intensity_croping(intensity:np.ndarray, crop_size:List, frame_size:List, sample_counter:int, save_base:str, cam_name:str|int):
    crop_counter = 0
    hc, wc = crop_size
    num_h = frame_size[0]//hc
    num_w = frame_size[1]//wc
    for i in range(num_h):
        hi = i*hc
        for j in range(num_w):
            wj = j*wc
            crop = intensity[:, hi:hi+hc, wj:wj+wc]
            data = dict(crop=crop, label=int(cam_name))
            crop_name = f"crop_{sample_counter}.pkl"
            save_path = os.path.join(save_base, f"crop_{crop_counter}")
            save_as_pickle(file_name=crop_name, file_path=save_path, data=data)
            crop_counter += 1
            

    

def central_crop(imgs:torch.Tensor, crop_limit:List):
    t, h, w, c = imgs.shape
    hc = h//2
    wc = w//2
    hh, hw = crop_limit[0]//2, crop_limit[1]//2
    central_crop = imgs[:, hc-hh:hc+hh, wc-hw:wc+hw, :]
    return central_crop




class Vison_video_sampling:
    """
    get patch samples
    """

    def __init__(self, dataset_name:str, central_crop_size:List, crop_size:List, 
                 num_samples:Dict, paths:Paths, videos_dir:str) -> None:

        self.crop_size = crop_size
        self.dataset_name = dataset_name
        self.paths = paths
        self.central_crop_size = central_crop_size
        self.videos_dir = videos_dir
        self.dataset_root_path = os.path.join(videos_dir, dataset_name)
        self.num_samples = num_samples
        


    
    def get_cam_samples(self, cam_name:str):
        """
        we suppose we have flate, indoor and outdoor folders for each cam videos
        """
        sample_counter = 0
        # flat, indoor and outdoor folders
        cam_videos_path = os.path.join(self.dataset_root_path, cam_name, "videos")
        fio_folders = os.listdir(cam_videos_path)
        for video_type in fio_folders:
            frames = 0
            video_type_path = os.path.join(cam_videos_path, video_type)
            for i, video_name in enumerate(os.listdir(video_type_path)):
                video_path = os.path.join(video_type_path, video_name)
                if i==0:
                    frames = get_video_frames(video_path=video_path, fmt="THWC")
                else:
                    new_frames = get_video_frames(video_path=video_path, fmt="THWC")
                    frames = torch.cat((frames, new_frames), dim=0)
            
            frames = central_crop(imgs=frames, crop_limit=self.central_crop_size)
            sample_counter = self.get_video_samples(video_frames=frames, num_sample=self.num_samples[video_type], 
                                   cam_name=cam_name, count_samples=sample_counter)
            # if video_type == "flat":
            #     k = get_prnu_np(stack_imgs=frames)
            #     kdata = dict(k=k)
            #     k_save_path = os.path.join(self.paths.dataset, self.dataset_name, cam_name, "prnu")
            #     save_as_pickle(file_name="kdata.pkl", file_path=k_save_path, data=kdata)
            #     intensity_croping(intensity=np.expand_dims(k, axis=0), crop_size=self.crop_size, frame_size=self.central_crop_size, 
            #                       sample_counter=0, save_base=k_save_path, cam_name=cam_name)




    def get_video_samples(self, video_frames:torch.Tensor, num_sample:int, cam_name:str, count_samples:int):
        num_frames, h, w, c = video_frames.shape
        frames_indexs = np.linspace(start=0, stop=num_frames-5, num=num_sample)
        base_save_path = os.path.join(self.paths.dataset, self.dataset_name, cam_name, "imgs")
        self.paths.crtdir(base_save_path)

        for frame_index in frames_indexs:
            idx = int(frame_index)
            sample_pack = video_frames[idx:idx+3]
            intensity_sample = rgb2gray_pack(pack=sample_pack)
            intensity_croping(intensity=intensity_sample, crop_size=self.crop_size, frame_size=self.central_crop_size, 
                              sample_counter=count_samples, save_base=base_save_path, cam_name=cam_name)
            count_samples += 1
        
        return count_samples



    def run(self):
        for cam_name in os.listdir(self.dataset_root_path):
            self.get_cam_samples(cam_name=cam_name)
            

        
        












def main():
    """docs"""

    paths = Paths()

    root_ext_path = "/home/hasadi/project/Dataset"
    sample_per_type = dict(flat=50, indoor=100, outdoor=150)
    vision_video_sampling = Vison_video_sampling(dataset_name="vision", central_crop_size=[720, 1280], crop_size=[64, 64], 
                 num_samples=sample_per_type, paths=paths, videos_dir=root_ext_path)

    
    
    vision_video_sampling.run()





if __name__ == "__main__":

    main()