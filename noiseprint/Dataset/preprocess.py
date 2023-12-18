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
from noiseprint.noiseprint.Dataset.utils import get_video_frames, central_crop, intensity_croping, rgb2gray_pack





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
            




def extract_samples(root_dir:str, dataset_name:str, seq_len:int=3, num_samples:int=400, paths:Paths=Paths()):
    dataset_path = os.path.join(root_dir, dataset_name)
    crop_counter = 0
    for cam_name in os.listdir(dataset_path):
        cam_path = os.path.join(dataset_path, cam_name)
        for i, video_name in enumerate(os.listdir(cam_path)):
            video_path = os.path.join(cam_path, video_name)
            try:
                if i==0:
                    frames = get_video_frames(video_path=video_path, fmt="THWC")
                else:
                    new_frames = get_video_frames(video_path=video_path, fmt="THWC")
                    frames = torch.cat((frames, new_frames), dim=0)
            except Exception as e:
                print(f"{cam_name}: {video_name} : {e}")

        frames = central_crop(imgs=frames, crop_limit=[720, 1280])
        t,h,w,c = frames.shape
        indices = np.linspace(start=0, stop=t-seq_len, num=num_samples)
        smple_counter = 0
        for index in indices:
            idx = int(index)
            seq_frames = frames[idx:idx+seq_len]
            seq_intensity = rgb2gray_pack(pack=seq_frames, num_frame_per_pack=seq_len)
            cam_crop_counter = 0
            num_h, num_w = h//64, w//64
            counter = crop_counter*(num_h*num_w)
            
            for h_idx in range(num_h):
                hi = h_idx*64
                for w_idx in range(num_w):
                    wi = w_idx*64
                    crop = seq_intensity[:, hi:hi+64, wi:wi+64]
                    data = dict(crop=crop, label=cam_crop_counter+counter)
                    sav_path = os.path.join(paths.dataset, dataset_name, f"crop_{cam_crop_counter+counter}")
                    save_as_pickle(file_name=f"patch_{smple_counter}.pkl", file_path=sav_path, data=data)
                    cam_crop_counter += 1
            smple_counter += 1

        crop_counter += 1

            









def main():
    """docs"""

    paths = Paths()
    video_ext = "/home/hasadi/project/Dataset"
    dataset_name = "socraties"

    extract_samples(root_dir=video_ext, dataset_name=dataset_name)


if __name__ == "__main__":

    main()