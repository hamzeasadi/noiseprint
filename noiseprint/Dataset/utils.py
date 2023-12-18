"""
utility function
"""

import sys
sys.path.append("../../..")
from typing import List
import os

import numpy as np
import torch
from torchvision.io import read_video

from noiseprint.noiseprint.Utils.gutils import save_as_pickle
from noiseprint.noiseprint.Prnu.utils import rgb2gray






def get_video_frames(video_path:str, fmt="TCHW")->torch.Tensor:
    """
    read video and extract all frames
    """
    frames, _, _ = read_video(filename=video_path, start_pts=0.0, pts_unit='sec', output_format=fmt)
    return frames


def central_crop(imgs:torch.Tensor, crop_limit:List):
    t, h, w, c = imgs.shape
    hc = h//2
    wc = w//2
    hh, hw = crop_limit[0]//2, crop_limit[1]//2
    central_crop = imgs[:, hc-hh:hc+hh, wc-hw:wc+hw, :]
    return central_crop




def intensity_croping(intensity:np.ndarray, crop_size:List, frame_size:List, sample_counter:int, save_base:str):
    crop_counter = 0
    hc, wc = crop_size
    num_h = frame_size[0]//hc
    num_w = frame_size[1]//wc
    for i in range(num_h):
        hi = i*hc
        for j in range(num_w):
            wj = j*wc
            crop = intensity[:, hi:hi+hc, wj:wj+wc]
            data = dict(crop=crop)
            crop_name = f"crop_{sample_counter}.pkl"
            save_path = os.path.join(save_base, f"crop_{crop_counter}")
            save_as_pickle(file_name=crop_name, file_path=save_path, data=data)
            crop_counter += 1




def rgb2gray_pack(pack:torch.Tensor, num_frame_per_pack:int=3):
    intensity_list = []
    for sample_idx in range(num_frame_per_pack):
        sample = pack[sample_idx]
        intensity = rgb2gray(im=sample.numpy())
        intensity_list.append(np.expand_dims(intensity, axis=0))
    
    pack_intensity = np.concatenate(intensity_list, axis=0)
    return pack_intensity