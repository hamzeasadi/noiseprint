"""
example of prnu extraction
"""

import os
import random
from typing import List
from glob import glob
from multiprocessing import cpu_count, Pool
import sys
sys.path.append("../../../")
import argparse

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch

from noiseprint.noiseprint.Dataset.utils import get_video_frames
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Prnu import utils
from noiseprint.noiseprint.Dataset.preprocess import central_crop, intensity_croping
from noiseprint.noiseprint.Utils.gutils import save_as_pickle


def get_prnu_np(stack_imgs:torch.Tensor):
    imgs = list(stack_imgs.numpy())
    random.shuffle(imgs)
    K = utils.extract_multiple_aligned(imgs=imgs, processes=cpu_count())
    return K





# class VideoCamPrnuExtract:
#     """
#     docs
#     """
#     def __init__(self, paths:Paths, dataset_name:str="64x64xs", crop_size:List=[720, 720]) -> None:
#         # root_path = os.path.join(paths.dataset, dataset_name)
#         self.root_path = "/home/hasadi/project/Dataset/sub_videos"
#         self.half = crop_size[0]//2
        
    
#     def get_np_frames(self, cam_name:str|int):
        
#         came_path = os.path.join(self.root_path, str(cam_name))
#         frames = 0
#         video_names = os.listdir(came_path)
#         for i, video_name in enumerate(video_names):
#             video_path = os.path.join(came_path, video_name)
#             if i==0:
#                 frames = get_video_frames(video_path=video_path, fmt="THWC")
#             else:
#                 new_frames = get_video_frames(video_path=video_path, fmt="THWC")
#                 frames = torch.cat((frames, new_frames), dim=0)

#         fshape = frames.shape
#         print(f"original shape: {fshape}")
#         hc, wc = fshape[1]//2, fshape[2]//2
#         crops = frames[:, hc-self.half:hc+self.half, wc-self.half:wc+self.half, :]
#         crops = list(crops.numpy())
#         random.shuffle(crops)
#         print(crops[0].dtype, crops[0].shape)
#         K = utils.extract_multiple_aligned(imgs=crops, processes=cpu_count())
#         print(K.dtype)
#         K.tofile(f"data/prnu/{cam_name}_prnu.raw", sep=" ")
#         plt.imshow(K, cmap="gray")
#         plt.axis("off")
#         plt.savefig(f"data/figs/{cam_name}_prnu.png", bbox_inches='tight', pad_inches=0)
#         plt.close()
        





def main():
    """
    docs
    """
    # region

    # ff_dirlist = np.array(sorted(glob("data/ff-jpg/*.JPG")))
    # ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    # nat_dirlist = np.array(sorted(glob('data/nat-jpg/*.JPG')))
    # nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])
    
    # fingerprint_device = sorted(np.unique(ff_device))
    # k = []
    # for device in fingerprint_device:
    #     imgs = []
    #     for img_path in ff_dirlist[ff_device==device]:
    #         img = Image.open(img_path)
    #         img_np = np.asarray(img)
    #         if img_np.dtype != np.uint8:
    #             print(f"Error while reading {img_path}")
    #             continue
    #         if img_np.ndim != 3:
    #             print(f"Image is not RGB {img_path}")
    #             continue
            
    #         img_cnt_crop = utils.cut_ctr(array=img_np, sizes=(512, 512, 3))
    #         imgs.append(img_cnt_crop)

    #     K = utils.extract_multiple_aligned(imgs=imgs, processes=cpu_count())
    #     plt.imshow(K, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(f"data/{device}_prnu.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

        # endregion
        
    # parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="prnu extraction config")
    # parser.add_argument("--cam_name", type=int, required=True)
    # args = parser.parse_args()

    paths = Paths()
    hh = 720//2
    wh = 1280//2
    dataset_root = "/home/hasadi/project/Dataset/vision"
    for cam_name in os.listdir(dataset_root):
        flat_path = os.path.join(dataset_root, cam_name, "videos", "flat")
        frames = 0
        for i, flat_video in enumerate(os.listdir(flat_path)):
            video_path = os.path.join(flat_path, flat_video)
            if i==0:
                frames = get_video_frames(video_path=video_path, fmt="THWC")
                
            else:
                new_frames = get_video_frames(video_path=video_path, fmt="THWC")
                frames = torch.cat((frames, new_frames), dim=0)
        fshape = frames.shape
        print(cam_name)
        print(f"frame-size: {fshape}")
        frames = central_crop(imgs=frames, crop_limit=[720, 1280])
        print(f"crop-size: {frames.shape}")
        k = get_prnu_np(stack_imgs=frames)
        kn = (k - np.min(k))/(np.max(k) - np.min(k)+1e-8)
        print(f"k-shape {kn.shape}")
        save_path = os.path.join(paths.dataset, "vision", cam_name, "prnu")
        save_as_pickle(file_name=f"{cam_name}_prnu.pkl", file_path=save_path, data=dict(prnu=k))
        intensity_croping(intensity=np.expand_dims(kn, axis=0), crop_size=[64, 64], frame_size=[720, 1280], 
                          sample_counter=0, save_base=save_path, cam_name=cam_name)
        print("==="*20)




if __name__ == "__main__":
    main()

