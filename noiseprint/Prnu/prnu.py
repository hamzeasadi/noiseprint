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

# from noiseprint.noiseprint.Dataset.preprocess import get_video_frames
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Prnu import utils



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
        
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="prnu extraction config")
    parser.add_argument("--cam_name", type=int, required=True)
    args = parser.parse_args()

    paths = Paths()
    video_cam_prnu = VideoCamPrnuExtract(paths=paths)
    video_cam_prnu.get_np_frames(cam_name=args.cam_name)

    # K = np.fromfile(f"data/prnu/{args.cam_name}_prnu.raw", sep=" ", dtype=np.float32).reshape(1080, 1920)
    # plt.imshow(K, cmap="gray")
    # plt.axis("off")
    # plt.savefig(f"data/figs/{args.cam_name}_prnu.png", bbox_inches='tight', pad_inches=0)
    # plt.close()



if __name__ == "__main__":
    main()

