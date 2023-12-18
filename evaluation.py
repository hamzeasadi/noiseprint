"""
docs
"""

import os
import sys
sys.path.append("../")

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Networks.network import Disc, Noiseprint
from noiseprint.noiseprint.Dataset.utils import get_video_frames, rgb2gray_pack


def evaluation(model:nn.Module, video_path:str, num_seq:int=1, paths=Paths()):
    frames = get_video_frames(video_path=video_path, fmt="THWC")
    model.eval()
    for idx in range(num_seq):
        sample_pack = frames[idx:idx+3]
        intensity = rgb2gray_pack(pack=sample_pack, num_frame_per_pack=3)
        X = torch.from_numpy(intensity).unsqueeze(dim=0)
        out = model(X).detach().squeeze().numpy()
        vmin = np.min(out[34:-34,34:-34])
        vmax = np.max(out[34:-34,34:-34])
        plt.imshow(out.clip(vmin,vmax), clim=[vmin,vmax], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(paths.report, f"res_{idx}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()








def main():
    """
    docs
    """
    paths = Paths()
    num_samples = 1
    video_name = "forged_YT720.mp4"
    video_path = os.path.join(paths.dataset, "valid", video_name)

    ckp_path = os.path.join(paths.model, f"ckpoint_{4}.pt")
    state = torch.load(ckp_path, map_location=torch.device("cpu"))
    model = Noiseprint(input_ch=3, output_ch=1, num_layer=15)
    model.load_state_dict(state['model'])

    evaluation(model=model, video_path=video_path, num_seq=num_samples)





if __name__ == "__main__":

    main()

    
