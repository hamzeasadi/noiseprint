"""
docs
"""

import os
import sys
sys.path.append("../")
import argparse

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


def evaluation(model:nn.Module, video_path:str, base_name:str, num_seq:int=1, paths=Paths()):
    frames = get_video_frames(video_path=video_path, fmt="THWC")
    model.eval()
    for idx in range(num_seq):
        sample_pack = frames[idx:idx+3]
        intensity = rgb2gray_pack(pack=sample_pack, num_frame_per_pack=3)
        X = torch.from_numpy(intensity).unsqueeze(dim=0)
        out = model(X).detach().squeeze().numpy()
        vmin = np.min(out[34:-34,34:-34])
        vmax = np.max(out[34:-34,34:-34])
        # plt.imshow(out.clip(vmin,vmax), clim=[vmin,vmax], cmap='gray')
        plt.imshow(out, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(paths.report, f"{base_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()




def img_evaluation(model:nn.Module, base_name:str, paths=Paths()):
    img_path = "/home/hasadi/project/noiseprintPro/data/dataset/rnd_imgs/inpainting.png"
    img = Image.open(img_path)
    img_gray = img.convert("L")
    img_t = torch.from_numpy(np.asarray(img_gray)/255.0).unsqueeze(dim=0)
    imgs_name = []
    model.eval()
    
    X = torch.cat((img_t, img_t, img_t), dim=0).unsqueeze(dim=0)
    out = model(X.type(torch.float32)).detach().squeeze().numpy()
    vmin = np.min(out[34:-34,34:-34])
    vmax = np.max(out[34:-34,34:-34])
    # plt.imshow(out.clip(vmin,vmax), clim=[vmin,vmax], cmap='gray')
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(paths.report, f"{base_name}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()






def main():
    """
    docs
    """
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="evaluation checkpoint")
    parser.add_argument("--ckp_num", type=int, required=True)
    parser.add_argument("--ckp_name", type=str, required=True)
    args = parser.parse_args()


    paths = Paths()
    num_samples = 1
    video_name = "forged_YT720.mp4"
    video_path = os.path.join(paths.dataset, "valid", video_name)

    ckp_base_name = f"{args.ckp_name}_{args.ckp_num}"
    ckp_path = os.path.join(paths.model, f"{ckp_base_name}.pt")
    state = torch.load(ckp_path, map_location=torch.device("cpu"))
    print(f"epoch={state['epoch']} loss={state['loss']}")

    model = Noiseprint(input_ch=3, output_ch=1, num_layer=17)
    model.load_state_dict(state['model'])

    evaluation(model=model, video_path=video_path, num_seq=num_samples, base_name=ckp_base_name)
    img_evaluation(model=model, base_name=f"{ckp_base_name}_inpaint")




if __name__ == "__main__":

    main()

    
