"""
utility function
"""

import sys
sys.path.append("../../..")

import torch
from torchvision.io import read_video







def get_video_frames(video_path:str, fmt="TCHW")->torch.Tensor:
    """
    read video and extract all frames
    """
    frames, _, _ = read_video(filename=video_path, start_pts=0.0, pts_unit='sec', output_format=fmt)
    return frames