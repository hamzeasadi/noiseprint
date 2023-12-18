"""
docs
"""


import os
import argparse
import sys
sys.path.append("../")

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from noiseprint.noiseprint.Dataset.dataset import create_loader
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Networks.network import Noiseprint
from noiseprint.noiseprint.LossFunctions.loss_functions import NP_Loss
from noiseprint.noiseprint.Engine.engine import np_train




def main():
    """
    pass
    """

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="sth")

    parser.add_argument("--dev", type=str, default="cuda")
    parser.add_argument("--lamda", type=float, default=10.0)
    parser.add_argument("--glr", type=float, default=0.01)
    parser.add_argument("--ggamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    paths = Paths()
    dev = torch.device(args.dev)

    loader = create_loader()
    Gen = Noiseprint(input_ch=3, output_ch=1, num_layer=15)
    gen_crt = NP_Loss(lamda=args.lamda)
    gen_opt = Adam(params=Gen.parameters(), lr=args.glr)
    gen_sch = ExponentialLR(optimizer=gen_opt, gamma=args.ggamma)
    

    for epoch in range(args.epochs):
        np_train(gen=Gen, gen_opt=gen_opt, gen_crt=gen_crt, gen_sch=gen_sch, 
                 dataloader=loader, dev=dev, epoch=epoch, paths=paths)



if __name__ == "__main__":
    main()