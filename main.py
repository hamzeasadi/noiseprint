"""
docs
"""


import os
import argparse
import sys
sys.path.append("../")

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from noiseprint.noiseprint.Dataset.dataset import create_loaders
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Networks.network import Disc, Noiseprint
from noiseprint.noiseprint.LossFunctions.loss_functions import NP_Loss
from noiseprint.noiseprint.Engine.engine import init_train




def main():
    """
    pass
    """

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="sth")

    parser.add_argument("--dev", type=str, default="cuda")
    parser.add_argument("--lamda", type=float, default=10.0)
    parser.add_argument("--glr", type=float, default=0.01)
    parser.add_argument("--dlr", type=float, default=0.01)
    parser.add_argument("--ggamma", type=float, default=0.9)
    parser.add_argument("--dgamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    paths = Paths()
    dev = torch.device(args.dev)

    loaders = create_loaders(dataset_name="socraties", paths=paths, seq_size=10)
    Gen = Noiseprint(input_ch=3, output_ch=1, num_layer=15)
    disc = Disc(inch=1)

    gen_crt = NP_Loss(lamda=args.lamda)
    disc_crt = nn.BCEWithLogitsLoss()

    gen_opt = Adam(params=Gen.parameters(), lr=args.glr)
    disc_opt = Adam(params=disc.parameters(), lr=args.dlr)

    gen_sch = ExponentialLR(optimizer=gen_opt, gamma=args.ggamma)
    disc_sch = ExponentialLR(optimizer=disc_opt, gamma=args.dgamma)

    for epoch in range(args.epochs):

        init_train(gen=Gen, disc=disc, gen_opt=gen_opt, disc_opt=disc_opt, gen_sch=gen_sch, paths=paths,
                disc_sch=disc_sch, gen_crt=gen_crt, disc_crt=disc_crt, dataset=loaders, epoch=epoch, dev=dev)



if __name__ == "__main__":
    main()