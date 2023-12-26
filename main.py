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

from noiseprint.noiseprint.Dataset.dataset import create_loader
from noiseprint.noiseprint.Utils.gutils import Paths
from noiseprint.noiseprint.Networks.network import Noiseprint, Disc
from noiseprint.noiseprint.LossFunctions.loss_functions import NP_Loss
from noiseprint.noiseprint.Engine.engine import rgan_train




def main():
    """
    pass
    """

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="sth")

    parser.add_argument("--dev", type=str, default="cuda:0")
    parser.add_argument("--lamda", type=float, default=1.1)
    parser.add_argument("--glr", type=float, default=0.01)
    parser.add_argument("--ggamma", type=float, default=0.95)
    parser.add_argument("--dlr", type=float, default=0.01)
    parser.add_argument("--dgamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--ckp_num", type=int, default=111)
    args = parser.parse_args()

    paths = Paths()
    dev = torch.device(args.dev)

    loader = create_loader()
    print("loader created!!!")

    Gen = Noiseprint(input_ch=3, output_ch=1, num_layer=15, const=True, dev=dev)
    gen_crt = NP_Loss(lamda=args.lamda)
    gen_crt.to(dev)
    gen_opt = Adam(params=Gen.parameters(), lr=args.glr, weight_decay=0.00001)
    gen_sch = ExponentialLR(optimizer=gen_opt, gamma=args.ggamma)

    if args.ckp_num != 111:
        state = torch.load(os.path.join(paths.model, "mynp", f"np_ckpoint_{args.ckp_num}.pt"), map_location='cpu')
        Gen.load_state_dict(state['model'])
        Gen.to(dev)

    disc = Disc(inch=1)
    disc_crt = nn.MSELoss(reduction='mean')
    disc_crt.to(dev)
    disc_opt = Adam(params=Gen.parameters(), lr=args.dlr, weight_decay=0.00001)
    disc_sch = ExponentialLR(optimizer=disc_opt, gamma=args.dgamma)


    print(args)
    print(dev)


    for epoch in range(args.epochs):
        rgan_train(gen=Gen, gen_opt=gen_opt, gen_crt=gen_crt, gen_sch=gen_sch,
                   disc=disc, disc_opt=disc_opt, disc_crt=disc_crt, disc_sch=disc_sch,
                   dataloader=loader, epoch=epoch, dev=dev, paths=paths)



if __name__ == "__main__":
    main()