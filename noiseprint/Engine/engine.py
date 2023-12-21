"""
training engine
"""

import os
import sys
sys.path.append("../../..")
from typing import List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# from noiseprint.noiseprint.Dataset.dataset import create_batch




def odd_eve_lbl(batch_size:int):
    odd = torch.arange(start=1, end=batch_size, step=2)
    even = torch.arange(start=0, end=batch_size, step=2)

    return odd, even




def np_train(gen:nn.Module, gen_opt:Optimizer, gen_crt:nn.Module, dev:torch.device,
             gen_sch:nn.Module, dataloader:DataLoader, epoch:int, paths):
    print(epoch)
    train_loss = 0
    gen.to(dev)
    gen.train()
    num_batches = len(dataloader)
    cntr = 0
    for X, y in dataloader:
        out = gen(X.to(dev))
        loss = gen_crt(embeddings=out, labels=y.to(dev))
        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()
        train_loss+=loss.item()
        if cntr%10==0:
            print(f"epoch={epoch} loss={train_loss/10}")

        cntr+=1

    
    state = dict(model=gen.eval().state_dict(), loss=train_loss/num_batches, epoch=epoch)
    torch.save(state, f=os.path.join(paths.model, f"np_ckpoint_{epoch}.pt"))






def rgan_train(gen:nn.Module, gen_opt:Optimizer, gen_crt:nn.Module, gen_sch:nn.Module,
               disc:nn.Module, disc_opt:Optimizer, disc_crt:nn.Module, disc_sch:nn.Module,
               dataloader:DataLoader, epoch:int, dev:torch.device, paths):
    
    train_loss = 0
    num_batches = len(dataloader)
    gen.to(dev)
    disc.to(dev)
    gen.train()
    disc.train()
    cntr = 0
    for X, Y in dataloader:
        b,c,h,w = X.shape
        out = gen(X.to(dev))
        odd_indices, even_indices = odd_eve_lbl(batch_size=b)
        fake = out[odd_indices].detach()
        real = out[even_indices].detach()

        # train discriminator
        disc_fake = disc(fake)
        disc_real = disc(real)
        disc_loss = disc_crt(disc_real - disc_fake.mean(dim=0, keepdim=True), torch.ones_like(disc_real, requires_grad=False))
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # train generator
        disc_fake = disc(out[odd_indices])
        disc_real = disc(out[even_indices])
        disc_loss = disc_crt(disc_fake - disc_real.mean(dim=0, keepdim=True), torch.ones_like(disc_real, requires_grad=False))
        gen_loss = disc_loss + gen_crt(embeddings=out, labels=Y.to(dev))
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        train_loss += gen_loss.item()

        if cntr%100 == 0:
            print(f"epoch={epoch} loss={gen_loss.item()}")
        cntr += 1


    if gen_sch is not None:
        gen_sch.step()
    
    if disc_sch is not None:
        disc_sch.step()

    
    state = dict(model=gen.eval().state_dict(), epoch=epoch, loss=train_loss/num_batches)
    torch.save(obj=state, f=os.path.join(paths.model, f"gan_ckpoint_{epoch}.pt"))
    print(f"epoch={epoch} loss={train_loss/num_batches}")



    










if __name__ == "__main__":
    print(__file__)

    x = torch.randn(size=(10, 1))
    print(x)
    print(x.mean(dim=0, keepdim=True))