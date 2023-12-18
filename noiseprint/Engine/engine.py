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


from noiseprint.noiseprint.Dataset.dataset import create_batch




def odd_eve_lbl(batch_size:int):
    odd = torch.arange(start=1, end=batch_size, step=2)
    even = torch.arange(start=0, end=batch_size, step=2)

    return odd, even





def init_train(gen:nn.Module, disc:nn.Module, gen_opt:Optimizer, disc_opt:Optimizer, gen_sch:nn.Module, paths,
               disc_sch:nn.Module, gen_crt:nn.Module, disc_crt:nn.Module, dataset:List, epoch:int, dev:torch.device):
    
    gen.to(dev)
    gen.train()
    disc.to(dev)
    disc.train()
    train_loss = 0
    batch_cnt = 0
    for bbatch in zip(*dataset):
        X, Y = create_batch(batch=bbatch)
        bs, _, _, _ = X.shape
        out = gen(X.to(dev))
        odd_idx, even_idx = odd_eve_lbl(batch_size=bs)
        fake = out[odd_idx].detach()
        real = out[even_idx].detach()

        disc_fake = disc(fake)
        disc_real = disc(real)

        disc_loss = disc_crt(disc_real - disc_fake, torch.ones_like(disc_real, requires_grad=False))
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()


        gen_loss = gen_crt(embeddings=out, labels=Y.to(dev), psd_flag=True, epoch=1)
        disc_real = disc(out[even_idx])
        disc_fake = disc(out[odd_idx])
        disc_loss = disc_crt(disc_fake - disc_real, torch.ones_like(disc_real, requires_grad=False))
        loss = disc_loss + gen_loss
        print(loss.item())
        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()

        train_loss += loss.item()
        batch_cnt += 1

    if gen_sch is not None:
        gen_sch.step()
    
    if disc_sch is not None:
        disc_sch.step()

    state = dict(model=gen.state_dict(), epoch=epoch, loss=train_loss/batch_cnt)

    print(f"epoch={epoch} loss={train_loss/batch_cnt}")

    torch.save(obj=state, f=os.path.join(paths.model, f"ckpoint_{epoch}.pt"))




def np_train(gen:nn.Module, gen_opt:Optimizer, gen_crt:nn.Module, dev:torch.device,
             gen_sch:nn.Module, dataloader:DataLoader, epoch:int, paths):
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
