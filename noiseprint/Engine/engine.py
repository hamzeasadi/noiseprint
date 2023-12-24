"""
training engine
"""

import os
import sys
sys.path.append("../../..")
from typing import List
import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# from noiseprint.noiseprint.Dataset.dataset import create_batch




def odd_eve_lbl(batch_size:int):
    odd = torch.arange(start=1, end=batch_size, step=2)
    even = torch.arange(start=0, end=batch_size, step=2)

    return odd, even




def get_noise(batch_size:int, in_shape:List=[1,3,64,64]):
    ones = torch.ones(size=in_shape)
    AGWN = []
    awgn_labels = []
    for i in range(batch_size//4):
        noise = torch.randn(size=[64, 64])
        for j in range(4):
            sigma = torch.randint(low=15, high=45, size=(1,))
            awgn = noise*sigma/255.0
            AGWN.append(awgn*ones)
            awgn_labels.append(awgn.unsqueeze(dim=0))
    
    return torch.cat(AGWN, dim=0), torch.cat(awgn_labels, dim=0)



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
    file_path = os.path.join(paths.report, "np_loss_log.txt")
    train_loss = 0
    num_batches = len(dataloader)
    gen.to(dev)
    gen.train()
    cntr = 0

    if epoch<5:
        for X, Y in dataloader:
            b,_,_,_ = X.shape
            XG, YG = get_noise(batch_size=b)
            Xhat = X+XG
            noise_k = gen(Xhat.to(dev))
            loss = disc_crt(noise_k,  YG.to(dev))
            disc_opt.zero_grad()
            loss.backward()
            disc_opt.step()
            train_loss += loss.item()
            if cntr%20 == 0:
                if cntr==0:
                    with open(file_path, "w") as log_file:
                        log_file.write(f"epoch={cntr} loss={train_loss/num_batches}\n")
                else:
                    with open(file_path, "a") as log_file:
                        log_file.write(f"epoch={cntr} loss={train_loss/num_batches}\n")
            cntr += 1

        if disc_sch is not None:
            disc_sch.step()

        state = dict(model=gen.eval().state_dict(), epoch=epoch, loss=train_loss/num_batches)
        torch.save(obj=state, f=os.path.join(paths.model, f"np_noise_ckpoint_{epoch}.pt"))

    else:
        for X, Y in dataloader:

            out = gen(X.to(dev))
            gen_loss = gen_crt(embeddings=out, labels=Y.to(dev))
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            train_loss += gen_loss.item()

            if cntr%20 == 0:
                if cntr==0:
                    with open(file_path, "w") as log_file:
                        log_file.write(f"epoch={cntr} loss={gen_loss.item()}\n")
                else:
                    with open(file_path, "a") as log_file:
                        log_file.write(f"epoch={cntr} loss={gen_loss.item()}\n")
            cntr += 1


        if gen_sch is not None:
            gen_sch.step()

        state = dict(model=gen.eval().state_dict(), epoch=epoch, loss=train_loss/num_batches)
        torch.save(obj=state, f=os.path.join(paths.model, f"np_ckpoint_{epoch}.pt"))

        thresh = int(math.sqrt(epoch))
        if thresh%2==0:
            for X, Y in dataloader:
                b,_,_,_ = X.shape
                XG, YG = get_noise(batch_size=b)
                Xhat = X+XG
                noise_k = gen(Xhat.to(dev))
                loss = disc_crt(noise_k,  YG.to(dev))
                disc_opt.zero_grad()
                loss.backward()
                disc_opt.step()
                train_loss += loss.item()

                if cntr%20 == 0:
                    if cntr==0:
                        with open(file_path, "w") as log_file:
                            log_file.write(f"epoch={cntr} loss={train_loss/num_batches}\n")
                    else:
                        with open(file_path, "a") as log_file:
                            log_file.write(f"epoch={cntr} loss={train_loss/num_batches}\n")
                cntr += 1

            if disc_sch is not None:
                disc_sch.step()
    
    



    










if __name__ == "__main__":
    print(__file__)

    x = torch.randn(size=(10, 1))
    print(x)
    print(x.mean(dim=0, keepdim=True))