"""
loss functions
"""
from typing import List
import math

import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.nn import functional as F




        


class NP_Loss(nn.Module):
    """
    noiseprint loss implementation
    """
    def __init__(self, lamda:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lamda = lamda
    
    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor):
        loss = (-self.lamda * self.psd(embeddings=embeddings))
        loss += self.prnu_loss(embeddings=embeddings, labels=labels)
        return loss
    

    def prnu_loss(self, embeddings:torch.Tensor, labels:torch.Tensor):
        """
        """
        b, c, h, w = embeddings.shape
        embeddings = embeddings.view(b, -1)
        labels = labels.squeeze().type(torch.float32)
        x_dist = torch.cdist(x1=embeddings, x2=embeddings)
        x_dist = x_dist.flatten()[1:].view(b-1, b+1)[:,:-1].reshape(b, b-1)
        x_dist = torch.softmax(-x_dist, dim=1)

        y_dist = torch.cdist(x1=labels.view(-1, 1), x2=labels.view(-1, 1))
        y_dist = y_dist.flatten()[1:].view(b-1, b+1)[:,:-1].reshape(b, b-1)
        y_dist = y_dist==0
        dist = x_dist*y_dist.int()
        dist = torch.sum(dist, dim=1)
        dist = torch.mean(torch.clamp(-torch.log(dist+1e-6), min=0, max=100.0))
        return dist
    

    def psd(self, embeddings:torch.Tensor):
        """
        docs
        """
        x = embeddings.squeeze()
        b, h, w = x.shape
        k = h*w
        dft = torch.fft.fft2(x)
        avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
        loss_psd = torch.clamp((1/k)*torch.sum(torch.log(avgpsd)) - torch.log((1/k)*torch.sum(avgpsd)), min=0.0, max=100.0)
        return loss_psd
    









def prnu_loss(embeddings:torch.Tensor, labels:torch.Tensor):
    """
    """
    b, c, h, w = embeddings.shape
    embeddings = embeddings.view(b, -1)
    labels = labels.squeeze().type(torch.float32)
    x_dist = torch.cdist(x1=embeddings, x2=embeddings)
    x_dist = x_dist.flatten()[1:].view(b-1, b+1)[:,:-1].reshape(b, b-1)
    x_dist = torch.softmax(-x_dist, dim=1)

    y_dist = torch.cdist(x1=labels.view(-1, 1), x2=labels.view(-1, 1))
    y_dist = y_dist.flatten()[1:].view(6-1, 6+1)[:,:-1].reshape(6, 6-1)
    y_dist = y_dist==0
    dist = x_dist*y_dist.int()
    dist = torch.sum(dist, dim=1)
    dist = torch.mean(torch.clamp(-torch.log(dist), min=0, max=100.0))
    return dist




    








def main():
    """"
    docs
    """
   
    # x = torch.tensor([
    #     [1,2,3], [2,2,3],[3,2,3],
    #     [60,70,80],[61,68,81],[60,75,85]
    # ], dtype=torch.float32)

    # y = torch.tensor([0,0,0,2,0,2], dtype=torch.float32)
    
  
    # x = torch.randn(size=(2, 2))
    # ones = torch.ones(size=(1, 3, 2,2))
    # print(x)
    # print(x*ones)
    # x, y = get_noise(batch_size=4)
    
    x = torch.randn(size=(4, 3, 2, 2))
    y = torch.randn(size=(4, 1, 2, 2))
    yhat = (x+y) - x

    crt = torch.nn.MSELoss()
    loss = crt(x, yhat)
    print(loss)
 

   





if __name__ == "__main__":

    main()