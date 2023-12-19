"""
loss functions
"""

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
    
    def forward(self, embeddings:torch.Tensor):

        loss = (-self.lamda * self.psd(embeddings=embeddings))
        return loss
    
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






def psd(x:torch.Tensor):
    """
    
    """
    
    x = 1





    





def main():
    """"
    docs
    """
   

    




   





if __name__ == "__main__":

    main()