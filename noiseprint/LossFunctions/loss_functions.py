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
    
    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor, psd_flag:bool=True, epoch=1):
        b, c, _, _ = embeddings.shape
        loss = 0.0
        if psd_flag:
            loss += (-self.lamda * epoch * self.psd(embeddings=embeddings))
        embeddings = embeddings.view(b, -1)
        labels = labels.squeeze()
        num_lbls = labels.size()[0]
        distance_matrix = torch.cdist(x1=embeddings, x2=embeddings, p=2)
        distance_matrix = distance_matrix.flatten()[1:].view(num_lbls-1, num_lbls+1)[:,:-1].reshape(num_lbls, num_lbls-1)
        distance_sm = torch.softmax(input=-distance_matrix, dim=1)
        
        for i in range(num_lbls):
            lbl = labels[i]
            distance_sm_lbl = distance_sm[i]
            indices = torch.cat((labels[:i], labels[i+1:]), dim=0)
            indices_ind = indices==lbl
            probs = torch.sum(distance_sm_lbl[indices_ind])
            loss += -torch.log(probs)/200.0
        
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