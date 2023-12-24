"""
general layers
"""

import sys
sys.path.append("../../..")

import torch
from torch import nn








class ConstConv(nn.Module):
    """
    const conv
    """
    def __init__(self, inch:int, outch:int, ks:int, stride:int, dev:torch.device, num_samples:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dev = dev
        self.ks = ks
        self.inch = inch
        self.num_samples = num_samples
        self.conv0 = nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=ks, stride=stride, padding=0, bias=False)
        self.padding = nn.ZeroPad2d(1)
        self.bn = nn.BatchNorm2d(num_features=outch)
        self.act = nn.ReLU()
        self.data = self._get_data()

    def forward(self, x):
        if self.training:
            out0 = self.conv0(self.data['zero'])
            out1 = self.conv0(self.data['one'])
            out = self.padding(x)
            out = self.conv0(out)
            out = self.act(out)
            return dict(out=out, out1=out1, out0=out0)
        
        x = self.padding(x)
        x = self.conv0(x)
        x = self.act(x)
        return dict(out=x)

    

    def _get_data(self):
        ones = torch.ones(size=(self.num_samples, self.inch, self.ks, self.ks), dtype=torch.float32)
        ones[:, :, self.ks//2, self.ks//2] = 0.0
        zeros = torch.zeros_like(ones)
        zeros[:, :, self.ks//2, self.ks//2] = 1.0

        return dict(one=ones.to(self.dev), zero=zeros.to(self.dev))



    



class ConvLayer(nn.Module):
    """
    conv layer
    """
    def __init__(self, inch:int=64, outch:int=64, ks:int=3, stride:int=1, 
                 bias:bool=True, padding:int=1, bn:bool=True, act:str="relu") -> None:
        super().__init__()
        modulelist = []

     
        pad = nn.ZeroPad2d(padding)
        conv = nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=ks, stride=stride, bias=bias, padding=0)
        relu = nn.ReLU(inplace=True)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        relu6 = nn.ReLU6(inplace=True)
        prelu = nn.PReLU()
        # momentum=0.9, eps=1e-04, affine=True
        batch_norm = nn.BatchNorm2d(num_features=outch, momentum=0.9, eps=1e-4, affine=True)

        if padding!=0:
            modulelist.append(pad)
        modulelist.append(conv)

        if bn:
            modulelist.append(batch_norm)
        
        if act=='relu':
            modulelist.append(relu)
        elif act == "leaky":
            modulelist.append(lrelu)
        elif act == "relu6":
            modulelist.append(relu6)
        elif act == "prelu":
            modulelist.append(prelu)

        self.blk = nn.Sequential(*modulelist)

    def forward(self, x):
        return self.blk(x)





class FirstBloack(nn.Module):

    def __init__(self, inch) -> None:
        super().__init__()
        self.blk = ConvLayer(inch=inch, outch=64, padding=1, bn=False, act="relu")
    
    def forward(self, x):
        return self.blk(x)




class MidBlock(nn.Module):
    """docs"""
    def __init__(self, ch:int, num_layers:int) -> None:
        super().__init__()
        self.ch = ch
        self.num_layers = num_layers
        self.blk = self._build_blk()


    def _build_blk(self):
        module_list = []
        for i in range(self.num_layers):
            layer = ConvLayer(inch=self.ch, outch=self.ch, act="relu")
            module_list.append(layer)
        return nn.Sequential(*module_list)

    def forward(self, x):
        return self.blk(x)







if __name__ == "__main__":

    net = ConstConv(inch=3, outch=3, ks=3, stride=1, dev="cpu", num_samples=2)

    data = net.data

    print(data)
