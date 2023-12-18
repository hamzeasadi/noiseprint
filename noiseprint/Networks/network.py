"""
docs
"""

import sys
sys.path.append("../../..")

import torch
from torch import nn

from noiseprint.noiseprint.Networks.utils import FirstBloack
from noiseprint.noiseprint.Networks.utils import ConvLayer
from noiseprint.noiseprint.Networks.utils import MidBlock




class Noiseprint(nn.Module):
    """docs"""
    def __init__(self, input_ch, output_ch, num_layer:int=15) -> None:
        super().__init__()
        self.fblk = FirstBloack(inch=input_ch)
        self.mblk = MidBlock(ch=64, num_layers=num_layer)
        self.hblk = ConvLayer(inch=64, outch=output_ch, bn=False, act="None")
    

    def forward(self, x):
        x = self.fblk(x)
        x = self.mblk(x)
        x = self.hblk(x)

        return x
    




class Disc(nn.Module):

    def __init__(self, inch:int) -> None:
        super().__init__()
        
        layer0 = ConvLayer(inch=1, outch=16, stride=2, act="leaky", padding=0)
        layer1 = ConvLayer(inch=16, outch=32, stride=2, act="leaky", padding=0)
        layer2 = ConvLayer(inch=32, outch=48, stride=2, act="leaky", padding=0)
        layer3 = ConvLayer(inch=48, outch=64, stride=2, act="leaky", padding=0)
        self.bb = nn.Sequential(*[layer0, layer1, layer2, layer3])
        d_size = (3**2)*64
        self.fc = nn.Linear(in_features=d_size, out_features=1)
        self.flatten = nn.Flatten()
    def forward(self, x):

        x = self.bb(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    








if __name__ == "__main__":
    print(__file__)

    model = Noiseprint(input_ch=3, output_ch=1)
    x = torch.randn(size=(1,3,64,64))

    out = model(x)
    print(model)
    print(out.shape)