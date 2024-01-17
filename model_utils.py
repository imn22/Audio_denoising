import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleConvLayer= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out= self.doubleConvLayer(x)
        return out

class DownSample(nn.Module):
    """ if input is [b_size, in_channels, h, w] returns out of size [b_size, out_channel, h/2, w/2]"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleConvLayer= DoubleConv(in_channels, out_channels)
        self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        cov= self.doubleConvLayer(x)
        out= self.maxpool(cov)
        return cov, out
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample= nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.doubleConvLayer= DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1= self.upsample(x1)
        x= torch.cat([x1, x2],1) #concat the output of the downsampler
        x=self.doubleConvLayer(x)
        return x