#inspiré from :: put ref

import torch 
from torch import nn
from model_utils import DoubleConv, DownSample, UpSample

class My_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # down sample layer
        self.downconv1= DownSample(in_channels, 64)
        self.downconv2= DownSample(64, 128)
        self.downconv3= DownSample(128, 256)
        self.downconv4= DownSample(256, 512)
        #conv layers in botelneck
        self.bottel= DoubleConv(512 , 1024)
        #upsample layers
        self.upconv1= UpSample(1024, 512)
        self.upconv2= UpSample(512, 256)
        self.upconv3= UpSample(256, 128)
        self.upconv4= UpSample(128, 64)
        #last conv layer
        self.out = nn.Conv2d(in_channels= 64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        cov1 , down1= self.downconv1(x)
        cov2 , down2= self.downconv2(down1)
        cov3 , down3= self.downconv3(down2)
        cov4 , down4= self.downconv4(down3)
    
        bottleneck= self.bottel(down4)

        up1= self.upconv1(bottleneck, cov4)
        up2= self.upconv2(up1, cov3)
        up3= self.upconv3(up2, cov2)
        up4= self.upconv4(up3, cov1)

        out =self.out(up4)
        return out

            