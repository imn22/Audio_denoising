import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleConvLayer= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,9), padding= (1,4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,9), padding= (1,4)),
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
        self.maxpool= nn.MaxPool2d(kernel_size= (1,2), stride=2)
    def forward(self, x):
        cov= self.doubleConvLayer(x)
        out= self.maxpool(cov)
        return  cov, out
    
    
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
    
class My_unet_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # down sample layer
        self.downconv1= DownSample(in_channels, 64)
        self.downconv2= DownSample(64, 128)
        self.downconv3= DownSample(128, 256)
        self.mod= DownSample(256, 512)
        # self.conv4= mod.doubleConvLayer()
        self.down4= nn.AvgPool2d(kernel_size= (1,2), stride=2)
                                       
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
        cov4 = self.mod.doubleConvLayer(down3)
        down4= self.down4(cov4)
    
        bottleneck= self.bottel(down4)

        up1= self.upconv1(bottleneck, cov4)
        up2= self.upconv2(up1, cov3)
        up3= self.upconv3(up2, cov2)
        up4= self.upconv4(up3, cov1)

        out =self.out(up4)
        return out
