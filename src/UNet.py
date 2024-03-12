import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pooling = F.max_pool2d
        self.pool_params = {"kernel_size":2, "stride":2}
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # decoder (upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3,
                      padding = 1)
        )

    def forward(self, x):
        # encoder                                                  (size, features)

        e0 = self.enc_conv0(x) #(256, 32)
        e1 = self.enc_conv1(self.pooling(e0, **self.pool_params)) #(128, 64)
        e2 = self.enc_conv2(self.pooling(e1, **self.pool_params)) #(64, 128)
        e3 = self.enc_conv3(self.pooling(e2, **self.pool_params)) #(32, 256)

        b = self.bottleneck_conv(self.pooling(e3, **self.pool_params)) #(16, 256)


        d0 = self.dec_conv0(torch.cat((self.upsample(b), e3), 1)) #(32, 128)
        d1 = self.dec_conv1(torch.cat((self.upsample(d0), e2), 1)) #(64, 64)
        d2 = self.dec_conv2(torch.cat((self.upsample(d1), e1), 1)) #(128, 32)
        d3 = self.dec_conv3(torch.cat((self.upsample(d2), e0), 1)) #(256, 1)

        return d3