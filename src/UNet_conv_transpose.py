import torch
from torch import nn
import torch.nn.functional as F

class UNet_conv_transpose(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()

        # encoder (downsampling)
        # Each enc_conv/dec_conv block should look like this:
        # nn.Sequential(
        #     nn.Conv2d(...),
        #     ... (2 or 3 conv layers with relu and batchnorm),
        # )
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.downsamp0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2,
                      stride = 2)
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
        self.downsamp1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2,
                                   stride=2)
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
        self.downsamp2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2,
                      stride = 2)
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
        self.downsamp3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2,
                      stride = 2)

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
        self.upsample0 = nn.ConvTranspose2d(256, out_channels = 256, kernel_size=2,
                                        stride=2)
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
        self.upsample1 = nn.ConvTranspose2d(128, out_channels = 128, kernel_size=2,
                                        stride=2)
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
        self.upsample2 = nn.ConvTranspose2d(64, out_channels = 64, kernel_size=2,
                                        stride=2)
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
        self.upsample3 = nn.ConvTranspose2d(32, out_channels = 32, kernel_size=2,
                                        stride=2)
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
        e1 = self.enc_conv1(self.downsamp0(e0)) #(128, 64)
        e2 = self.enc_conv2(self.downsamp1(e1)) #(64, 128)
        e3 = self.enc_conv3(self.downsamp2(e2)) #(32, 256)

        b = self.bottleneck_conv(self.downsamp3(e3)) #(16, 256)


        d0 = self.dec_conv0(torch.cat((self.upsample0(b), e3), 1)) #(32, 128)
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0), e2), 1)) #(64, 64)
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1), e1), 1)) #(128, 32)
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2), e0), 1)) #(256, 1)

        return d3