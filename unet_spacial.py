import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop

class SpatialUNet(nn.Module):
    """
    Implementation of U-net that operates in the spatial domain. https://arxiv.org/pdf/1505.04597
    This version is currently simplified and contains only 3 levels instead of the 5 shown in the original U-net architecture.
    
    """
    def __init__(self, device=torch.device("cpu"), imChannels=1, imSize=28, numClasses=2, img_target = 30):
        """
        Inputs:
            imSize: input size of image
            img_target: size of target image
            numClasses: number of expected output channels
        """
        super().__init__()
        self.imSize = imSize
        self.numClasses = numClasses
        self.img_target = img_target

        # contractive path
        # first block
        self.conv11 = nn.Conv2d(imChannels, 64, kernel_size=3)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)

        # second block
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)

        # third block
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)

        # expanding
        # first up block
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conve11 = nn.Conv2d(256, 128, kernel_size=3)
        self.conve12 = nn.Conv2d(128, 128, kernel_size=3)

        # second up block
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conve21 = nn.Conv2d(128, 64, kernel_size=3)
        self.conve22 = nn.Conv2d(64, 64, kernel_size=3)

        # output layer
        self.final_conv = nn.Conv2d(64, numClasses, kernel_size=1)

    def forward(self, x):
        c11 = F.relu(self.conv11(x))
        c12 = F.relu(self.conv12(c11))
        p1 = self.pool1(c12)
        
        c21 = F.relu(self.conv21(p1))
        c22 = F.relu(self.conv22(c21))
        p2 = self.pool2(c22)

        c31 = F.relu(self.conv31(p2))
        c32 = F.relu(self.conv32(c31))

        u1 = self.upconv1(c32)
        u1 = torch.cat([u1, center_crop(c22, u1.shape[2:])], dim=1)
        e11 = F.relu(self.conve11(u1))
        e12 = F.relu(self.conve12(e11))

        u2 = self.upconv2(e12)
        u2 = torch.cat([u2, center_crop(c12, u2.shape[2:])], dim=1)
        e21 = F.relu(self.conve21(u2))
        e22 = F.relu(self.conve22(e21))

        out = self.final_conv(e22)
        return out
