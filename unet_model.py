import torch
import torch.nn as nn
import torchonn as onn
from torchonn.models import ONNBaseModel
import torch.nn.functional as F
from torch.types import Device
from pyutils.general import logger
import numpy as np



class u_net(ONNBaseModel):
    """
    Implementation of U-net that operates in the Fourier domain. https://arxiv.org/pdf/1505.04597
    This version is currently simplified and contains only 3 levels instead of the 5 shown in the original U-net architecture.
    """
    def __init__(self,
                 device=torch.device("cpu"),
                 imChannels: int = 1,
                 imSize: int = 70,
                 numClasses = 2,
                 img_target = 30
                 ):
        """
        
        Inputs:
            imSize: input size of image
            img_target: size of target image
            numClasses: number of expected output channels
        """
        super().__init__()

        self.imSize = imSize
        self.numClasses = numClasses

        # contractive path
        # first set
        self.img_target = img_target
        self.poolSize = self.imSize-2
        self.convd11 = onn.layers.FourierConv2d(
            in_channels=imChannels,
            out_channels=64,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize2 = self.poolSize- 2
        self.convd12 = onn.layers.FourierConv2d(
            in_channels=64,
            out_channels=64,
            groups = 16,
            pool_size=self.poolSize2,
            bias=True,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )

        # second set
        self.poolSize = self.poolSize//2
        self.poolSize -= 2
        self.convd21 = onn.layers.FourierConv2d(
            in_channels=64,
            out_channels=128,
            groups = 16,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False, 
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.convd22 = onn.layers.FourierConv2d(
            in_channels=128,
            out_channels=128,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            groups = 16,
            sum_channels=False, 
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )

        # third set
        self.poolSize = self.poolSize//2
        self.poolSize -= 2
        self.convd31 = onn.layers.FourierConv2d(
            in_channels=128,
            out_channels=256,
            groups = 16,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.convd32 = onn.layers.FourierConv2d(
            in_channels=256,
            out_channels=256,
            groups = 16,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )

        
        #expanding path
        #first
        #upconv (performed on previous output)
        self.pad_layer1 = nn.ZeroPad2d(self.poolSize) 
        self.poolSize *= 2
        self.conveu1 = onn.layers.FourierConv2d(
            in_channels=256, #1024,
            out_channels=128, #512,
            groups = 16,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.conve11 = onn.layers.FourierConv2d(
            in_channels=256, #1024,
            out_channels=128, #512,
            pool_size=self.poolSize,
            bias=True,
            groups = 32,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.conve12 = onn.layers.FourierConv2d(
            in_channels=128,#512,
            out_channels=128, #128,#512,
            pool_size=self.poolSize,
            groups = 32,
            bias=True,
            miniblock=8,
            sum_channels=False, 
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )

        #second
        #upconv (performed on previous output)
        self.pad_layer2 = nn.ZeroPad2d(self.poolSize) 
        self.poolSize *= 2
        self.conveu2 = onn.layers.FourierConv2d(
            in_channels=128, #128, #512,
            out_channels=64,#64, #256,
            pool_size=self.poolSize,
            bias=True,
            #groups = 8,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.conve21 = onn.layers.FourierConv2d(
            in_channels=128,#128, #512,
            out_channels=64, #256,
            pool_size=self.poolSize,
            bias=True,
            groups = 16,
            miniblock=8,
            sum_channels=False, 
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.conve22 = onn.layers.FourierConv2d(
            in_channels=64, #256,
            out_channels=64, #256,
            pool_size=self.poolSize,
            bias=True,
            groups = 16,
            miniblock=8,
            sum_channels=False,  
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        
        #final
        self.convclass = onn.layers.FourierConv2d(
            in_channels=64,
            out_channels=self.numClasses,
            pool_size=self.img_target, 
            bias=True,
            miniblock=8,
            #groups =2,
            sum_channels=False, 
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )



        # Reset parameters
        self.convd11.reset_parameters()
        self.convd12.reset_parameters()
        self.convd21.reset_parameters()
        self.convd22.reset_parameters()
        self.convd31.reset_parameters()
        self.convd32.reset_parameters()

        self.conve11.reset_parameters()
        self.conve12.reset_parameters()
        self.conve21.reset_parameters()
        self.conve22.reset_parameters()

        self.conveu1.reset_parameters()
        self.conveu2.reset_parameters()

        self.convclass.reset_parameters()
    
    # Swish activation function
    def swish(self, x):
        return x*torch.sigmoid(x)


    def center_crop_to(self, tensor, target):
        _, _, h, w = tensor.shape
        _, _, th, tw = target.shape
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return tensor[:, :, start_h:start_h+th, start_w:start_w+tw]


    def forward(self, x):
        xd1 = self.swish(self.convd12(self.swish(self.convd11(x))))
        xd2 = self.swish(self.convd22(self.swish(self.convd21(xd1))))
        xd3 = self.swish(self.convd32(self.swish(self.convd31(xd2))))
        
        xd5p = self.pad_layer1(xd3)
        xe1 = self.swish(self.conveu1(xd5p))
        xd2 = self.center_crop_to(xd2, xe1)
        xe1c = torch.cat([xe1, xd2], dim=1)
        xe1f = self.swish(self.conve12(self.swish(self.conve11(xe1c))))

        xe1p = self.pad_layer2(xe1f)
        xe2 = self.swish(self.conveu2(xe1p))
        xd1 = self.center_crop_to(xd1, xe2)
        xe2c = torch.cat([xe2, xd1], dim=1)
        xe2f = self.swish(self.conve22(self.swish(self.conve21(xe2c))))
        
        x_final = self.convclass(xe2f)
        return torch.square(x_final.real) + torch.square(x_final.imag)
    
