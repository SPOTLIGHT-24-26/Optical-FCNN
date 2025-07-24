import torch
import torch.nn as nn
import torchonn as onn
from torchonn.models import ONNBaseModel
import torch.nn.functional as F
from torch.types import Device
from pyutils.general import logger

import numpy as np
import time

class simpleCNN(ONNBaseModel):
    def __init__(self,
                 device=torch.device("cuda"),
                 imChannels: int = 1,
                 imSize: int = 28):
        super().__init__()
        '''
        Create a simple CNN model to test on MNIST and CIFAR10 datasets
        Inputs
            device:     device, device to create the model on
            imChannels: int, number of channels in input images
            imSize:     int, size of the images (images assumed to be square) e.g., 28, 32
        '''

        self.conv1 = onn.layers.MZIConv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            device=device
        )
        self.conv2 = onn.layers.MZIConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            device=device
        )
        self.pool = nn.MaxPool2d(2)
        # Each conv layer reduce the image size by 2 and pool layers cuts it in half
        denseLayerDims = (imSize - 4)//2
        self.linear1 = onn.layers.MZIBlockLinear(
            in_features=64*denseLayerDims*denseLayerDims,
            out_features=128,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            dtype=torch.float,
            device=device,
        )
        self.linear2 = onn.layers.MZIBlockLinear(
            in_features=128,
            out_features=10,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            dtype=torch.float,
            device=device,
        )
        # Reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    # Swish activation function
    def swish(self, x):
        return x*torch.sigmoid(x)
    
    def forward(self, x):
        #x= torch.relu(self.conv1(x))
        #x=torch.relu(self.conv2(x))
        x = self.swish(self.conv1(x))
        x = self.swish(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        #x = torch.relu(self.linear1(x))
        x = self.swish(self.linear1(x))
        x = self.linear2(x)
        return x
    
class simpleFCNN(ONNBaseModel):
    def __init__(self,
                 device=torch.device("cpu"),
                 imChannels: int = 1,
                 imSize: int = 28
                 ):
        super().__init__()

        self.g_taper = 1.0
        self.g = 1.75 * np.pi
        self.phi_b = np.pi

        self.imSize = imSize

        # for sptial CNNs, image dimension is reduced by 2 after each convolution layer
        # Set pool size to the same size as in spatial CNNs
        self.poolSize = self.imSize-2
        self.conv1 = onn.layers.FourierConv2d(
            in_channels=imChannels,
            out_channels=32,
            pool_size=self.poolSize,
            bias=True,
            miniblock=8,
            sum_channels=False,  # sum all input channels
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.denseLayerDims = self.poolSize//2
        self.conv2 = onn.layers.FourierConv2d(
            in_channels=32,
            out_channels=64,
            pool_size=self.denseLayerDims,
            bias=True,
            miniblock=8,
            sum_channels=False,  # sum all input channels
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.linear1 = onn.layers.MZIBlockLinear(
            in_features=64*self.denseLayerDims*self.denseLayerDims,
            out_features=128,
            bias=True,
            miniblock=8,
            mode="usv",
            decompose_alg="clements",
            dtype=torch.cfloat,
            photodetect=False,
            device=device,
        )
        self.linear2 = onn.layers.MZIBlockLinear(
            in_features=128,
            out_features=10,
            bias=True,
            miniblock=8,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            dtype=torch.cfloat,
            device=device,
        )
        self.EOActivation = onn.layers.ElectroOptic(
            in_features = 25,   # needed only when using bias
            bias = False,
            alpha = 0.1,
            g = self.g * (self.g_taper ** 0), # here, 0 -> i, which in neuroptica usage increments with layers
            phi_b = self.phi_b,
            device=device
        )
        # Reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
    
    def swish(self, x):
        return x*torch.sigmoid(x)

    def forward(self, x):
        # EOActivation has discrepancies near 0. Swish helps in better training
        #x = self.EOActivation(self.conv1(x))
        #x = self.EOActivation(self.conv2(x))
        x = self.swish(self.conv1(x))
        # Combiner network simulation to sum all input channels
        #x = torch.sum(x, 1, keepdim=True)
        x = self.swish(self.conv2(x))
        x = x.flatten(1)
        #x = self.EOActivation(self.linear1(x))
        x = self.swish(self.linear1(x))
        x = self.linear2(x)
        x = torch.square(x.real) + torch.square(x.imag)
        return x

class fftLinear(ONNBaseModel):

    __constants__ = [
        "in_features",
        "layer1_out"
    ]

    in_features: int
    layer1_out: int

    def __init__(self,
                 in_features: int,
                 layer1_out_features: int,
                 miniblock: int = 4,
                 device: Device = torch.device("cpu")
                 ):
        super().__init__()

        self.in_features = in_features
        self.device = device

        self.layer1_out_features = layer1_out_features
        self.miniblock = miniblock

        self.lin1 = onn.layers.FFTONNBlockLinear(
            in_features = in_features,
            out_features = self.layer1_out_features,
            miniblock = self.miniblock,
            device = self.device
        )
        self.lin2 = onn.layers.FFTONNBlockLinear(
            in_features = self.layer1_out_features,
            out_features = 10,
            miniblock = miniblock,
            device = self.device
        )

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def swish(self, x):
        return x*torch.sigmoid(x)
    
    def forward(self, x):
        x = self.swish(self.lin1(x))
        x = self.lin2(x)
        return x
    
class FFTConv(ONNBaseModel):
    def __init__(self,
                 imChannels: int = 1,
                 imSize: int = 28,
                 miniblock: int = 4,
                 device=torch.device("cpu")):
        super().__init__()
        '''
        CNN model with fft convolution layers proposed in
        "Gu, Jiaqi, et al. "Toward hardware-efficient optical neural networks: Beyond FFT architecture 
        via joint learnability." IEEE Transactions on Computer-Aided Design of Integrated Circuits and 
        Systems 40.9 (2020): 1796-1809."
        Inputs:
            imChannels: int, number of channels in input image
            imSize: int, 2D size of input images (input images assumed to be square)
            miniblock: int, size of miniblock - choose from (1,2,4,8)
            device: torch.Device, cpu or cuda
        '''

        self.conv1 = onn.layers.FFTONNBlockConv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            bias = True,
            miniblock = miniblock,
            photodetect = False,
            device = device
        )
        self.conv2 = onn.layers.FFTONNBlockConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            bias = True,
            miniblock = miniblock,
            photodetect = False,
            device = device
        )

        # Calculate dense layer dimensions
        self.pool = nn.MaxPool2d(2)
        linLayerIn = (imSize - 4)//2

        self.linear1 = onn.layers.MZIBlockLinear(
            in_features=64*linLayerIn*linLayerIn,
            out_features=128,
            bias=True,
            miniblock=miniblock,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            dtype=torch.float,
            device=device,
        )
        self.linear2 = onn.layers.MZIBlockLinear(
            in_features=128,
            out_features=10,
            bias=True,
            miniblock=miniblock,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            dtype=torch.float,
            device=device,
        )
        # reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    # Swish activation function
    def swish(self, x):
        return x*torch.sigmoid(x)
    
    def forward(self, x):
        x = self.swish(self.conv1(x))
        x = self.swish(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.swish(self.linear1(x))
        x = self.linear2(x)
        return x

class simpleDCNN(nn.Module):
    def __init__(self,
                 device = torch.device("cpu"),
                 imChannels: int = 1,
                 imSize: int = 28):
        super(simpleDCNN, self).__init__()

        self.fcTime = 0

        self.conv1 = nn.Conv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            device=device
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            device=device
        )
        self.pool = nn.MaxPool2d(2)
        denseLayerDims = (imSize - 4)//2
        self.linear1 = nn.Linear(
            in_features=64*denseLayerDims*denseLayerDims,
            out_features=128,
            bias=True,
            device=device,
        )
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=10,
            bias=True,
            device=device,
        )

    # Swish activation function
    def swish(self, x):
        return x*torch.sigmoid(x)
    
    def getFCTime(self):
        return self.fcTime

    def forward(self, x):
        # conv layers
        # for i in range(self.num_layers):
        #     x = self.swish(self.convs[i](x))
        x = self.swish(self.conv1(x))
        x = self.swish(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        # FC layers
        #torch.cuda.synchronize()
        #t0 = time.time()
        x = self.swish(self.linear1(x))
        x = torch.square(torch.abs(self.linear2(x)))
        #torch.cuda.synchronize()
        #t1 = time.time()
        #self.fcTime = t1-t0
        return x