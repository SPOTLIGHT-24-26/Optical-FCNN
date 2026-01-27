import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torch.types import Device

from pyutils.general import logger

class MnistData():
    def __init__(self,
                 dataName: str = 'mnist',
                 mode: str = 'spatial',
                 batchSize: int = 64,
                 numComponents: list = None
                 ):
        '''
        Although the name is 'MnistData', this class loads. MNIST, FashionMNIST and CIFAR10 data in grayscale and color.
        The class also supports fourier transform of the data and sub-smapling Fourier frequency components.
        Inputs -
            dataName        str, data to load. MNIST or FMNIST or CIFAR
            mode            str, load data in spatial or fourier format
            batchSize       int, batch size
            numCompoenents  int, number of fourier components (applies only for Fourier mode)
        Outputs -
            trainLoader DataLoader, training data loader
            testLoader  DataLoader, test data loader
        '''
        # assert data selection is valid
        assert dataName in {'mnist', 'fmnist', 'cifar10Color', 'cifar10BW'}, logger.error(
            f"data not supported. Expected one from (mnist, fmnist, cifar10, cifar10BW) but got {dataName}."
        )
        # assert mode is valid
        assert mode in {'spatial', 'fourier', 'fftLin', 'rfft'}, logger.error(
            f"Mode not supported. Expected one from (spatial, fourier, fftLin, rfft) but got {mode}."
        )
        # assert if mode=fourier, numcomponents is not none
        if mode == 'fourier' or mode == 'rfft':
            assert numComponents is not None, logger.error(
                f"fourier mode selected, but numcomponents is not passed. Please pass \
                original image size as numComponenets if all fourier componenets are to be considered."
            )

        self.batchSize = batchSize
        self.numComponents = numComponents

        # Setup data and target transforms
        transformsList = []
        transformsList.append(transforms.ToImage())
        transformsList.append(transforms.ToDtype(torch.float32, scale=True))
        if dataName == 'mnist':
            transformsList.append(transforms.Normalize((0.1307,), (0.3081,)))
        elif dataName == 'fmnist':
            transformsList.append(transforms.Normalize((0.2860), (0.3526)))
        elif dataName == 'cifar10Color':
            transformsList.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
        elif dataName == 'cifar10BW':
            # Grayscale scales the images to [0,1]
            # For the transformed dataset:
            # mean = 0.4808, std = 0.2385
            transformsList.append(transforms.Grayscale())
            #transformsList.append(transforms.Normalize(mean=[0.4808], std=[0.2385]))
        else:
            # default case, normalize with mean=0.5 and std=0.5
            transformsList.append(transforms.Normalize((0.5), (0.5)))
        
        if mode == 'fftLin':
            transformsList.append(transforms.Lambda(lambda x: torch.flatten(x)))
        
        if mode == 'fourier':
            transformsList.append(
                transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2,-1)))
            )
            transformsList.append(transforms.CenterCrop(self.numComponents))
        if mode == 'rfft':
            transformsList.append(
                transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm="ortho"), dim=(-2,-1)))
            )
            transformsList.append(transforms.CenterCrop(self.numComponents))

        self.dataTransform = transforms.Compose(transformsList)
        self.targetTransform = transforms.Lambda(
            lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
        )

        trainDataArgs = {'root':'data', 'train': True, 'download': False,
                         'transform': self.dataTransform, 'target_transform': self.targetTransform}
        testDataArgs = {'root':'data', 'train': False, 'download': False,
                         'transform': self.dataTransform, 'target_transform': self.targetTransform}
        self.trainData = None
        self.testData = None
        # Setup train and test data
        if dataName == 'mnist':
            self.trainData = MNIST(**trainDataArgs)
            self.testData = MNIST(**testDataArgs)
        elif dataName == 'fmnist':
            self.trainData = FashionMNIST(**trainDataArgs)
            self.testData = FashionMNIST(**testDataArgs)
        elif dataName in {'cifar10Color', 'cifar10BW'}:
            self.trainData = CIFAR10(**trainDataArgs)
            self.testData = CIFAR10(**testDataArgs)

        # setup data loaders
        self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                        shuffle=True, num_workers=4)
        self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                        shuffle=False, num_workers=4)
        #raise NotImplementedError("Only spatial, fourier, and fftLin modes supported")
    
    def getDataLoaders(self):
        return self.trainLoader, self.testLoader