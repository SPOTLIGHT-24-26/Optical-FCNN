import torch

import models

import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# Create a random image
sizes = [28,14,16,28,32,64,32]
K = [3,3,3,3,3,3,3]
numConv = [2,2,2,2,2,5,2]
# Pass 1 as frist channel, as we are working with only grayscale images
c = [[1,32,64], [1,32,64], [1,32,64], [1,32,64], [1,32,64],
      [1,32,64,128,256,512], [1,32,64]]
timeHybrid = []
timeGPU = []
for i in range(7):
    # get current conv_layer parameters
    s = sizes[i]
    k = K[i]
    nC = numConv[i]
    channels = c[i]
    print('processing s = {}'.format(s))
    im = torch.rand(1,1,s,s,device=device)
    # digital-electric and hybrid models
    model1 = models.simpleDCNN(device, s, k, nC, channels)
    # Fourier transform and time for fc layers
    tHybrid = 0
    torch.cuda.synchronize()
    tFourier0 = time.time()
    fftIm = torch.fft.fftshift(torch.fft.fft2(im, norm="forward"), dim=(-2,-1))
    torch.cuda.synchronize()
    tFourier1 = time.time()
    tHybrid += tFourier1 - tFourier0
    print('time for FFT = {}'.format(tHybrid))
    # forward pass - digital
    torch.cuda.synchronize()
    tDig0 = time.time()
    out1 = model1(im)
    torch.cuda.synchronize()
    tDig1 = time.time()
    tGPU = tDig1 - tDig0
    timeGPU.append(tGPU)
    print('time for electric = {}'.format(tGPU))
    fcTime = model1.getFCTime()
    tHybrid += fcTime
    timeHybrid.append(tHybrid)
    print('time for conv = {}'.format(tGPU - fcTime))
    print('time for hybrid = {}'.format(tHybrid))

# plot times
print('hybrid times: {}'.format(timeHybrid))
print('digital times: {}'.format(timeGPU))