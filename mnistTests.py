import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#from torcheval.metrics import PeakSignalNoiseRatio

import models
import data_loader

from datetime import datetime
import time
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
EPOCHS = 100
lr = 1e-4
moment = 0.99

best_vloss = 1_000_000.
best_vacc = 0.0
bestEpochNum = 0
timestamp = datetime.now().strftime('%m%d_%H%M')
#model_path = 'models/'
tLoss = []
vLoss = []
tAcc = []
vAcc = []
timeAccumulator = [0.]

# --------------------------Data Parameters--------------------------
#mode = 'fourier'
mode = 'spatial'
#mode = 'fftLin'
#Note : fftlin is similar to spatial/digital but flattens the images

""" dataName = 'cifar10Color'
imSize = 32
imChannels = 3 """
dataName = 'cifar10BW'
imSize = 32
imChannels = 1
""" dataName = 'mnist'
imSize = 14
imChannels = 1 """
""" dataName = 'fmnist'
imSize = 28
imChannels = 1 """

numComponents = 32

# Name says 'mnist', but can load cifar10 in BW, color, spatial, and Fourier modes and fashion-mnist in both spatial and Fourier modes
mnistData =  data_loader.MnistData(dataName, mode, batch_size, numComponents=numComponents)
trainLoader, testLoader = mnistData.getDataLoaders()

# Calculate dataset's mean and std
""" meanDatasetR, stdDatasetR = 0.0, 0.0
#meanDatasetI, stdDatasetI = 0.0, 0.0
for i, data in enumerate(trainLoader):
    input, _ = data
    meanDatasetR += torch.mean(input, dim=(0, 2, 3))
    #meanDatasetI += torch.mean(input.imag, dim=(0, 2, 3))
    stdDatasetR += torch.std(input, dim=(0,2,3))
    #stdDatasetI += torch.std(input.imag, dim=(0,2,3))
meanDatasetR /= len(trainLoader)
#meanDatasetI /= len(trainLoader)
stdDatasetR /= len(trainLoader)
#stdDatasetI /= len(trainLoader) """

#model = models.simpleCNN(device, imChannels, imSize)
#model = models.simpleFCNN(device, imChannels, imSize)
model = models.simpleDCNN(device, imChannels, imSize)
#model = models.fftLinear(in_features=imSize*imSize, layer1_out_features=1024, miniblock=8, device=device)
#model = models.FFTConv(imChannels=imChannels, imSize=imSize, miniblock=4, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

# load checkpoint
""" checkpoint = torch.load('models/cifar10BW_fourier_0606_1224_1e-05_32_98_chkPnt.tar', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss'] """

def train_one_epoch(epochIdx):
    running_loss = 0.
    last_loss = 0.
    running_acc = 0.
    last_acc = 0.
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        running_acc += sum(torch.argmax(outputs, 1) == torch.argmax(labels, 1))/batch_size
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            last_acc = running_acc / 100
            print('  batch {} loss: {} acc: {}'.format(i + 1, last_loss, last_acc))
            running_loss = 0.
            running_acc = 0.
    
    return last_loss, last_acc

bestModel = model.state_dict()
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    t0 = time.time()
    avg_loss, avg_acc = train_one_epoch(epoch)
    timeAccumulator.append(timeAccumulator[-1] + time.time() - t0)
    #scheduler.step()

    running_vloss = 0.0
    running_vacc = 0.0
    model.eval()

    with torch.no_grad():

        # sigma = np.linspace(5e-3, 5e-2, 10)
        # testAcc = []
        # w1 = torch.acos(model.conv1.weight)
        # w2 = torch.acos(model.conv2.weight)
        # for ind, s in enumerate(sigma):
        #     dTheta1 = np.random.normal(scale=s, size=w1.shape)
        #     dTheta2 = np.random.normal(scale=s, size=w2.shape)
        #     model.conv1.weight.data = torch.cos(w1 + torch.from_numpy(dTheta1.astype('float32')).to(device))
        #     model.conv2.weight.data = torch.cos(w2 + torch.from_numpy(dTheta2.astype('float32')).to(device))
        #     running_vacc = 0.0

        for i, vdata in enumerate(testLoader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()
            running_vacc += sum(torch.argmax(voutputs, 1) == torch.argmax(vlabels, 1))/batch_size
        #     testAcc.append((running_vacc / (i+1)).item())
        # accDict = {"acc" : testAcc}
        # with open('result_lists/fmnist_sig_28.json', "w") as outfile:
        #     json.dump(accDict, outfile, indent=2)
        # print("sigma tests done")
    
    avg_vloss = running_vloss / (i+1)
    avg_vacc = running_vacc / (i+1)
    tLoss.append(avg_loss)
    tAcc.append(avg_acc.item())
    vLoss.append(avg_vloss)
    vAcc.append(avg_vacc.item())
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('Accuracy train {} valid {}'.format(avg_acc, avg_vacc))
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_vacc = avg_vacc
        bestEpochNum = epoch + 1
        model_path = 'models/'+ dataName + '_' + mode + '_{}_{}_{}_{}'.format(timestamp, lr, numComponents, bestEpochNum)
        bestModel = model.state_dict()
    # stop training if loss is diverging
    # if epoch - bestEpochNum > 15:
    #     break

# Write the best model
bestPath = model_path + '.pt'
torch.save(bestModel, bestPath)
# Save the final model and optimizer dicts for checkpoints
checkPath = model_path + '_chkPnt.tar'
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tLoss[-1]
            },  checkPath)

# Print some statistics
print('Best vloss: {} at epoch {} with accuracy {}'.format(best_vloss, bestEpochNum, best_vacc))
print('Time, epoch: {} or: {}'.format(timeAccumulator[1], timeAccumulator[-1]/(EPOCHS+1)))

# To load saved model
# saved_model = models.simpleCNN(device)
# saved_model.load_state_dict(torch.load(PATH))

# write lists to files
file_path = 'result_lists/' + dataName + '_' + mode + '_{}_{}_{}_{}'.format(timestamp, lr, numComponents, bestEpochNum) + '.json'
out_dict = {"tLoss": tLoss, "vLoss": vLoss, "tAcc": tAcc, "vAcc": vAcc}
json.dump(out_dict,open(file_path,"w"), indent=2)
#out_dict = json.load(open(file_path,"r"))

# Plot training and validation losses and, training times
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(tLoss)
ax1.set_title("Train Loss")
ax2.plot(vLoss)
ax2.set_title("Val Loss")
ax3.plot(tAcc)
ax3.set_title("Train Acc")
ax4.plot(vAcc)
ax4.set_title("Val Acc")
plt.show()