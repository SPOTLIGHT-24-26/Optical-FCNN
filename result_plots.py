import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import models
import data_loader

import json

device = "cuda" if torch.cuda.is_available() else "cpu"

pathList = ["result_lists/New_final/mnist_fourier_1e-05_20_subset.json",
            "result_lists/New_final/mnist_fourier_1e-05_28_subset.json",
            "result_lists/New_final/mnist_spatial_1e-06_20250319_124749_28_100.json",
            "result_lists/New_final/mnist_digital_1e-06_20250319_132842_28_100.json",
            "result_lists/New_final/fmnist_fourier_1e-05_20_subset.json",
            "result_lists/New_final/fmnist_fourier_1e-05_28_subset.json",
            "result_lists/New_final/fMnist_spatial_1e-06_20250325_182622_28_100.json",
            "result_lists/New_final/fMnist_digital_1e-06_20250325_190055_28_100.json"
            ]

# Join results from checkpoints
# appendDict = {}
# with open(pathList[0], "r") as resultData:
#     appendDict = json.load(resultData)
# for path in pathList[1:]:
#     with open(path, "r") as resultData:
#         tempDict = json.load(resultData)
#         appendDict["tLoss"] += tempDict["tLoss"]
#         appendDict["vLoss"] += tempDict["vLoss"]
#         appendDict["tAcc"] += tempDict["tAcc"]
#         appendDict["vAcc"] += tempDict["vAcc"]

# appendDict["tLoss"] = appendDict["tLoss"][:-30]
# appendDict["vLoss"] = appendDict["vLoss"][:-30]
# appendDict["tAcc"] = appendDict["tAcc"][:-30]
# appendDict["vAcc"] = appendDict["vAcc"][:-30]
# json.dump(appendDict,open("result_lists/mnist_fourier_1e-05_28_appended_final.json","w"), indent=2)

# # sub-sample 100 epochs for fourier results
# with open("result_lists/New_final/mnist_fourier_1e-05_28_appended_final.json", "r") as resultData:
#     outDict = json.load(resultData)
# idx = np.round(np.linspace(0, 570 - 1, 100)).astype(int)
# outDict["tLoss"] = [outDict["tLoss"][ind] for ind in idx]
# outDict["vLoss"] = [outDict["vLoss"][ind] for ind in idx]
# outDict["tAcc"] = [outDict["tAcc"][ind] for ind in idx]
# outDict["vAcc"] = [outDict["vAcc"][ind] for ind in idx]
# json.dump(outDict,open("result_lists/New_final/fmnist_fourier_1e-05_20_subset.json","w"), indent=2)

resultDicts = []
# get losses and accuracies
for path in pathList:
    with open(path, "r") as resultData:
        resultDicts.append(json.load(resultData))

# labels and markers
labels = ["$MNIST - \\mathcal{S}_{F-20}$",
          "$MNIST - \\mathcal{S}_{F-28}$",
          "$MNIST - \\mathcal{S}_{sp}$",
          "$MNIST - \\mathcal{S}_{d}$",
          "$FMNIST - \\mathcal{S}_{F-20}$",
          "$FMNIST - \\mathcal{S}_{F-28}$",
          "$FMNIST - \\mathcal{S}_{sp}$",
          "$FMNIST - \\mathcal{S}_{d}$"
          ]
markers = ['*', '+', 'x', '1', '2', '3', '>', '<']

fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.add_subplot(ax1)
fig1.add_subplot(ax2)
# Iterate and plot all
for i, tDict in enumerate(resultDicts):
    ax1.plot(tDict["vLoss"], label=labels[i], marker=markers[i], markevery=20, markersize=10)
    ax2.plot(tDict["vAcc"], label=labels[i], marker=markers[i], markevery=20, markersize=10)

#ax1.set_title("Loss", fontsize=20)
ax1.set_xlabel("Epochs", fontsize=25)
ax1.set_ylabel("Loss", fontsize=25)
ax1.tick_params(labelsize=25)
ax1.legend(loc="upper right", prop = { "size": 25 })

#ax2.set_title("Accuracy", fontsize=20)
ax2.set_xlabel("Epochs", fontsize=25)
ax2.set_ylabel("Accuracy", fontsize=25)
ax2.tick_params(labelsize=25)
ax2.legend(loc="lower right", prop = { "size": 25 })

# ------------------Phase perturbations and evaluation---------------------------
modelPaths = [['models/abs_squared_final/mnist_spatial_model_20250319_124749_1e-06_28_100.pt',
               'models/abs_squared_final/mnist_fourier_model_20250317_150048_1e-05_28_184.pt',
               'models/abs_squared_final/mnist_fourier_model_20250314_151634_1e-05_20_292.pt'],
               ['models/abs_squared_final/fMnist_spatial_model_20250325_182622_1e-06_28_100.pt',
                'models/abs_squared_final/fMnist_fourier_model_20250319_192911_1e-05_28_107.pt',
                'models/abs_squared_final/fMnist_fourier_model_20250320_215610_1e-05_20_164.pt']
]
mode = ['spatial', 'fourier', 'fourier']
dataName = ['mnist', 'fMnist']
normalize = ['True', 'False', 'False']
imSize = [28, 28, 20]
numComponents = [28, 28, 20]
sigma = np.linspace(5e-3, 5e-2, 10)
acc = np.zeros((2,3,10))
batch_size = 64
# save original accuracies for plotting
# acc[0,0,0] = 97.7
# acc[0,1,0] = 96.0
# acc[0,2,0] = 96.9
# acc[1,0,0] = 88.3
# acc[1,1,0] = 84.8
# acc[1,2,0] = 85.0

for d in range(2):
    for m in range(3):
        #model = None
        testSet = None
        if dataName[d] == 'mnist':
            _, testSet = data_loader.MnistData(mode[m], normalize[m], batch_size, numComponents[m]).getDataLoaders()
        else:
            _, testSet = data_loader.FashionMnistData(mode[m], normalize[m], batch_size, numComponents[m]).getDataLoaders()
        if mode[m] == 'spatial':
            model = models.simpleCNN(device, 1, imSize[m])
        else:
            model = models.simpleFCNN(device, 1, imSize[m])
        # Make perturbations to the weights of convolution layer
        # checkpoint = torch.load(modelPaths[d][m], weights_only=True)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        model.load_state_dict(torch.load(modelPaths[d][m], weights_only=True))
        # convert weights to phases
        U1, S1, V1 = None, None, None
        U2, S2, V2 = None, None, None
        w1, w2 = None, None
        if mode[m] == 'spatial':
            U1, S1, V1 = model.conv1.U, model.conv1.S, model.conv1.V
            U2, S2, V2 = model.conv2.U, model.conv2.S, model.conv2.V
        else:
            w1 = torch.acos(model.conv1.weight)
            w2 = torch.acos(model.conv2.weight)
        print("Original accuracy - data {}, model {}, components {} = {}".format(dataName[d], mode[m], numComponents[m], acc[d,m,0]))
        for ind, s in enumerate(sigma):
            if mode[m] == 'spatial':
                dThetaU1, dThetaU2 = np.random.normal(scale=s, size=U1.shape), np.random.normal(scale=s, size=U2.shape)
                dThetaS1, dThetaS2 = np.random.normal(scale=s, size=S1.shape), np.random.normal(scale=s, size=S2.shape)
                dThetaV1, dThetaV2 = np.random.normal(scale=s, size=V1.shape), np.random.normal(scale=s, size=V2.shape)
                model.conv1.U.data = U1 + torch.from_numpy(dThetaU1.astype('float32')).to(device)
                model.conv1.S.data = S1 + torch.from_numpy(dThetaS1.astype('float32')).to(device)
                model.conv1.V.data = V1 + torch.from_numpy(dThetaV1.astype('float32')).to(device)
                model.conv2.U.data = U2 + torch.from_numpy(dThetaU2.astype('float32')).to(device)
                model.conv2.S.data = S2 + torch.from_numpy(dThetaS2.astype('float32')).to(device)
                model.conv2.V.data = V2 + torch.from_numpy(dThetaV2.astype('float32')).to(device)
            else:
                dTheta1 = np.random.normal(scale=s, size=w1.shape)
                dTheta2 = np.random.normal(scale=s, size=w2.shape)
                model.conv1.weight.data = torch.cos(w1 + torch.from_numpy(dTheta1.astype('float32')).to(device))
                model.conv2.weight.data = torch.cos(w2 + torch.from_numpy(dTheta2.astype('float32')).to(device)) 
            # evaluate on the test set
            running_vacc = 0.0
            model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(testSet):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = model(vinputs)
                    running_vacc += sum(torch.argmax(voutputs, 1) == torch.argmax(vlabels, 1))/batch_size
            testAcc = (running_vacc / (i+1)).item()
            acc[d,m,ind] = testAcc
            print("Perturbed accuracy at sigma {} = {}".format(s, testAcc))

# Load saved accuracies for fourier models
sigTestPaths = [
    "result_lists/mnist_sig_28.json",
    "result_lists/mnist_sig_20.json",
    "result_lists/fmnist_sig_28.json",
    "result_lists/fmnist_sig_20.json"
]

pNum = 0
for d in range(2):
    for m in [1,2]:
        with open(sigTestPaths[pNum], "r") as resultData:
            accDict = json.load(resultData)
            acc[d,m,:] = np.asarray(accDict["acc"])
            pNum += 1


# Plot accuracies
np.insert(sigma, 0, 0.0)
labels1 = ["$MNIST - \\mathcal{S}_{sp}$",
          "$MNIST - \\mathcal{S}_{F-28}$",
          "$MNIST - \\mathcal{S}_{F-20}$",
          "$FMNIST - \\mathcal{S}_{sp}$",
          "$FMNIST - \\mathcal{S}_{F-28}$",
          "$FMNIST - \\mathcal{S}_{F-20}$"
          ]
fig2 = plt.figure()
for datNum in range(acc.shape[0]):
    for modNum in range(acc.shape[1]):
        plt.plot(sigma, acc[datNum, modNum, :], label=labels1[datNum*3 + modNum], 
                 marker=markers[datNum*3 + modNum], markevery=1, markersize=10)
fig2.supxlabel("Phase perturbation variance ($\\sigma$)", fontsize=25)
fig2.supylabel("Accuracy", fontsize=25)
plt.tick_params(labelsize=25)
plt.legend(loc="lower left", prop = { "size": 25 })

plt.show()