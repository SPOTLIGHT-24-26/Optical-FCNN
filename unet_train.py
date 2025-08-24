import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from unet_model import u_net
from unet_spacial import SpatialUNet
from coco_stuff_loader import CocoPanoptic
from datetime import datetime
import json
import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex
import random
from torchmetrics import Accuracy

"""
Expected structure:
top-dir
|
| - images
|
| - result_lists
|
| - Optical-FCNN
|   |-  coco_stuff_loader.py
|   |-  unet_model.py
|   |-  unet_spacial.py
|   |_  unet_train.py
|
| - data
    |- coco
        |- train2017
        |       |_ trainset pictures
        |
        | - val2017
        |       |_ valset pictures
        |
        | - annotations
                | - panoptic_train2017.json
                | - panoptic_val2017.json
                | - panoptic_train2017
                |           |_ trainset masks
                | - panoptic_val2017
                            |_ valset masks
"""
timestamp = datetime.now().strftime('%m%d_%H%M')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
EPOCHS = 50
lr = 1e-4
moment = 0.99

tLoss, vLoss = [], []
tIoU, vIoU = [], []
tAcc, vAcc = [], []

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# when training spatial model use CocoPanoptic(...., mode = 'spatial')
train_data = CocoPanoptic(split="train", download=False, img_size = 150, img_target = 108) # target 108 because that is what spatial gives for input 150
test_data = CocoPanoptic(split="val", download=False, img_size = 150, img_target = 108)
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
num_classes = train_data.get_num_classes()
print("Number of classes possible:", num_classes)

#model = SpatialUNet(device=device, imChannels=1, imSize = 150, img_target =108, numClasses=num_classes).to(device) # spatial
model = u_net(device=device, imChannels=1, imSize = 150, img_target =108, numClasses=num_classes).to(device) # fourier domain

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_one_epoch():
    model.train()
    running_loss = 0.0
    miou = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pixel_acc = Accuracy(task = "multiclass", num_classes = num_classes).to(device)

    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 99:
            print('  batch {} loss: {}'.format(i + 1, loss.item()))
        
        preds = outputs.argmax(dim=1)
        miou.update(preds, labels)
        pixel_acc.update(preds, labels)

    avg_loss = running_loss / len(trainLoader)
    iou = miou.compute().mean().item()
    miou.reset()
    acc = pixel_acc.compute().item()
    pixel_acc.reset()
    #print("finished one")
    return avg_loss, iou, acc

def validate():
    model.eval()
    running_loss = 0.0
    miou = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pixel_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            miou.update(preds, labels)
            pixel_acc.update(preds, labels)

    avg_loss = running_loss / len(testLoader)
    iou = miou.compute().mean().item()
    miou.reset()
    acc = pixel_acc.compute().item()
    pixel_acc.reset()
    return avg_loss, iou, acc

# train loop
best_vloss = float('inf')
best_model = model.state_dict()
model_path = f'models/fourier_unet_{timestamp}'

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch+1}/{EPOCHS}')

    train_loss, train_iou, train_acc = train_one_epoch()
    val_loss, val_iou, val_acc = validate()
    tLoss.append(train_loss)
    vLoss.append(val_loss)
    tAcc.append(train_acc)
    vAcc.append(val_acc)
    tIoU.append(train_iou)
    vIoU.append(val_iou)

    print(f'Train Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}, acc: {train_acc:.4f}')
    print(f'Val   Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, acc: {val_acc:.4f}')
    """
    if val_loss < best_vloss:
        best_vloss = val_loss
        best_model = model.state_dict()
        torch.save(best_model, model_path + '.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, model_path + '_chkPnt.tar')
    """

file_path = f'result_lists/fourier_unet_{timestamp}.json'
out_dict = {"tLoss": tLoss, "vLoss": vLoss, "tIoU": tIoU, "vIoU": vIoU}
json.dump(out_dict, open(file_path, "w"), indent=2)

# plots
fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.plot(tLoss)
ax1.set_title("Train Loss")
ax2.plot(vLoss)
ax2.set_title("Val Loss")
ax3.plot(tIoU)
ax3.set_title("Train mIoU")
ax4.plot(vIoU)
ax4.set_title("Val mIoU")
ax5.plot(tAcc)
ax5.set_title("Train Pixel acc")
ax6.plot(vAcc)
ax6.set_title("Val Pixel Acc")
plt.savefig('images/training_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'Best val loss: {best_vloss:.4f}')
