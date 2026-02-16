import numpy as np
import os
import utils
import pandas as pd
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

class CustomDataset(Dataset):
    """ Derived from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """
    def __init__(self, imgfile, lblfile, imgshape, imgskipoffset=0, imgskipint=1, nclass=10, binarize=False, device="cpu", transform=None):
        H, W, C = imgshape[0], imgshape[1], imgshape[2]
        # Load image data
        img = np.fromfile(f"{imgfile}", dtype=np.float32)
        img = img.reshape(-1, H, W, C)
        print (f"Loading {imgfile}")
        # Load label data
        lbl_onehot = np.fromfile(f"{lblfile}", dtype=np.float32).reshape(-1, nclass)
        print (f"Original loaded {img.shape}, {lbl_onehot.shape}")
        img = img.transpose(0, 3, 1, 2)
        img = img[imgskipoffset::imgskipint]
        lbl = np.argmax(lbl_onehot, axis=1)
        print (f"Reshaped {img.shape}, {lbl.shape}")
        # Tranform to torch tensor
        self.img = torch.from_numpy(img)
        self.lbl = torch.from_numpy(lbl)
        # Binarize grayscale images
        # if (binarize==True):
        #     assert self.img.shape[1]==1
        #     img_bin = torch.zeros(len(img), 2, H, W, device=device)
        #     img_bin[:,0] = self.img.view(len(img), H, W) 
        #     img_bin[:,1] = 1 - img_bin[:,0]
        #     self.img = img_bin
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.lbl[idx]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

class Linear(nn.Module):
    def __init__(self, ni=28*28):
        super().__init__()
        self.fc = nn.Linear(ni, 10)
            
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x    
    
class MLP(nn.Module):
    def __init__(self, ni=28*28):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 3000)
        self.fc2 = nn.Linear(3000, 10)
            
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x);
        return x
    
def evaluate(net, dataloader):
    with torch.no_grad():
        net.eval()
        corrects = []
        num_corr = 0
        num_pat = 0
        # Forward pass
        for data, targets in iter(dataloader):
            data = data.to(device)
            targets = targets.to(device)
            pred = net(data)
            corrects.append(torch.argmax(pred, axis=1) == targets)
        # Compute total accuracy
        corrects = torch.cat(corrects, dim=0).detach().cpu().numpy()
        return corrects.mean()*100, corrects

if __name__ == "__main__":

    # Meta-parameters
    SEED = 1128
    batch_size = 64
    num_epochs = 2 # 20
    device = get_device()

    # Set seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    parentdir = "/cfs/klemming/scratch/n/nbrav/logs/"
    all_model_name = ["rate-ffwd", "rate-full", "spk-ffwd", "spk-full", "sparsespk-ffwd", "sparsespk-full"]

    for model_name in all_model_name:

        for datetimedir in os.listdir(f"{parentdir}/{model_name}/"):

            #if not "2024-04-17_19" in datetimedir: continue

            datadir = f"{parentdir}/{model_name}/{datetimedir}"

            # Parse parameter file
            paramfilename = f"{datadir}/net.par"
            param = utils.parseparam(paramfilename)
            
            # Load data
            Hx, Hy, M = 10, 10, 100
            train_dataset = CustomDataset(imgfile=f"{datadir}/predict.ffwd.train.zj.01.bin",
                                  lblfile=param["datadir"]+param["trlblfile"],
                                  nclass=10, imgshape=(Hx,Hy,M), imgskipoffset=0, imgskipint=1, device=device)
            test_dataset = CustomDataset(imgfile=f"{datadir}/predict.attractor.test.zj.01.bin",
                                 lblfile=param["datadir"]+param["telblfile"],
                                 nclass=10, imgshape=(Hx,Hy,M), imgskipoffset=0, imgskipint=1, device=device)
            complete_dataset = CustomDataset(imgfile=f"{datadir}/predict.attractor.complete.zj.01.bin",
                                     lblfile=param["datadir"]+param["complete_telblfile"],
                                     nclass=10, imgshape=(Hx,Hy,M), imgskipoffset=0, imgskipint=1, device=device)
            rivalry_dataset = CustomDataset(imgfile=f"{datadir}/predict.attractor.rivalry.zj.01.bin",
                                    lblfile=param["datadir"]+param["rivalry_telblfile"],
                                    nclass=10, imgshape=(Hx,Hy,M), imgskipoffset=0, imgskipint=1, device=device)
            distort_dataset = CustomDataset(imgfile=f"{datadir}/predict.attractor.distort.zj.01.bin",
                                    lblfile=param["datadir"]+param["distort_telblfile"],
                                    nclass=10, imgshape=(Hx,Hy,M), imgskipoffset=0, imgskipint=1, device=device)
    
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            rivalry_dataloader = DataLoader(rivalry_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            distort_dataloader = DataLoader(distort_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
            #  Initialize Network
            net = Linear(Hx*Hy*M).to(device)
        
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
            loss_fn = nn.CrossEntropyLoss()
    
            loss_hist = []
            test_acc_hist = []

            # Outer training loop
            counter = 0
            for epoch in range(num_epochs):        
                # Training loop
                for data, targets in iter(train_dataloader):
                    data = data.to(device)
                    targets = targets.to(device)
                    # forward pass
                    net.train()
                    preds = net.forward(data)
                    # initialize the loss & sum over time
                    loss_val = loss_fn(preds, targets)
                    # Gradient calculation + weight update
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
                    # Store loss history for future plotting
                    loss_hist.append(loss_val.item())
                    # Test set
                    if counter % 500 == 0:
                        acc, _ = evaluate(net, test_dataloader)
                        test_acc_hist.append(acc)
                        print(f"Epoch: {epoch:-2d}, Batch: {counter:-4d}, Loss: {loss_val.item():-5.5f}, Test Acc: {acc:-5.2f}%")
                    counter += 1

            # #  Test on assomem tasks
            # num_diff = 10 # number of difficulty levels
            # task_names = ["PattCmp", "PercRiv", "DistRes"]
            # task_dataloaders = [complete_dataloader, rivalry_dataloader, distort_dataloader]
            # # Iterate over all tasks
            # for task_name, task_dataloader in zip(task_names, task_dataloaders):    
            #     # Pass through net and get correct/wrong predictions
            #     acc, corrects = evaluate(net, task_dataloader)
            #     # Record accuracy per difficulty
            #     acc_per_diff = corrects.reshape(num_diff,-1).mean(axis=1)*100
            #     print (task_name, acc_per_diff)
            #     # Store log
            #     with open(f"assomemlog.corrects.{model_name}.{task_name}.txt", "w") as log_file:
            #         for correct in corrects.flatten():
            #             log_file.write(f"{correct*1:1d},")

    print('Fin.')