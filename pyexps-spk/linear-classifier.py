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
    def __init__(self, imgfile, lblfile, imgshape, nclass=10, binarize=False, device="cpu", transform=None):
        assert (len(imgshape)==2)
        H, M = imgshape[0], imgshape[1]
        # Load image data
        img = np.fromfile(f"{imgfile}", dtype=np.float32)
        img = img.reshape(-1, H, M)
        print (f"Loading {imgfile}")
        # Load label data
        lbl_onehot = np.fromfile(f"{lblfile}", dtype=np.float32).reshape(-1, nclass)
        lbl = np.argmax(lbl_onehot, axis=1)
        # Tranform to torch tensor
        self.img = torch.from_numpy(img)
        self.lbl = torch.from_numpy(lbl)
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.lbl[idx]

class Linear(nn.Module):
    def __init__(self, inp_dim=28*28, out_dim=10):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
            
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
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
    num_epochs = 20
    device = get_device()

    # Set seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    parentdir = "/cfs/klemming/scratch/n/nbrav/logs/"
    modeldir_all = ["test-mnist1k"] #["rate-ffwd", "rate-full", "spk-ffwd", "spk-full", "sparsespk-ffwd", "sparsespk-full"]

    for modeldir in modeldir_all:

        runid = 0

        for datetimedir in [""]: #datetimedir in os.listdir(f"{parentdir}/{modeldir}/"):

            #if "2024-04-20_" in datetimedir: continue # skip old runs

            datadir = f"{parentdir}/{modeldir}/{datetimedir}"

            # Parse parameter file
            paramfilename = "apps/hidassospk/hidassospk.par" #f"{datadir}/net.par"
            param = utils.parseparam(paramfilename)
            
            # Set params 
            H = param["Hhid"]
            M = param["Mhid"]
            dat_shape = (H, M)

            # Set data file names
            trdatfile = f"{datadir}/predict.ffwd.train.zj.01.bin" # use ffwd acts for training
            tedatfile = f"{datadir}/predict.attractor.test.zj.01.bin" # attr acts for testing
            # tedatfile_complete = f"{datadir}/predict.attractor.complete.zj.01.bin"
            # tedatfile_rivalry = f"{datadir}/predict.attractor.rivalry.zj.01.bin"
            # tedatfile_distort = f"{datadir}/predict.attractor.distort.zj.01.bin"            
            
            # Set label file names
            trlblfile = param["datadir"]+param["trlblfile"]
            telblfile = param["datadir"]+param["telblfile"]
            # telblfile_complete = param["datadir"]+param["complete_telblfile"]
            # telblfile_rivalry = param["datadir"]+param["rivalry_telblfile"]
            # telblfile_distort = param["datadir"]+param["distort_telblfile"]
            
            # Load data
            train_dataset = CustomDataset(imgfile=trdatfile, lblfile=trlblfile, imgshape=dat_shape, device=device)
            test_dataset = CustomDataset(imgfile=tedatfile, lblfile=telblfile, imgshape=dat_shape, device=device)
            # complete_dataset = CustomDataset(imgfile=tedatfile_complete, lblfile=telblfile_complete, imgshape=dat_shape, device=device)
            # rivalry_dataset = CustomDataset(imgfile=tedatfile_rivalry, lblfile=telblfile_rivalry, imgshape=dat_shape, device=device)
            # distort_dataset = CustomDataset(imgfile=tedatfile_distort, lblfile=telblfile_distort, imgshape=dat_shape, device=device)
    
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            # complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            # rivalry_dataloader = DataLoader(rivalry_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            # distort_dataloader = DataLoader(distort_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
            #  Initialize Network
            net = Linear(H*M).to(device)
        
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
                    if counter % 10 == 0:
                        acc, _ = evaluate(net, test_dataloader)
                        test_acc_hist.append(acc)
                        print(f"Epoch: {epoch:-2d}, Batch: {counter:-4d}, Loss: {loss_val.item():-5.5f}, Test Acc: {acc:-5.2f}%")
                    counter += 1

            #  Test on assomem tasks
            # num_diff = 10 # number of difficulty levels
            # task_names = ["test", "complete", "rivalry", "distort"]
            # task_dataloaders = [test_dataloader, complete_dataloader, rivalry_dataloader, distort_dataloader]
            # # Iterate over all tasks
            # for task_name, task_dataloader in zip(task_names, task_dataloaders):    
            #     # Pass through net and get correct/wrong predictions
            #     acc, corrects = evaluate(net, task_dataloader)
            #     # Record accuracy per difficulty
            #     acc_per_diff = corrects.reshape(num_diff,-1).mean(axis=1)*100
            #     print (task_name, acc_per_diff)
            #     # Store log
            #     with open(f"assomemlog.corrects.{modeldir}.{task_name}.{runid}.txt", "w") as log_file:
            #         for correct in corrects.flatten():
            #             log_file.write(f"{correct*1:1d},")
                        
            runid += 1
            
    print('Fin.')