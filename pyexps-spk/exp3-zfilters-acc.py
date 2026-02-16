from utils import *

import csv        
import pandas as pd
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

    SHOW = 0

    parentdatadir = "/cfs/klemming/scratch/n/nbrav/logs/exp3-zfilter-logs/"
    
    # Meta-parameters
    SEED = 1128
    batch_size = 64
    num_epochs = 20
    device = get_device()

    # Set seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Classifier
    
    for rundir in os.listdir(parentdatadir):
                
        datadir = f"{parentdatadir}/{rundir}"
                
        # Parse parameter file
        paramfilename = f"{datadir}/net.par"
        param = parseparam(paramfilename)

        # Set params 
        H = param["Hhid"]
        M = param["Mhid"]
        dat_shape = (H, M)

        # Set data file names
        trdatfile = f"{datadir}/predict.ffwd.train.zj.01.bin" # use ffwd acts for training
        tedatfile = f"{datadir}/predict.attractor.test.zj.01.bin" # attr acts for testing
        tedatfile_complete = f"{datadir}/predict.attractor.complete.zj.01.bin"
        tedatfile_rivalry = f"{datadir}/predict.attractor.rivalry.zj.01.bin"
        tedatfile_distort = f"{datadir}/predict.attractor.distort.zj.01.bin"            
            
        # Set label file names
        trlblfile = param["datadir"]+param["trlblfile"]
        telblfile = param["datadir"]+param["telblfile"]
        telblfile_complete = param["datadir"]+param["complete_telblfile"]
        telblfile_rivalry = param["datadir"]+param["rivalry_telblfile"]
        telblfile_distort = param["datadir"]+param["distort_telblfile"]
            
        # Load data
        train_dataset = CustomDataset(imgfile=trdatfile, lblfile=trlblfile, imgshape=dat_shape, device=device)
        test_dataset = CustomDataset(imgfile=tedatfile, lblfile=telblfile, imgshape=dat_shape, device=device)
        complete_dataset = CustomDataset(imgfile=tedatfile_complete, lblfile=telblfile_complete, imgshape=dat_shape, device=device)
        rivalry_dataset = CustomDataset(imgfile=tedatfile_rivalry, lblfile=telblfile_rivalry, imgshape=dat_shape, device=device)
        distort_dataset = CustomDataset(imgfile=tedatfile_distort, lblfile=telblfile_distort, imgshape=dat_shape, device=device)
    
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        rivalry_dataloader = DataLoader(rivalry_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        distort_dataloader = DataLoader(distort_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
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
                if counter % 500 == 0:
                    acc, _ = evaluate(net, test_dataloader)
                    test_acc_hist.append(acc)
                    print(f"Epoch: {epoch:-2d}, Batch: {counter:-4d}, Loss: {loss_val.item():-5.5f}, Test Acc: {acc:-5.2f}%")
                counter += 1

        #  Test on assomem tasks
        task_name = "test"
        task_dataloader = test_dataloader
        # Pass through net and get correct/wrong predictions
        acc, corrects = evaluate(net, task_dataloader)
        # Record accuracy per difficulty
        acc_per_diff = corrects.mean(axis=0)*100
        print (task_name, len(corrects), acc_per_diff)
        # Store log
        # with open(f"assomemlog.corrects.{modeldir}.{task_name}.{runid}.txt", "w") as log_file:
        #     for correct in corrects.flatten():
        #         log_file.write(f"{correct*1:1d},")
                        
        # runid += 1    

    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 5))
    # plt.subplots_adjust(left=0.12, right=0.73, bottom=0.1, top=0.9, wspace=0.5, hspace=0.3)

    # nrun = 5 # number of randomized runs
    # num_diff = 10
    # diff_all = np.arange(0.1, 1.1, 0.1)
    # new_num_diff = 5
    # new_diff_all = np.arange(0.2, 1.1, 0.2)    

    # # Just print out test performance
    # taskname = "test"
    # # Iterate over models
    # for modelid in range(len(model_all)):         
    #     modelname = model_all[modelid]    
    #     acc = []
    #     # Iterate over randomized runs
    #     for runid in range(nrun):        
    #         # Read correct/wrong log file
    #         corrects = []
    #         with open(f"assomemlog.corrects.{modelname}.{taskname}.{runid}.txt", "r") as log_file:
    #             csvreader = csv.reader(log_file, delimiter=',')
    #             for row in csvreader:
    #                 for col in row[:-1]:
    #                     corrects.append(int(col))
    #         corrects = np.asarray(corrects)    
    #         acc.append(corrects.mean(axis=0)*100)
    #     # Calcualte accuracy per difficulty
    #     acc = np.asarray(acc)
    #     print (f"{modelname} {acc.mean():4.2f} {acc.std():4.2f}")
    
    # # Iterate over tasks
    # for taskid in range(len(tasknames)):      
    #     ax = axs[taskid]
    #     taskname = tasknames[taskid]
    #     # Iterate over models
    #     for modelid in range(len(model_all)): 
    #         modelname = model_all[modelid]
    #         acc_per_diff = []
    #         # Iterate over randomized runs
    #         for runid in range(nrun):
    #             # Read correct/wrong log file
    #             corrects = []
    #             with open(f"assomemlog.corrects.{modelname}.{taskname}.{runid}.txt", "r") as log_file:
    #                 csvreader = csv.reader(log_file, delimiter=',')
    #                 for row in csvreader:
    #                     for col in row[:-1]:
    #                         corrects.append(int(col))
    #             corrects = np.asarray(corrects)
    #             corrects = corrects.reshape(num_diff,-1).mean(axis=1)*100
    #             acc_per_diff.append(corrects)
            
    #         # Calcualte accuracy per difficulty
    #         acc_per_diff = np.asarray(acc_per_diff)
            
    #         # Show 5 not 10 difficulty
    #         acc_per_diff = acc_per_diff[:,1::2]
    #         print (acc_per_diff.shape)
            
    #         # PLot variables            
    #         gap_fraction = 1.25

    #         # Plot bars
    #         x = [diffid * len(model_all) * gap_fraction + modelid for diffid in range(new_num_diff)]
    #         y = acc_per_diff.mean(axis=0)
    #         yerr = acc_per_diff.std(axis=0)
            
    #         print (x, y, yerr)
                 
    #         im = ax.bar(x=x, height=y, yerr=yerr, width=1, capsize=2, edgecolor="black", linewidth=0, color=modelcolors[modelname], alpha=0.7, hatch=hatchtype[modelname])
            
    #         # Set labels and limits
    #         ax.set_ylim(0,100)
    #         # Set ticks
    #         tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(new_num_diff)]
    #         ax.set_xticks(tickpos, [])

    #         # Remove axis lines
    #         ax.spines[['right', 'top']].set_visible(False)
    #         ax.spines[['left', 'bottom']].set_linewidth(1)
           
    # # Set label
    # axs[-1].set_xlabel("Difficulty level", fontsize=12)
    # axs[-1].set_ylabel("Accuracy (%)",  fontsize=12)

    # # Set ticks
    # tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(new_num_diff)]
    # tickprettyname = [f"{diff:2.1f}" for diff in new_diff_all]
    # axs[-1].set_xticks(tickpos, tickprettyname, fontsize=12)

    # # Put a legend above upper axis
    # axs[1].legend(modelprettyname_all, 
    #            loc='center', 
    #            bbox_to_anchor=(1.2, 0.5),
    #            ncol=1, fontsize=12, frameon=False) 

    # # Set title as taskprettyname
    # for taskid in range(len(tasknames)):
    #     ax = axs[taskid]
    #     taskname = tasknames[taskid]
    #     axs[taskid].text(0.5, 1.1, 
    #                   taskprettynames[taskname], 
    #                   transform=ax.transAxes, 
    #                   color="black", 
    #                   horizontalalignment="center", 
    #                   verticalalignment="center",
    #                   fontsize=15
    #                   )

    # plt.savefig(f"exp4-task-acc.svg", format='svg', dpi=400)
    # plt.savefig(f"exp4-task-acc.png", dpi=400)
    # if (SHOW): plt.show()

    # exit()

    # # Find per distortion type
    # task = "distort"
    # num_disttype = 5
    # corrects_all = {}
    # for modelid in range(len(model_all)): 
    #     modelname = model_all[modelid]
    #     # Read correct/wrong log file
    #     corrects = []
    #     with open(f"assomemlog.corrects.{modelname}.{task}.txt", "r") as log_file:
    #         csvreader = csv.reader(log_file, delimiter=',')
    #         for row in csvreader:
    #             for col in row[:-1]:
    #                 corrects.append(int(col))
    #     corrects_all[modelname] = np.asarray(corrects).reshape(num_diff,num_disttype,-1)
    # # Start plot
    # fig, axs = plt.subplots(nrows=num_diff, ncols=1, figsize=(num_disttype, num_diff))
    # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
    # for diffid in range(num_diff):
    #     axid = diffid
    #     for disttype in range(num_disttype):
    #         for modelid in range(len(model_all)): 
    #             modelname = model_all[modelid]
    #             x = disttype*len(model_all)*1.5 + modelid
    #             height = corrects_all[modelname].mean(axis=2)[diffid,disttype]*100
    #             axs[axid].bar(x=x, height=height, width=1, edgecolor="black", color=modelcolors[modelname], alpha=1, hatch=hatchtype[modelname])
    #     axs[axid].set_ylim(0,100)
    #     axs[axid].set_title(f"Difficulty {diff_all[diffid]:.1f}", fontsize=12)
    #     # Remove ticks
    #     axs[axid].set_xticks([])
    #     # Remove axis lines
    #     axs[axid].spines[['right', 'top']].set_visible(False)
    #     axs[axid].spines[['left', 'bottom']].set_linewidth(1)
    # # Set ticks
    # tickpos = [disttype*len(model_all)*1.5+len(model_all)/2-0.5 for disttype in range(len(disttypename))]
    # tickprettyname = [disttypename[disttype] for disttype in range(len(disttypename))]
    # axs[-1].set_xticks(ticks=tickpos, labels=tickprettyname, fontsize=12)
    # axs[-1].set_ylabel("Accuracy (%)")
    # # Finalize plot
    # plt.savefig(f"assomemtask.disttype.png", dpi=400)
    # plt.show()
    
    print ("Fin.")
