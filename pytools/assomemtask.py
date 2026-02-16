import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import parseparam, loadbin
import os
import csv        
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":

    SHOW = 1
    
    # matplotlib params
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size':15})
    
    num_diff = 10
    diff_all = np.arange(0.1, 1.1, 0.1)
    tasknames = ["PattCmp", "PercRiv", "DistRes"]
    taskprettynames = {"PattCmp": "Pattern Completion", 
                  "PercRiv": "Perceptual Rivalry", 
                  "DistRes": "Distortion Resistance" }
    disttypename = ["noise", "grid", "clutter", "deletion", "occlusion"]
    model_all = ["rate-ffwd", "rate-full", "spk-ffwd", "spk-full", "sparsespk-ffwd", "sparsespk-full"]
    modelcolors = {
        "rate-ffwd": "skyblue",
        "rate-full": "blue",
        "spk-ffwd": "lime",
        "spk-full": "limegreen",
        "sparsespk-ffwd": "orange",
        "sparsespk-full": "red",
        }
    hatchtype = {
        "rate-ffwd": "",
        "rate-full": "",
        "spk-ffwd": "//",
        "spk-full": "//",
        "sparsespk-ffwd": "..",
        "sparsespk-full": "..",
    }
    modelprettyname_all = [
        '$\it{RateFfwd}$',
        '$\it{RateFull}$',
        '$\it{SpkFfwd}$',
        '$\it{SpkFull}$',
        '$\it{SpspkFfwd}$',
        '$\it{SpspkFull}$',
    ]
        
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
    plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9, wspace=0.5, hspace=0.3)

    for taskid in range(len(tasknames)):
            
        ax = axs[taskid]
        
        taskname = tasknames[taskid]

        for modelid in range(len(model_all)): 

            modelname = model_all[modelid]
            
            # Read correct/wrong log file
            corrects = []
            with open(f"assomemlog.corrects.{modelname}.{taskname}.txt", "r") as log_file:
                csvreader = csv.reader(log_file, delimiter=',')
                for row in csvreader:
                    for col in row[:-1]:
                        corrects.append(int(col))
            corrects = np.asarray(corrects)

            # Calcualte accuracy per difficulty
            acc_per_diff = corrects.reshape(num_diff,-1).mean(axis=1)*100
            print (corrects.shape, acc_per_diff.shape)

            gap_fraction = 1.25

            # Plot bars
            x = [diffid * len(model_all) * gap_fraction + modelid for diffid in range(num_diff)]
            y = acc_per_diff            
            im = ax.bar(x=x, height=y, width=1, edgecolor="black", linewidth=1, color=modelcolors[modelname], alpha=1, hatch=hatchtype[modelname])

            # Set labels and limits
            ax.set_ylim(0,100)
            # Set ticks
            tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(num_diff)]
            ax.set_xticks(tickpos, [])

            # Remove axis lines
            ax.spines[['right', 'top']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(1)

    # Set label
    axs[-1].set_xlabel("Difficulty level", fontsize=12)
    axs[-1].set_ylabel("Accuracy (%)",  fontsize=12)

    # Set ticks
    tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(num_diff)]
    tickprettyname = [f"{diff:2.1f}" for diff in diff_all]
    axs[-1].set_xticks(tickpos, tickprettyname, fontsize=12)

    # Put a legend above upper axis
    axs[1].legend(modelprettyname_all, 
               loc='center', 
               bbox_to_anchor=(1.2, 0.5),
               ncol=1, fontsize=12, frameon=False) 

    # Set title as taskprettyname
    for taskid in range(len(tasknames)):
        ax = axs[taskid]
        taskname = tasknames[taskid]
        axs[taskid].text(0.5, 1.1, 
                      taskprettynames[taskname], 
                      transform=ax.transAxes, 
                      color="black", 
                      horizontalalignment="center", 
                      verticalalignment="center",
                      fontsize=15
                      )


    plt.savefig(f"assomemtask.svg", format='svg', dpi=400)
    plt.savefig(f"assomemtask.png", dpi=400)
    if (SHOW): plt.show()

    exit()

    # Find per distortion type
    task = "DistRes"
    num_disttype = 5
    corrects_all = {}
    for modelid in range(len(model_all)): 
        modelname = model_all[modelid]
        # Read correct/wrong log file
        corrects = []
        with open(f"assomemlog.corrects.{modelname}.{task}.txt", "r") as log_file:
            csvreader = csv.reader(log_file, delimiter=',')
            for row in csvreader:
                for col in row[:-1]:
                    corrects.append(int(col))
        corrects_all[modelname] = np.asarray(corrects).reshape(num_diff,num_disttype,-1)
    # Start plot
    fig, axs = plt.subplots(nrows=num_diff, ncols=1, figsize=(num_disttype, num_diff))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
    for diffid in range(num_diff):
        axid = diffid
        for disttype in range(num_disttype):
            for modelid in range(len(model_all)): 
                modelname = model_all[modelid]
                x = disttype*len(model_all)*1.5 + modelid
                height = corrects_all[modelname].mean(axis=2)[diffid,disttype]*100
                axs[axid].bar(x=x, height=height, width=1, edgecolor="black", color=modelcolors[modelname], alpha=1, hatch=hatchtype[modelname])
        axs[axid].set_ylim(0,100)
        axs[axid].set_title(f"Difficulty {diff_all[diffid]:.1f}", fontsize=12)
        # Remove ticks
        axs[axid].set_xticks([])
        # Remove axis lines
        axs[axid].spines[['right', 'top']].set_visible(False)
        axs[axid].spines[['left', 'bottom']].set_linewidth(1)
    # Set ticks
    tickpos = [disttype*len(model_all)*1.5+len(model_all)/2-0.5 for disttype in range(len(disttypename))]
    tickprettyname = [disttypename[disttype] for disttype in range(len(disttypename))]
    axs[-1].set_xticks(ticks=tickpos, labels=tickprettyname, fontsize=12)
    axs[-1].set_ylabel("Accuracy (%)")
    # Finalize plot
    plt.savefig(f"assomemtask.disttype.png", dpi=400)
    plt.show()
    
    print ("Fin.")
