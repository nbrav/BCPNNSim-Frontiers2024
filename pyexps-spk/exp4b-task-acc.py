from utils import *

import csv        
import pandas as pd

if __name__ == "__main__":

    SHOW = 0
    
    tasknames = ["complete", 
                 "rivalry", 
                 "distort"]
    taskprettynames = {"complete": "Pattern Completion", 
                       "rivalry": "Perceptual Rivalry", 
                       "distort": "Distortion Resistance" }
    model_all = ["rate-ffwd", 
                 "rate-full", 
                 "spk-ffwd", 
                 "spk-full", 
                 "sparsespk-ffwd", 
                 "sparsespk-full"]
    modelcolors = {
        "rate-ffwd": "white",
        "rate-full": "gray",
        "spk-ffwd": "white",
        "spk-full": "gray",
        "sparsespk-ffwd": "white",
        "sparsespk-full": "gray",
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
        '$\it{RateFf}$',
        '$\it{RateFull}$',
        '$\it{SpkFf}$',
        '$\it{SpkFull}$',
        '$\it{SpspkFf}$',
        '$\it{SpspkFull}$',
    ]
    disttypename = ["noise", "grid", "clutter", "deletion", "occlusion"]
                        
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 5))
    plt.subplots_adjust(left=0.12, right=0.73, bottom=0.1, top=0.9, wspace=0.5, hspace=0.3)

    nrun = 5 # number of randomized runs
    num_diff = 10
    diff_all = np.arange(0.1, 1.1, 0.1)
    new_num_diff = 5
    new_diff_all = np.arange(0.2, 1.1, 0.2)    

    # Just print out test performance
    taskname = "test"
    # Iterate over models
    for modelid in range(len(model_all)):         
        modelname = model_all[modelid]    
        acc = []
        # Iterate over randomized runs
        for runid in range(nrun):        
            # Read correct/wrong log file
            corrects = []
            with open(f"assomemlog.corrects.{modelname}.{taskname}.{runid}.txt", "r") as log_file:
                csvreader = csv.reader(log_file, delimiter=',')
                for row in csvreader:
                    for col in row[:-1]:
                        corrects.append(int(col))
            corrects = np.asarray(corrects)    
            acc.append(corrects.mean(axis=0)*100)
        # Calcualte accuracy per difficulty
        acc = np.asarray(acc)
        print (f"{modelname} {acc.mean():4.2f} {acc.std():4.2f}")
    
    # Iterate over tasks
    for taskid in range(len(tasknames)):      
        ax = axs[taskid]
        taskname = tasknames[taskid]
        # Iterate over models
        for modelid in range(len(model_all)): 
            modelname = model_all[modelid]
            acc_per_diff = []
            # Iterate over randomized runs
            for runid in range(nrun):
                # Read correct/wrong log file
                corrects = []
                with open(f"assomemlog.corrects.{modelname}.{taskname}.{runid}.txt", "r") as log_file:
                    csvreader = csv.reader(log_file, delimiter=',')
                    for row in csvreader:
                        for col in row[:-1]:
                            corrects.append(int(col))
                corrects = np.asarray(corrects)
                corrects = corrects.reshape(num_diff,-1).mean(axis=1)*100
                acc_per_diff.append(corrects)
            
            # Calcualte accuracy per difficulty
            acc_per_diff = np.asarray(acc_per_diff)
            
            # Show 5 not 10 difficulty
            acc_per_diff = acc_per_diff[:,1::2]
            print (acc_per_diff.shape)
            
            # PLot variables            
            gap_fraction = 1.25

            # Plot bars
            x = [diffid * len(model_all) * gap_fraction + modelid for diffid in range(new_num_diff)]
            y = acc_per_diff.mean(axis=0)
            yerr = acc_per_diff.std(axis=0)
            
            print (x, y, yerr)
                 
            im = ax.bar(x=x, height=y, yerr=yerr, width=1, capsize=2, edgecolor="black", linewidth=1, color=modelcolors[modelname], alpha=0.7, hatch=hatchtype[modelname])
            
            # Set labels and limits
            ax.set_ylim(0,100)
            # Set ticks
            tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(new_num_diff)]
            ax.set_xticks(tickpos, [])

            # Remove axis lines
            ax.spines[['right', 'top']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(1)
           
    # Set label
    axs[-1].set_xlabel("Difficulty level", fontsize=12)
    axs[1].set_ylabel("Accuracy (%)",  fontsize=12)

    # Set ticks
    tickpos = [diffid * len(model_all) * gap_fraction + modelid - len(model_all)/2 + 0.5 for diffid in range(new_num_diff)]
    tickprettyname = [f"{diff:2.1f}" for diff in new_diff_all]
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

    plt.savefig(f"exp4-task-acc.svg", format='svg', dpi=400)
    plt.savefig(f"exp4-task-acc.png", dpi=400)
    if (SHOW): plt.show()

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
