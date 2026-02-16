import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils

if __name__ == "__main__":

    paramfilename = "apps/reprlearn/reprlearn.par"
    param = utils.parseparam(paramfilename)
    storedir = param["storedir"]
    
    # Load connection data
    cij = np.fromfile(f"{storedir}/learn.cij.l1.bin", dtype=np.int32)
    cij = cij.reshape(-1, param["Hhid"], 32, 32)    
    print (cij.shape, cij.min(), cij.max())

    # Plot per log step
    for t in range(len(cij)):
        fig, axs = plt.subplots(20, 20, figsize=(5, 6))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.8, wspace=0.15, hspace=0.15)
        axid = 0
        for ax in axs.flatten():
            ax.imshow(cij[t,axid], vmin=0, vmax=1, cmap="binary")
            ax.set_xticks([])
            ax.set_yticks([])
            axid += 1
        plt.savefig(f"cij.t{t:02d}.png", dpi=300)
        plt.close()

    exit()
        
    # Load test activity (layer 0)
    actl0 = np.fromfile("predict.teact.l0.log", dtype=np.float32)
    actl0 = actl0.reshape(10000,3,32,32,20)
    actl0sum = actl0.sum(axis=(4))
    print (actl0sum.min(), actl0sum.max())

    # Plot histogram activity (layer 0)
    plt.hist(actl0.flatten(), bins=100)
    plt.yscale("log")
    plt.savefig("actl0.hist.png", dpi=300)
    plt.close()
    print (actl0.shape, actl0.min(), actl0.max())

    # Load test activity (layer 1)
    actl1 = np.fromfile("predict.teact.l1.log", dtype=np.float32)
    actl1 = actl1.reshape(10000,3,30,100)
    actl1sum = actl1.sum(axis=(3))
    print (actl1sum.min(), actl1sum.max())

    # Plot histogram activity (layer 1)
    plt.hist(actl1.flatten(), bins=100)
    plt.yscale("log")
    plt.savefig("actl1.hist.png", dpi=300)
    plt.close()
    print (actl1.shape, actl1.min(), actl1.max())
