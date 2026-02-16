from utils import *
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":

    SHOW = 0
    
    # Declare the datadir and paramfile
    datadir = "/cfs/klemming/scratch/n/nbrav/logs/sparsespk-full/2024-04-21_14:40:38:083781"
    paramfilename = f"{datadir}/net.par"
    
    param = parseparam(paramfilename)
    
    """ Plot the receptive field formation """    

    # Create x-ticks
    probes = [numer*np.power(10, expon) for expon in range(10) for numer in range(1,10) ]
    probes = np.asarray(probes)
    
    # Setup plots
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.2)
    
    colid = 0
    for prjid in ["01", "12"]: # 11
        Hi, Mi, Hj, Mj, prjname = getprjinfo(prjid, param)
        rowid = 0
        for field in ["nswap", "nmi"]:
            if field in ["nmi"]:
                cij = loadbin(datadir, filename=f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
                dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.float32, shape=(-1, Hj, Hi), verbose=1)
                dat = np.multiply(dat, cij)
                dat = dat.reshape(-1,Hj,Hi).sum(axis=2)
            elif field in ["nswap"]:
                dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.int32, shape=(-1, Hj), verbose=1)
                dat = dat[1:] # first element is just zero
            elif field in ["del_cij"]:
                cij = loadbin(datadir, filename=f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
                dat = cij[1:] - cij[:-1]
                dat = dat.reshape(-1, Hj, Hi)
                dat = np.abs(dat).sum(axis=2)
                dat = dat / 2. # two conn changes is one swap
            elif field in ["mean_del_pij"]:
                dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.float32, shape=(-1, 1), verbose=1)
            # Create x- and y-lines
            y_mean = dat.mean(axis=1)[22:]
            y_std = dat.std(axis=1)[22:]
            x = probes[22:][:len(y_mean)] #np.arange(len(y_mean)) # probes
            # Plot stuff
            ax = axs[rowid,colid]
            ax.plot(x, y_mean, "ro-", linewidth=1.5)
            ax.fill_between(x=x, y1=y_mean-y_std, y2=y_mean+y_std, alpha=0.25, color="red")
            # Set x- and y-ticks and grid
            axs[0,colid].set_title(f"{prjname}", fontsize=20)
            ax.set_xlim(400)
            #ax.set_ylim(-0.1*np.max(dat), 1.1*np.max(dat))
            #ax.set_ylim(-0.1*np.max(dat[23:]), 1.1*np.max(dat[23:]))
            ax.set_ylim(0)
            ax.set_xscale('log')
            ax.grid(which='both')
            ax.minorticks_on()
            if field=="nswap":
                axs[rowid,0].set_ylabel(f"Num. of flips")
            if field=="nmi":
                axs[rowid,0].set_ylabel(r"$\tilde{M}$")                
            rowid += 1
        colid += 1
    axs[-1, 0].set_xlabel("Training samples")
    plt.savefig("exp2b-rewiring-scores.png", dpi=400)
    plt.savefig("exp2b-rewiring-scores.svg", format='svg', dpi=400)
    if (SHOW): plt.show(block=True)
    plt.close()
