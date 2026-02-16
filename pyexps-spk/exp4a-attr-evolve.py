from utils import *

if __name__ == "__main__":

    SHOW = 0
    
    # Declare the datadir and paramfile
    datadir = "/cfs/klemming/scratch/n/nbrav/logs/test-mnist1k/"
    # "/cfs/klemming/scratch/n/nbrav/logs/sparsespk-full/2024-04-21_14:40:38:083781"
    paramfilename = "apps/hidassospk/hidassospk.par" #f"{datadir}/net.par"
    
    param = parseparam(paramfilename)
    
    H = param["Hin"]
    M = param["Min"]
    
    nstep_gap = param['nstep_gap']
    nstep_ffwd = param['nstep_ffwd']
    nstep_overlap = param['nstep_overlap']
    nstep_recr = param['nstep_recr']
    
    nstep_pat = nstep_ffwd + nstep_overlap + nstep_recr
    nstep = nstep_gap + nstep_pat
    
    dataset_all = ["test"]#, "complete", "rivalry", "distort"]

    """ Plot the attractor timeline """
    
    print ("attractor_imagestrip")
        
    for dataset in dataset_all:

        # Initialize variables
        fieldnames = ["act_inp", "z_inp", "act_inprc", "z_inprc"]
        fieldfilenames = [f"everystep.{dataset}.act.0.bin",
                      f"everystep.{dataset}.zi.01.bin",
                      f"everystep.{dataset}.act.2.bin",
                      f"everystep.{dataset}.zj.12.bin"]
        fieldprettynames = [r"${INP}$ spikes",
                        r"${INP}$ $z$-traces",
                        r"${INPRC}$ spikes",
                        r"${INPRC}$ $z$-traces"]
        # Params
        nfield = len(fieldnames) # number of log variables to show
        nrow = nfield
        ncol = 15 # Time checkpoints to plot
        vmin, vmax = 0, 1 # color min and max range
        nshowstep = nstep_pat # gap period is boring to show
        npat = 40

        # Load all data file
        dat = {} 
        for fieldid in range(nfield):
            fieldname = fieldnames[fieldid]
            print (datadir, fieldfilenames[fieldid])
            dat[fieldname] = loadbin(datadir=datadir, 
                                 filename=fieldfilenames[fieldid], 
                                 offset=(0*nstep*14*14*2*np.dtype(np.float32).itemsize), 
                                 count=(npat*nstep*14*14*2), 
                                 shape=(-1,nstep,14*14,2), 
                                 verbose=1) 
            dat[fieldname] = dat[fieldname]
            
        # Do the plots for a few patterns
        for patid in range(0, npat):
            # Initialize plot
            fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
            plt.subplots_adjust(left=0.05, right=0.8, bottom=0.2, top=0.95, wspace=0.1, hspace=0.1)
            for rowid in range(nrow):
                fieldid = rowid
                fieldname = fieldnames[fieldid]
                for colid in range(ncol):
                    # find time in simulation land to show
                    t = nstep_gap + int(colid*nshowstep/ncol)
                    imgdat = dat[fieldname][patid,t].reshape(14,14,M)[:,:,0]
                    ax =  axs[rowid,colid]
                    ax.imshow(imgdat, cmap="binary", vmin=vmin, vmax=vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
            # Set x labels as times
            for colid in range(ncol):
                t = int(colid*nshowstep/ncol)
                axs[-1,colid].set_xlabel(t) 
        
            # Set x text as time
            axs[-1,ncol//2].text(0, -0.8, 
                      "Time since pattern onset (ms)", 
                      transform=axs[-1,ncol//2].transAxes, 
                      color="black", 
                      horizontalalignment="center", 
                      verticalalignment="center",
                      fontsize=20
                      )
            
            # Set ylabel
            for rowid in range(nrow):
                ax = axs[rowid,-1]
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel('Y', rotation=0, labelpad=10)
                ax.set_ylabel(fieldprettynames[rowid], rotation="horizontal", va="center", ha="left", fontsize=20, labelpad=20)
        
            # Finalize plot
            plt.savefig(f"exp4-attr-evolve.{dataset}.pat{patid:02d}.png", dpi=400)
            plt.close()

