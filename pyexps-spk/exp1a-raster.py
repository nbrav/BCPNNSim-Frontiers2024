from utils import *

if __name__ == "__main__":

    SHOW = 0
    
    # Declare the datadir and paramfile
    datadir = "/cfs/klemming/scratch/n/nbrav/logs/sparsespk-full/2024-04-21_14:40:38:083781"
    paramfilename = f"{datadir}/net.par"
    
    param = parseparam(paramfilename)
    
    nstep_gap = param['nstep_gap']
    nstep_ffwd = param['nstep_ffwd']
    nstep_overlap = param['nstep_overlap']
    nstep_recr = param['nstep_recr']
    
    nstep_pat = nstep_ffwd + nstep_overlap + nstep_recr
    nstep = nstep_gap + nstep_pat

    """ Plot the raster and firing rate of each population """    

    # Setup parameters
    Pstart = 0 # Start of pattern to show
    Pshow = 3 # Number of patterns to show
    Hshow_all = [5, 4, 5] # Hypercolumns to show
    nspikyshow = 2 # Number of minicols to plot firing rate
    spikycolors = ["red", "blue", "green", "orange"] # Color of select minicols

    # Setup Gaussian smoothing operation
    kernel_window = 1000 # msec
    gaussian_sigma = 20 # msec
    moving_window = np.linspace(-kernel_window/2, kernel_window/2, kernel_window)
    gaussian_kernel = np.exp(-(moving_window/gaussian_sigma)**2/2) / np.sqrt(2*np.pi*gaussian_sigma**2)
    
    # Setup plots
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.15)
    
    # Fix the ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_linewidth(1)

    # Time (ms)        
    Tstart = Pstart * nstep
    Tshow = Pshow * nstep

    # Set axis limits
    for ax in axs.flatten():
        ax.set_xlim(0, Tshow)
            
    # Iterate over all poopulations
    for popid in range(0, 3):  
        
        # get pop infor
        H, M, popidname = getpopinfo(popid, param)

        colors = plt.cm.jet(np.linspace(0,1,M))
        
        # Decide what hypercols to plot
        Hstart = H//3 # np.random.randint(H//3, H*2//3)
        Hshow = Hshow_all[popid] 
        Nstart = Hstart * M
        Nshow = Hshow * M
        
        # Load activity
        spkact = loadbin(datadir=datadir, 
                         filename=f"everystep.test.act.{popid}.bin", 
                         dtype=np.float32, 
                         shape=(-1, H * M), 
                         offset=(Pstart * nstep * H * M * np.dtype(np.float32).itemsize),
                         count=(Pshow * nstep * H * M),
                         verbose=1)
                
        # Find spike timings
        spktimes_all = [] # List of spike timings per minicol
        for m in range(Nstart, Nstart+Nshow):            
            spktimes = np.argwhere(spkact[Tstart:Tstart+Tshow, m]).flatten() 
            spktimes_all.append(spktimes)
            
        # Find firing rates
        firingrate_all = []
        for m in range(Nstart, Nstart+Nshow):
            firingrate = np.convolve(spkact[:, m], gaussian_kernel, mode="same")
            #firingrate *= 0.7 * 1000. / param['maxfq'] # convert to Hz
            firingrate *= 1000. # convert to Hz
            firingrate_all.append(firingrate[Tstart:Tstart+Tshow])

        # find some nice high-firing minicols to show
        spiky_m = []
        for m in range(Nshow):
            avg_firing_rate = np.max(firingrate_all[m])
            if (avg_firing_rate > 0.1 * param['maxfq']):
                spiky_m.append(m)
        selectspiky_m = np.random.choice(spiky_m, size=nspikyshow, replace=False)
        # Make all minicols black except the selectspiky ones
        colors = ["black" for n in range(Nshow)]
        for spiky_id in range(nspikyshow):
            colors[selectspiky_m[spiky_id]] = spikycolors[spiky_id]

        # Plot spike raster
        linelength = 0.05 * Nshow # scale to size of y-axis
        lineoffsets = 0.5 + np.arange(0, Nshow)
        axs[0,popid].eventplot(spktimes_all, 
                               lineoffsets=lineoffsets,
                               linewidth=1, linelength=linelength, alpha=0.75, 
                               colors=colors
                               )  
        
        # Set yticks
        axs[0,popid].set_yticks(ticks=np.arange(0, Hshow*M+M, step=M))
        axs[0,popid].set_ylim(0, Nshow)

        # Plot firing rate for selectspiky_m's
        for show_id in range(nspikyshow):
            m = selectspiky_m[show_id]
            axs[1,popid].plot(firingrate_all[m],
                              linewidth=2, 
                              alpha=0.75, 
                              color=spikycolors[show_id],
                              )
        
        # Set yticks
        axs[1,popid].set_yticks(ticks=[0, param['maxfq']], labels=[0, param['maxfq']])
        axs[1,popid].set_ylim(0, param['maxfq']*1.5)

        # Draw line boundaries per hypercolumn   
        # for h in range(Hshow):
        #     axs[0,popid].axhline(y=h/Hshow, xmin=0, xmax=Tshow, linewidth=1, color="black", alpha=0.5)
    
        # Plot stimulus filling area
        stimulus = np.zeros((Tshow//nstep, nstep)) # gap period
        stimulus[:, nstep_gap:nstep_gap+nstep_pat] = 1 # pattern  duration
        stimulus = stimulus.flatten()
        axs[0, popid].fill_between(range(0, Tshow), y1=0, y2=Hshow*M, where=stimulus>0.5, facecolor='black', alpha=0.15)
        axs[1, popid].fill_between(range(0, Tshow), y1=0, y2=param['maxfq']*1.5, where=stimulus>0.5, facecolor='black', alpha=0.15)

        # Set plot text
        axs[0,popid].set_title(f"{popidname}", fontsize=20)
        
    # Set xticks    
    axs[1,0].set_xticks(ticks=np.arange(0, Tshow+nstep, step=nstep), labels=np.arange(0, Tshow+nstep, step=nstep))

    # Set ticks
    axs[0,0].set_ylabel("Minicolumn id")
    axs[-1,0].set_xlabel("Time (ms)")
    axs[-1,0].set_ylabel("Firing rate (Hz)")
    
    # Finalize plots
    plt.savefig(f"exp1a-raster.png", dpi=400)
    plt.savefig(f"exp1a-raster.svg", format="svg", dpi=400)
    if (SHOW): plt.show(block=True)
    plt.close()
