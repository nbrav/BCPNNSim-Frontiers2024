from utils import *

from sklearn.metrics.pairwise import cosine_similarity

def find_prototype(simmat, s_min = 0.1):
    
    """ return a dict of prototypes and patterns converging on it """
    # key: value =  prototype_pat_id: all pat_id converging on prototype
    prototypes = []
    # Iterate over patterns
    for pat_id in range(len(simmat)):        
        prototype_match = False
        prototype_match_id = -1
        # Iterate over prototypes and best match
        for prototype_id in range(len(prototypes)):
            # Find patterns of each prototype
            pat_per_prototype = prototypes[prototype_id]
            # Pick the first pattern for each prototype
            prototype_first_pat_id = pat_per_prototype[0] 
            # Find prototype with closest match
            if (simmat[pat_id, prototype_first_pat_id] > s_min):
                prototype_match = True
                prototype_match_id = prototype_id
            pass
        if (prototype_match == True):
            # Add pattern to the matched prototype
            prototypes[prototype_match_id].append(pat_id)
        else:
            # Make this pattern a new prototype and add it to the list
            prototypes.append([pat_id])

    return prototypes
    
def plot_prototypes(z_inprc, prototypes, popularity_ids):
    # Start plots
    ncol = 11
    nrow = math.ceil(len(prototypes)/ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    plt.subplots_adjust(left=0.25, right=0.75, bottom=0.01, top=0.49, wspace=0.1, hspace=0.1)
    # remove all axis first
    for ax in axs.flatten():
        ax.axis("off")
    # Do plotting
    axid = 0
    for prototype_id in popularity_ids:
        ax = axs.flatten()[axid]
        ax.axis("on")
        pats_per_prototype = prototypes[prototype_id]
        # find avg prototype
        avg_prototype = np.zeros((28*28*2))
        for pat_id in pats_per_prototype:
            avg_prototype += z_inprc[pat_id]
        avg_prototype /= len(pats_per_prototype)
        # Plot average prototype reconstr.
        ax.imshow(avg_prototype.reshape(28,28,2)[:,:,0], vmin=0, vmax=1, cmap="binary")
        ax.text(0, 0, len(prototypes[prototype_id]), fontsize=10, va='top', ha='left')
        axid += 1
    # Manage axis
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    # Find top middle axis to add text
    if len(axs.shape)==1:
        ax = axs[ncol//2]
    else:
        ax = axs[0,ncol//2]
    # Add text
    fig.text(0.5, 1.3, r"$S_{min}=$"+f"{s_min}", fontsize=20, transform=ax.transAxes, ha='center')
    plt.savefig(f"exp5-ptype-smin{s_min:1.2f}.png", dpi=400)
    if (SHOW): plt.show(block=True)
    plt.close()

if __name__ == "__main__":

    SHOW = 0
    
    # Declare the datadir and paramfile
    datadir = "/cfs/klemming/scratch/n/nbrav/logs/sparsespk-full/2024-04-21_14:40:38:083781"
    paramfilename = f"{datadir}/net.par"
    
    param = parseparam(paramfilename)
    
    H = param["Hin"]
    M = param["Min"]
    
    nstep_gap = param['nstep_gap']
    nstep_ffwd = param['nstep_ffwd']
    nstep_overlap = param['nstep_overlap']
    nstep_recr = param['nstep_recr']
    
    nstep_pat = nstep_ffwd + nstep_overlap + nstep_recr
    nstep = nstep_gap + nstep_pat
    
    dataset_all = ["test", "complete", "rivalry", "distort"]

    """ Plot the attractor timeline """    
    print ("exp6. prototype")
    
    # Set params
    npat = 10000 # numbers of patterns to use for simmat
    s_min_all = np.arange(0, 1, 0.01) # find prototypes for these s_min's
    s_min_plot = [0.01, 0.05, 0.1, 0.2] # plot these prototypes

    # Load labels
    telbl_1hot = loadbin(param['datadir'], param['telblfile'], dtype=np.float32, shape=(-1, param["Hout"]*param["Mout"]), verbose=1)
    telbl_1hot = telbl_1hot[:npat]
    telbl = telbl_1hot.argmax(axis=1)
    sortid = np.argsort(telbl)
    
    # Load z-traces    
    z_inp = loadbin(datadir, f"predict.attractor.test.zi.01.bin", dtype=np.float32, shape=(-1, param["Hin"]*param["Min"]), verbose=1)
    z_hid = loadbin(datadir, f"predict.attractor.test.zj.11.bin", dtype=np.float32, shape=(-1, param["Hhid"]*param["Mhid"]), verbose=1)
    z_inprc = loadbin(datadir, f"predict.attractor.test.zj.12.bin", dtype=np.float32, shape=(-1, param["Hin"]*param["Min"]), verbose=1)
    
    # Limit dataset to small size if needed
    z_inp = z_inp[:npat]
    z_hid = z_hid[:npat]
    z_inprc = z_inprc[:npat]    
    
    # Compute simmat
    simmat = cosine_similarity(z_hid, z_hid)
    
    # Iterate to find prototype ids
    num_prototype_found_all = []
    popularity_all = []
    
    for s_min in s_min_all:
        
        # Call and find ptypes
        prototypes = find_prototype(simmat, s_min)
        num_prototype_found_all.append(len(prototypes))
        print (s_min, len(prototypes))

        # Sort prototypes by popularity (num. of patterns converged on it) and store
        num_pat_per_prototype = np.array([len(pat_per_prototype) for pat_per_prototype in prototypes])
        popularity_ids = np.argsort(num_pat_per_prototype)[::-1]
        popularity = np.sort(num_pat_per_prototype)[::-1]
        popularity_all.append(popularity)
            
        # Plot prototypes for some s_min
        # if s_min in s_min_plot:
        #     plot_prototypes(z_inprc, prototypes, popularity_ids)    

    # Plot num. of prototype vs s_min
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.1, top=0.95, wspace=0.5, hspace=0.5)
    # Plot simmat histogram
    simmat[range(len(simmat)),range(len(simmat))] = -1
    axs[0].hist(simmat.flatten(), bins=100, color="black", alpha=0.6)
    axs[0].set_yscale("log")
    axs[0].set_xlim(-0.01, 1.01)
    axs[0].set_xlabel(r"$S$ (pair-wise similarity)")
    axs[0].set_ylim(1)
    axs[0].set_ylabel("Count")
    # Plot num prototype found
    axs[1].plot(s_min_all, num_prototype_found_all, ".-", linewidth=2, color="black", alpha=0.6)
    axs[1].set_xlabel(r"$S_{min}$ (similarity threshold)")
    axs[1].set_xlim(-0.01, 1.01)
    axs[1].set_ylabel("Num. of prototypes\nfound")
    axs[1].set_ylim(0, npat*1.1)
    # Add fig labels
    axs[0].text(-0.25, 1.1, "a", transform=axs[0].transAxes, color="black", ha="left", va="top", fontsize=25)       
    axs[1].text(-0.25, 1.1, "b", transform=axs[1].transAxes, color="black", ha="left", va="top", fontsize=25)       
    # Remove axis lines
    for ax in axs.flatten():
        ax.spines[['right', 'top']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1)
    plt.savefig("exp5-nptyp.png", dpi=400)
    plt.close()