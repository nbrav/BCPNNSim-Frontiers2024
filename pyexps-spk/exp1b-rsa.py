from utils import *
from sklearn.metrics.pairwise import cosine_similarity

def get_orthoscore(simmat, lbl):
    """ ratio of mean similarity within-class and across all samples (higher ratio implies more orthogonal representations) """
    same_class = np.zeros((len(lbl), len(lbl)))
    for idx1 in range(len(lbl)):
        for idx2 in range(len(lbl)):
            same_class[idx1,idx2] = lbl[idx1]==lbl[idx2]
    same_sim = np.multiply(same_class==1, simmat).sum() / (same_class==1).sum()
    diff_sim = np.multiply(same_class==0, simmat).sum() / (same_class==0).sum()
    simscore = same_sim / simmat.mean()
    print (f"Simscore: {simscore}", same_sim, diff_sim, simmat.mean())
    return simscore

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

    """ Plot the similarity matrix """    

    # Number of patterns to load and show for simmilarity
    npat = 10000
    cmap = "jet"
    
    # Load labels for later sorting
    N = param['Hout'] * param['Mout']
    telbl = loadbin(param['datadir'], param['telblfile'], dtype=np.float32, count=npat*N, shape=(-1, N), verbose=1)
    telbl = telbl[:npat].argmax(axis=1)
    sortid = np.argsort(telbl)
    telbl = telbl[sortid]

    # Start plots
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.85, wspace=0.3, hspace=0.15)

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_linewidth(1)

    # Print pop name on top
    pop_name = [r"$INP$", r"$HID$",r"$INRC$"]
    for colid, ax in enumerate(axs[0]):
        axs[0,0].text(0.5, 1.3, pop_name[colid], 
                      transform=ax.transAxes, 
                      color="black", ha="center", va="center",fontsize=20
                      )

    # Simmat for INP 
    N = param['Hin'] * param['Min']
    act = loadbin(datadir, f"predict.ffwd.test.zi.01.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
    act = act[sortid] 
    simmat = cosine_similarity(act)
    axs[0,0].set_title(r"$T$=100ms", fontsize=15)
    im = axs[0,0].imshow(simmat, cmap=cmap, vmin=0, vmax=1)
 
    # Print orthoscore
    s_ortho = get_orthoscore(simmat, telbl)   
    axs[0,0].text(0.05, 0.1,r"$s_{ortho}=$"+f"{s_ortho:3.2f}", 
                      transform=axs[0,0].transAxes, 
                      color="black", ha="left", va="center", fontsize=15
                      )       
        
    # Remove one plot
    axs[1,0].axis('off')

    # Simmat for HID (T=100)
    N = param['Hhid'] * param['Mhid']
    act = loadbin(datadir, f"predict.ffwd.test.zj.11.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
    act = act[sortid] 
    simmat = cosine_similarity(act)
    axs[0,1].set_title(r"$T$="+f"{nstep_ffwd:0d}ms", fontsize=15)
    im = axs[0,1].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

    # Print orthoscore
    s_ortho = get_orthoscore(simmat, telbl)   
    axs[0,1].text(0.05, 0.1,r"$s_{ortho}=$"+f"{s_ortho:3.2f}", 
                      transform=axs[0,1].transAxes, 
                      color="white", ha="left", va="center", fontsize=15
                      )       

    # Simmat for HID (T=200)
    N = param['Hhid'] * param['Mhid']
    act = loadbin(datadir, f"predict.attractor.test.zj.11.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
    act = act[sortid] 
    simmat = cosine_similarity(act)
    axs[1,1].set_title(r"$T$="+f"{nstep_pat:0d}ms", fontsize=15)
    im = axs[1,1].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

    # Print orthoscore
    s_ortho = get_orthoscore(simmat, telbl)   
    axs[1,1].text(0.05, 0.1,r"$s_{ortho}=$"+f"{s_ortho:3.2f}", 
                      transform=axs[1,1].transAxes, 
                      color="white", ha="left", va="center", fontsize=15
                      )       

    # Simmat for INPRC (T=100)
    N = param['Hin'] * param['Min']
    act = loadbin(datadir, f"predict.ffwd.test.zj.12.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
    act = act[sortid] 
    simmat = cosine_similarity(act)
    axs[0,2].set_title(r"$T$="+f"{nstep_ffwd:0d}ms", fontsize=15)
    im = axs[0,2].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

    # Print orthoscore
    s_ortho = get_orthoscore(simmat, telbl)   
    axs[0,2].text(0.05, 0.1,r"$s_{ortho}=$"+f"{s_ortho:3.2f}", 
                      transform=axs[0,2].transAxes, 
                      color="black", ha="left", va="center", fontsize=15
                      )       

    # Simmat for INPRC (T=200)
    N = param['Hin'] * param['Min']
    act = loadbin(datadir, f"predict.attractor.test.zj.12.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
    act = act[sortid] 
    simmat = cosine_similarity(act)
    axs[1,2].set_title(r"$T$="+f"{nstep_pat:0d}ms", fontsize=15)
    im = axs[1,2].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

    # Print orthoscore
    s_ortho = get_orthoscore(simmat, telbl)   
    axs[1,2].text(0.05, 0.1,r"$s_{ortho}=$"+f"{s_ortho:3.2f}", 
                      transform=axs[1,2].transAxes, 
                      color="black", ha="left", va="center", fontsize=15
                      )       

    # Compute label info and set ticks
    labelname = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    labelstartid = np.zeros(11)
    labelmiddleid = np.zeros(10)
    for labelid in range(10):
        labelstartid[labelid] = np.argwhere(telbl==labelid).flatten()[0]
    labelstartid[-1] = npat
    for labelid in range(1,11):
        labelmiddleid[labelid-1] = labelstartid[labelid-1] + (labelstartid[labelid] - labelstartid[labelid-1])//2
    axs[0,0].set_yticks(labelmiddleid)
    axs[0,0].set_yticklabels(labelname)
    axs[0,0].set_xlabel("Pattern id (sorted)")

    # Colorbars
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5]) # [left, bottom, width, height] 
    fig.colorbar(im, cax=cbar_ax, label="Similarity")

    # plt.suptitle("Representational Similarity")
    plt.savefig("exp1b-rsa.png", dpi=400)
    plt.savefig("exp1b-rsa.svg", format='svg', dpi=400)
    if (SHOW): plt.show(block=True)
    plt.close()
