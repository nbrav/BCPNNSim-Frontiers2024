from utils import *
from sklearn.metrics.pairwise import cosine_similarity

def plot_rcpfld(nrow, ncol, iters, conn, Ix, Iy, title, prjid):
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol/2, nrow/2))
    plt.subplots_adjust(left=0.19, right=0.99, bottom=0.1, top=0.9, wspace=0., hspace=0.)
    for row in range(nrow):
        for col in range(ncol):
            probe = row
            hc = col
            axs[row, col].imshow(conn[probe, hc].reshape(Ix, Iy), cmap="binary", interpolation="None")
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
    fig.text(0.03, 0.5, 'Training samples', va='bottom', ha='center', rotation='vertical')
    for row in range(nrow):
        axs[row, 0].set_ylabel(f"{iters[row]:d}", rotation=0, ha='right', va='center')
    plt.suptitle(f"{title}")
    plt.savefig(f"exp2a-{prjid}.png", dpi=400)
    plt.savefig(f"exp2a-{prjid}.svg", format='svg', dpi=400)
    if (SHOW): plt.show(block=True)
    plt.close()

if __name__ == "__main__":

    SHOW = 0
    
    # Declare the datadir and paramfile
    datadir = "/cfs/klemming/scratch/n/nbrav/logs/sparsespk-full/2024-04-21_14:40:38:083781"
    paramfilename = f"{datadir}/net.par"
    
    param = parseparam(paramfilename)
    
    """ Plot the receptive field formation """    

    # Set parameters
    Ix=28
    Iy=28
    num_hypercol_show=10
    ACTIVE, SILENT = 1, 0
    probes = [numer*np.power(10, expon) for expon in range(10) for numer in range(1,10) ]
    probes = np.array(probes)
    probeid = [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49, 54] # gets nice probes like 1000, 5000, 10000, etc. Shpuld be a better way though
    probes = probes[probeid] 
    
    for popid in range(1, param["nlayer"]):
        
        # Feedforward receptive field
        prjid = "01"
        Hi, _, Hj, _, prjname = getprjinfo(prjid, param)
        conn = loadbin(datadir, f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
        conn = conn[probeid]
        plot_rcpfld(nrow=conn.shape[0], ncol=num_hypercol_show, iters=probes, conn=conn, Ix=Ix, Iy=Iy, title=prjname, prjid=prjid)
        
        # Feedback receptive field
        prjid = "12"
        Hi, _, Hj, _, prjname = getprjinfo(prjid, param)
        conn = loadbin(datadir, f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
        conn = conn.transpose(0, 2, 1)
        conn = conn[probeid]
        plot_rcpfld(nrow=conn.shape[0], ncol=num_hypercol_show, iters=probes, conn=conn, Ix=Ix, Iy=Iy, title=prjname, prjid=prjid)


