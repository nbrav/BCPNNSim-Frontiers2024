import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip

def download():
    '''
    Download files from url
    '''
    os.makedirs(f"{DownloadDir}", exist_ok=True)

    for filename in filenames.values():

        os.system(f"wget -nc -P {DownloadDir} {url}/{filename}")

    return

def make_raw():

    f = gzip.open(f'{DownloadDir}/{filenames["trimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(60000*28*28)
    trimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(60000, 28, 28)/255.    
    print (trimg.shape, trimg.min(), trimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["teimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(10000*28*28)
    teimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(10000, 28, 28)/255.    
    print (teimg.shape, teimg.min(), teimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["trlbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(60000*1)
    trlbl = np.frombuffer(buf, dtype=np.uint8).reshape(60000, 1) 
    trlbl = np.eye(10)[trlbl]
    print (trlbl.shape, trlbl.min(), trlbl.max())

    f = gzip.open(f'{DownloadDir}/{filenames["telbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(10000*1)
    telbl = np.frombuffer(buf, dtype=np.uint8).reshape(10000, 1) 
    telbl = np.eye(10)[telbl]
    print (telbl.shape, telbl.min(), telbl.max())

    os.makedirs(f"{RawDir}", exist_ok=True)
    
    trimg.astype('float32').tofile(f"{RawDir}/mnist_trimg.bin")
    trlbl.astype('float32').tofile(f"{RawDir}/mnist_trlbl.bin")
    teimg.astype('float32').tofile(f"{RawDir}/mnist_teimg.bin")
    telbl.astype('float32').tofile(f"{RawDir}/mnist_telbl.bin")
    
    return

def make_completion(ndiff=5, ndisttype=4):
    # Load dataset
    teimg = np.fromfile(f"{RawDir}/mnist_teimg.bin", dtype=np.float32).reshape(10000, 28, 28)
    telbl = np.fromfile(f"{RawDir}/mnist_telbl.bin", dtype=np.float32).reshape(10000, 10)
    ntest = len(teimg)
    # Create of copy of test image data for distortions
    teimg = teimg[:ndisttype*250].reshape(ndisttype,250,28,28)
    telbl = telbl[:ndisttype*250].reshape(ndisttype,250,10)
    dteimg = np.zeros((ndiff,ndisttype,250,28,28))
    dtelbl = np.zeros((ndiff,ndisttype,250,10))
    # Loop across different difficulty levels
    for diffid in range(ndiff):
        w = int((diffid+1)/ndiff*14) # border width
        # Loop across different distortion levels
        for disttypeid in range(ndisttype):
            # Copy test set to distorted test set
            dteimg[diffid, disttypeid] = teimg[disttypeid].copy()
            # Iterate over patterns and distort each
            if disttypeid==0:
                dteimg[diffid, disttypeid, :, :w, :] = 0.5
            elif disttypeid==1:
                dteimg[diffid, disttypeid, :, -w:, :] = 0.5
            elif disttypeid==2:
                dteimg[diffid, disttypeid, :, :, :w] = 0.5
            elif disttypeid==3:
                dteimg[diffid, disttypeid, :, :, -w:] = 0.5
            else:
                print ("Invalid disttypeid!")
            dtelbl[diffid, disttypeid] = telbl[disttypeid].copy() 
    # Store images for plot & show
    PatCmpDir = f"Complete"    
    os.makedirs(f"{PatCmpDir}", exist_ok=True)
    dteimg.astype('float32').tofile(f"{PatCmpDir}/mnist_teimg.bin")
    dtelbl.astype('float32').tofile(f"{PatCmpDir}/mnist_telbl.bin")
    # Do plotting
    fig, axs = plt.subplots(ndisttype, ndiff, figsize=(ndiff*2/3, ndisttype*2/3))
    fig.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.8, hspace=0., wspace=0.)
    plotgrid(fig, axs, dteimg[:,:,0])
    for diffid in range(ndiff):
        axs[-1,diffid].set_xlabel(f"{(diffid+1)/ndiff:.1f}", fontsize=10)
    disttypename = ["top", "bottom", "left", "right"]
    for disttypeid in range(ndisttype):
        axs[disttypeid,0].set_ylabel(f"{disttypename[disttypeid]}", fontsize=10, rotation="horizontal", va="center", ha="right")
    fig.text(0.65, 0.05, r'Difficulty Level', ha='center', fontsize=10)
    fig.text(0.65, 0.90, r'Pattern Completion', ha='center', fontsize=10)
    fig.text(0.1, 0.95, "a", color="black", ha="left", va="top", fontsize=15)       
    plt.savefig(f"mnist.cmp.png", dpi=400)
    plt.savefig(f"mnist.cmp.svg", format="svg", dpi=400)
    if (SHOW): plt.show(block=False)

def make_rivalry(ndiff=5, ndisttype=4):
    # Load original dataset
    teimg = np.fromfile(f"{RawDir}/mnist_teimg.bin", dtype=np.float32).reshape(10000, 28, 28)
    telbl = np.fromfile(f"{RawDir}/mnist_telbl.bin", dtype=np.float32).reshape(10000, 10)
    ntest = len(teimg)
    # Shuffled rival ids for selecting types
    patids = np.arange(ntest)
    rivids = np.arange(ntest,0,-1)
    np.random.shuffle(rivids)
    # Create of copy of test image data for distortions
    teimg = teimg[:ndisttype*250].reshape(ndisttype,250,28,28)
    telbl = telbl[:ndisttype*250].reshape(ndisttype,250,10)
    dteimg = np.zeros((ndiff,ndisttype,250,28,28))
    dtelbl = np.zeros((ndiff,ndisttype,250,10))
    # Loop across different difficulty levels
    for diffid in range(ndiff):
        # Border width
        w = int((diffid+1)/ndiff*14)
        # Loop across different distortion levels
        for disttypeid in range(ndisttype):
            # Copy test set to distorted test set
            dteimg[diffid, disttypeid] = teimg[disttypeid].copy()
            # Iterate over patterns and distort each
            if disttypeid==0:
                # Rightward mask
                dteimg[diffid, disttypeid, :, :w, :] = teimg[disttypeid, ::-1, :w, :]
            elif disttypeid==1:
                # Leftward mask
                dteimg[diffid, disttypeid, :, -w:, :] = teimg[disttypeid, ::-1, -w:, :]
            elif disttypeid==2:
                # Upward mask
                dteimg[diffid, disttypeid, :, :, :w] = teimg[disttypeid, ::-1, :, :w]
            elif disttypeid==3:
                # Down mask
                dteimg[diffid, disttypeid, :, :, -w:] = teimg[disttypeid, ::-1, :, -w:]
            else:
                print ("Invalid disttypeid!")
            dtelbl[diffid, disttypeid] = telbl[disttypeid].copy() 
    # Store file
    RivDir = f"Rivalry"    
    os.makedirs(f"{RivDir}", exist_ok=True)
    dteimg.astype('float32').tofile(f"{RivDir}/mnist_teimg.bin")
    dtelbl.astype('float32').tofile(f"{RivDir}/mnist_telbl.bin")
    # Do plotting
    fig, axs = plt.subplots(ndisttype, ndiff, figsize=(ndiff*2/3, ndisttype*2/3))
    fig.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.8, hspace=0., wspace=0.)
    plotgrid(fig, axs, dteimg[:,:,0])
    for diffid in range(ndiff):
        axs[-1,diffid].set_xlabel(f"{(diffid+1)/ndiff:.1f}", fontsize=10)
    disttypename = ["top", "bottom", "left", "right"]
    for disttypeid in range(ndisttype):
        axs[disttypeid,0].set_ylabel(f"{disttypename[disttypeid]}", fontsize=10, rotation="horizontal", va="center", ha="right")
    fig.text(0.65, 0.05, r'Difficulty Level', ha='center', fontsize=10)
    fig.text(0.65, 0.90, r'Perceptual Rivalry', ha='center', fontsize=10)
    fig.text(0.1, 0.95, "b", color="black", ha="left", va="top", fontsize=15)       
    plt.savefig(f"mnist.riv.png", dpi=400)
    plt.savefig(f"mnist.riv.svg", format="svg", dpi=400)
    if (SHOW): plt.show(block=False)

def distort(disttypeid, pat, diff):
    
    if disttypeid==0:
        npixdistort = int(diff*28*28/2)
        """ noise: turn pixels to 1 randomly"""
        pixids_shuffled = np.arange(28*28)
        np.random.shuffle(pixids_shuffled)
        pixids_shuffled = pixids_shuffled[:npixdistort]
        for pixid in pixids_shuffled:
            pat[pixid//28,pixid%28] = 1
    # elif disttypeid==2:
    #     npixborder = int(diff*28/4)
    #     """ border: create black border on image """
    #     assert (npixborder<pat.shape[0])
    #     assert (npixborder<pat.shape[1])
    #     if npixborder==0:
    #         return
    #     pat[:npixborder,:] = 1
    #     pat[:,:npixborder] = 1
    #     pat[-npixborder:,:] = 1
    #     pat[:,-npixborder:] = 1
    # elif disttypeid==3:
    #     npatch = int(diff*20)
    #     """ patches: create black patches on image """
    #     pixids_shuffled = np.arange(28*28)
    #     np.random.shuffle(pixids_shuffled)
    #     for patchid in range(npatch):
    #         pix_x = pixids_shuffled[patchid]//28
    #         pix_y = pixids_shuffled[patchid]%28
    #         pix_startx = max(0,pix_x-2)
    #         pix_stopx = min(28, pix_x+2)
    #         pix_starty = max(0,pix_y-2)
    #         pix_stopy = min(28, pix_y+2)
    #         pat[pix_startx:pix_stopx,pix_starty:pix_stopy:] = 1
    elif disttypeid==1:
        nline = int(diff*10)
        """ grid: create regular black grid lines on image """
        if (nline==0):
            return
        ngap = 28//nline
        pat[::ngap,:] = 1
        pat[:,::ngap] = 1
    elif disttypeid==2:
        nline = int(diff*20)
        """ clutter: create irregular black lines (horz/vert) on image """ 
        if (nline==0):
            return
        for lineid in range(nline):
            # decide if line is horz or vert
            orient = np.random.choice(range(2)) 
            # decide on location
            npos = np.random.choice(range(28))
            # draw line
            if orient==0:
                pat[:,npos] = 1
            else:
                pat[npos,:] = 1
    elif disttypeid==3:
        nline = int(diff*20)
        """ deletion: create irregular white lines (horz/vert) on image """ 
        if (nline==0):
            return
        for lineid in range(nline):
            # decide if line is horz or vert
            orient = np.random.choice(range(2)) 
            # decide on location
            npos = np.random.choice(range(28))
            # draw line
            if orient==0:
                pat[:,npos] = 0
            else:
                pat[npos,:] = 0
    elif disttypeid==4:
        """ occlusion """ 
        t = int(diff*7) # thickness
        w = 7 # width
        v = 0.3 # value
        p0 = 14-w//2 # starting position
        p1 = 14+w//2 # ending position
        if (t==0):
            return
        pat[p0-t:p1+t, p0-t:p0] = v
        pat[p0-t:p1+t, p1:p1+t] = v
        pat[p0-t:p0, p0-t:p1+t] = v
        pat[p1:p1+t, p0-t:p1+t] = v
    else:
        print ("Distort type not valid!")

def make_distortion(ndisttype=5, ndiff=5, DistortDir="Distort"):
    # Load original dataset
    teimg = np.fromfile(f"{RawDir}/mnist_teimg.bin", dtype=np.float32).reshape(10000, 28, 28)
    telbl = np.fromfile(f"{RawDir}/mnist_telbl.bin", dtype=np.float32).reshape(10000, 10)
    ntest = len(teimg)
    npat_per_disttype = ntest//ndisttype//ndiff
    print (npat_per_disttype)
    # Create of copy of test image data for distortions
    teimg = teimg[:ndisttype*npat_per_disttype].reshape(ndisttype,npat_per_disttype,28,28)
    dteimg = np.zeros((ndiff,ndisttype,npat_per_disttype,28,28))
    telbl = telbl[:ndisttype*npat_per_disttype].reshape(ndisttype,npat_per_disttype,10)
    dtelbl = np.zeros((ndiff,ndisttype,npat_per_disttype,10))
    # Loop across different difficulty levels
    for diffid in range(ndiff):
        # Loop across different distortion levels
        for disttypeid in range(ndisttype):
            # Copy test set to distorted test set
            dteimg[diffid, disttypeid] = teimg[disttypeid].copy()
            # Iterate over patterns and distort each
            for patid in range(dteimg.shape[2]):
                diff = (diffid+1)/ndiff
                distort(disttypeid, dteimg[diffid, disttypeid, patid], diff)
            dtelbl[diffid, disttypeid] = telbl[disttypeid].copy() 
    # Do plotting
    fig, axs = plt.subplots(ndisttype, ndiff, figsize=(ndiff*2/3, ndisttype*2/3))
    fig.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.8, hspace=0., wspace=0.)
    plotgrid(fig, axs, dteimg[:,:,0])
    for diffid in range(ndiff):
        axs[-1,diffid].set_xlabel(f"{(diffid+1)/ndiff:.1f}", fontsize=10)
    disttypename = ["noise", "grid", "clutter", "deletion", "occlusion"]
    for disttypeid in range(ndisttype):
        axs[disttypeid,0].set_ylabel(disttypename[disttypeid], rotation="horizontal", va="center", ha="right", fontsize=10)
    fig.text(0.65, 0.05, r'Difficulty Level', ha='center', fontsize=10)
    fig.text(0.65, 0.90, r'Distortion Resistance', ha='center', fontsize=10)
    fig.text(0.1, 0.95, "c", color="black", ha="left", va="top", fontsize=15)       
    plt.savefig(f"mnist.dis.png", dpi=400)
    plt.savefig(f"mnist.dis.svg", format="svg", dpi=400)
    if (SHOW): plt.show(block=False)
    # Store file
    os.makedirs(f"{DistortDir}", exist_ok=True)
    dteimg.astype('float32').tofile(f"{DistortDir}/mnist_teimg.bin")
    dtelbl.astype('float32').tofile(f"{DistortDir}/mnist_telbl.bin")

def plotgrid(fig, axs, imgs, cmap="binary", vmin=0, vmax=1):
    for x in range(len(imgs)):
        for y in range(len(imgs[0])):
            axs[y,x].imshow(imgs[x,y].reshape(28,28), cmap=cmap, vmin=vmin, vmax=vmax)
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_linewidth(1)
    return

if __name__ == "__main__":

    # matplotlib params
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size':15})

    SHOW = False
    
    DownloadDir = "Download"
    RawDir = "Raw"
    
    #url = "http://yann.lecun.com/exdb/mnist/" # Original link is broken
    url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
   
    filenames = {"trimg": "train-images-idx3-ubyte.gz",
                 "trlbl": "train-labels-idx1-ubyte.gz",
                 "teimg": "t10k-images-idx3-ubyte.gz",
                 "telbl": "t10k-labels-idx1-ubyte.gz"}
    
    download()

    make_raw()
    make_completion()
    make_rivalry()
    make_distortion()

    if (SHOW): plt.show()
