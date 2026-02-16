import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import parseparam

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})

def stat(arr):

    return f"\tsize:{arr.shape} \tmin:{arr.min():.3f} \tmax:{arr.max():.3f} \tmean:{arr.mean():.3f} \tsum:{arr.sum():.3f} "

def loadbin(datadir, filename, shape, dtype=np.float32):

    dat = np.fromfile(datadir+"/"+filename, dtype=dtype)
    dat = dat.reshape(shape)
    print (f"Loaded \t{filename} {stat(dat)}")
    return dat

def raster():

    for l in range(0, nlayer):

        if l == 0:
            H, M = Hi, Mi
        else:
            H, M = Hh, Mh

        teact = loadbin(datadir, f"predict.teact.l{l}.log", dtype=np.float32, shape=(-1, H * M))
  
        # Plot activity raster
        teact = teact.reshape(-1, H * M)
        plt.imshow(teact[:1000, :500].T, aspect="auto", cmap="binary", interpolation="None")
        plt.colorbar()
        plt.xlabel("Time [ms]")
        plt.ylabel("Neuron id")
        plt.title(f"activity raster, layer{l}")
        plt.savefig(datadir+"/"+f"act.l{l}.png", dpi=200)
        if (SHOW): plt.show()
        plt.close()

        # Plot activity histogram
        plt.hist(teact[:100000].flatten(), bins=100)
        plt.ylim(1, 100000*H*M)
        plt.yscale("log")
        plt.title(f"activity hist, layer{l}")
        plt.savefig(datadir+"/"+f"hist.actl{l}.png", dpi=200)
        if (SHOW): plt.show()
        plt.close()
    
def spkraster():

    import itertools
    
    for l in range(1, nlayer):

        if l == 0:
            H, M = Hi, Mi
        else:
            H, M = Hh, Mh

        nstep_per_pat = param['nstep_per_pat']
        nstep_per_gap = param['nstep_per_gap']
        nstep = nstep_per_pat + nstep_per_gap        

        Nstart = 4*param['Mh'] + 50
        Nshow = 30 # param['Mh']
        Tstart = 120 * nstep
        Tshow = 20 * nstep

        tract = loadbin(datadir, f"predict.tract.l{l}.log", dtype=np.float32, shape=(-1, H * M))
        trsup = loadbin(datadir, f"predict.trsup.l{l}.log", dtype=np.float32, shape=(-1, H * M))
        
        assert Nstart + Nshow < H * M
        assert Tstart + Tshow < len(tract)
        
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

        # Plot mem voltage and spikes
        plt.figure(figsize=(5,3.5))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.90)
        offset = 0
        offset_inc = 0.25
        for neuronid in range(Nshow):
            trsup_per_neuron = trsup[Tstart:Tstart+Tshow, Nstart+neuronid]
            trsup_per_neuron = (trsup_per_neuron - trsup_per_neuron.min()) / (trsup_per_neuron.max() - trsup_per_neuron.min())
            trspk_per_neuron = tract[Tstart:Tstart+Tshow, Nstart+neuronid]
            trsup_per_neuron = trsup_per_neuron*0.25 + trspk_per_neuron*0.75
            plt.plot(offset + trsup_per_neuron, linewidth=0.75, alpha=1, color="black")
            offset = offset + offset_inc
        plt.xlim(0, Tshow)
        plt.xticks(ticks=np.arange(0, Tshow+nstep, step=5*nstep), labels=np.arange(0, Tshow+nstep, step=5*nstep)/1000.)
        plt.xlabel("Time [s]")
        plt.yticks([])
        plt.ylim(0, offset)
        plt.ylabel("")
        plt.title(f"Neuronal support")
        plt.savefig(datadir+"/"+f"l{l}.sup.png", dpi=200)
        plt.savefig(datadir+"/"+f"l{l}.sup.svg", dpi=200, format='svg', transparent=True)
        if (SHOW): plt.show()
        plt.close()

        # Plot firing rate
        Nstart = 4*param['Mh']
        Nshow = param['Mh']
        Tstart = 120 * nstep
        Tshow = 20 * nstep
        kernel_window = 1000 # msec
        gaussian_sigma = 50 # msec
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.90)

        # Create smoothing kernel
        unit_kernel = np.ones(200)/200.
        moving_window = np.linspace(-kernel_window/2, kernel_window/2, kernel_window)
        gaussian_kernel = np.exp(-(moving_window/gaussian_sigma)**2/2) / np.sqrt(2*np.pi*gaussian_sigma**2)

        # Plot firing rate
        for neuronid in range(Nstart, Nstart+Nshow):
            firingrate = 1000. * np.convolve(tract[:, neuronid], gaussian_kernel, mode="full")
            plt.plot(firingrate[Tstart:Tstart+Tshow], label=f"neuron {neuronid}", linewidth=2, alpha=1)

        # Plot stimulus filling area
        stimulus = np.zeros((Tshow//nstep, nstep))
        stimulus[:, 0:nstep_per_pat] = 1
        stimulus = stimulus.flatten()        
        ax.fill_between(range(0, Tshow), param['maxfq']*2, where=stimulus>0.5, facecolor='black', alpha=0.1)

        plt.xlim(0, Tshow)
        plt.xticks(ticks=np.arange(0, Tshow+nstep, step=5*nstep), labels=np.arange(0, Tshow+nstep, step=5*nstep)/1000.)
        plt.xlabel("Time [s]")
        plt.ylim(0, 1.5*param['maxfq'])
        plt.ylabel("spikes/s")
        plt.title(f"Firing rate")
        plt.savefig(datadir+"/"+f"firingrate.l{l}.png", dpi=200)
        plt.savefig(datadir+"/"+f"firingrate.l{l}.svg", dpi=200, format='svg', transparent=True)
        if (SHOW): plt.show()
        plt.close()

def receptive_field(offset=0, Nx=10, Ny=10):

    ACTIVE, SILENT, ABSENT = 1, 0, -1 # as done in c++ code

    all_conns = []

    for layerid in range(1, nlayer):

        if layerid==1:
            conn = loadbin(datadir, f"learn.cij.l{layerid}.bin", dtype=np.int32, shape=(Hh, Hi))
        else:
            conn = loadbin(datadir, f"learn.cij.l{layerid}.bin", dtype=np.int32, shape=(Hh, Hh))

        # receptive field of layer
        conn = (conn==ACTIVE)*1
        all_conns.append(conn)
        rf = all_conns[0]
        for prev_conn in all_conns[1:]:
            rf = prev_conn @ rf
        colorful_conn = rf # > 0 #  * 1.

        fig, axs = plt.subplots(Nx, Ny, figsize=(Ny, Nx))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05, hspace=0.05)
        for x in range(Nx):
            for y in range(Ny):
                hc = y + x * Ny + offset 
                axs[x, y].imshow(colorful_conn[hc].reshape(Ix, Iy), cmap="binary", interpolation="None") # , vmin=colorful_conn.min(), vmax=colorful_conn.max())
                axs[x, y].set_xticks([])
                axs[x, y].set_yticks([])
                # axs[x, y].axis("off")
        plt.suptitle(f"receptive fields, layer{layerid}")
        plt.savefig(datadir+"/"+f"learn.rcpfld.l{layerid}.offset{offset}.png", dpi=200)
        #if (SHOW):
        plt.show()
        plt.close()

        # average coverage
        coverage = colorful_conn.sum(axis=0).reshape(Ix, Iy)
        print (coverage.shape)
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05, hspace=0.05)
        plt.imshow(coverage.reshape(Ix, Iy), cmap="binary", interpolation="None") # , vmin=colorful_conn.min(), vmax=colorful_conn.max())
        #ax.set_xticks([])
        #ax.set_yticks([])
        plt.colorbar()
        plt.suptitle(f"coverage, layer{layerid}")
        plt.savefig(datadir+"/"+f"learn.coverage.l{layerid}.offset{offset}.png", dpi=200)
        #if (SHOW):
        plt.show()
        plt.close()
        
        continue
        
        # max activating input
        import random

        act = loadbin(datadir, f"predict.teact.l{layerid}.log", dtype=np.float32, shape=(-1, 4, Hh, Mh))[:, -1, :, :]
        img = loadbin(datadir, "Data/mnist/Raw/mnist_teimg.bin", dtype=np.float32, shape=(-1, Ix, Iy))[:, :, :]

        for hc in range(Hh):
            
            #plt.imshow(colorful_conn[hc].reshape(28, 28), cmap="binary", interpolation="None")
            #plt.show()
            
            num_okay_mc = 0
            okay_mc = []
            for mc in range(Mh):
                num_okay_mc = okay_mc + ((act[:, hc, mc]>0.5).sum()>=1)*1
                if ((act[:, hc, mc]>0.5).sum()>=1):
                    okay_mc.append(mc)
            #print (hc, mc, num_okay_mc)
            okay_mc = np.asarray(okay_mc)

            fig, axs = plt.subplots(Nx, Ny, figsize=(Ny, Nx))
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05, hspace=0.05)
            for x in range(Nx):
                for y in range(Ny):                        
                    maxid = random.choice(okay_mc)
                    maximg = img[maxid]
                    conn = colorful_conn[hc].reshape(Ix, Iy)
                    masked_maximg = np.ma.masked_where(conn==0, maximg)
                    #print (act[:, hc, mc].min(), act[:, hc, mc].max(), act[maxid, hc, mc])
                    axs[x, y].imshow(masked_maximg, cmap="bwr", interpolation="None", vmin=0, vmax=1) # , vmin=colorful_conn.min(), vmax=colorful_conn.max())
                    axs[x, y].set_xticks([])
                    axs[x, y].set_yticks([])
                    # axs[x, y].axis("off")
                    mc = mc + 1
            #plt.suptitle(f"receptive fields, layer{layerid}\n black: ACTIVE, white: SILENT+ABSENT")
            plt.savefig(datadir+"/"+f"learn.maximg.l{layerid}.{hc}.png", dpi=200)
            #if (SHOW):
            #plt.show()
            plt.close()
        
        # print (act.shape, act.min(), act.max())
    
def traces():

    for layerid in range(1, nlayer):

        if layerid==1:
            Hsrc, Msrc, Nsrc = Hi, Mi, Hi*Mi
            Htrg, Mtrg, Ntrg = Hh, Mh, Hh*Mh
        else:
            Hsrc, Msrc, Nsrc = Hh, Mh, Hh*Mh
            Htrg, Mtrg, Ntrg = Hh, Mh, Hh*Mh
            
        tract = loadbin(datadir, f"predict.tract.l{layerid}.log", dtype=np.float32, shape=(-1, nstep_per_pat, Htrg * Mtrg))
        tract = tract[:, -1, :]
        
        num_dead_minicols = []
        rho = 0.5
        num_dead_minicols = ((tract > rho).sum(axis=0)==0).sum()
        where_dead_minicols = np.argwhere((tract > rho).sum(axis=0)==0).flatten()
        where_alive_minicols = np.argwhere((tract > rho).sum(axis=0)!=0).flatten()

        pi = loadbin(datadir, f"learn.pi.l{layerid}.bin", dtype=np.float32, shape=(-1, Nsrc))
        pj = loadbin(datadir, f"learn.pj.l{layerid}.bin", dtype=np.float32, shape=(-1, Ntrg))
        pij = loadbin(datadir, f"learn.pij.l{layerid}.bin", dtype=np.float32, shape=(-1, Ntrg, Nsrc))
        wij = loadbin(datadir, f"learn.wij.l{layerid}.bin", dtype=np.float32, shape=(-1, Ntrg, Nsrc))

        t_all = [0, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9]

        for t in range(len(pj)):
    
            plt.title(r"$p_{j}$"+f", layer{layerid}, t="+f"{t_all[t]}")
            plt.hist([ pj[t, where_dead_minicols].flatten(), pj[t, where_alive_minicols].flatten()],
                     histtype="step", alpha=0.8, cumulative=False, color=["red", "green"], label=["dead", "alive"], bins=np.linspace(pj.min(), pj.max(), 100))
            plt.yscale("log")
            plt.xlim(pj.min()-1e-2, pj.max()+1e-2)
            plt.ylim(1, Htrg*Mtrg)
            plt.legend()
            plt.savefig(f"layer{layerid}.pj.t{t:03}.png", dpi=200)
            if (SHOW): plt.show()
            plt.close()
            
        for t in range(len(pj)):
            
            plt.title(r"$p_{ij}$"+f", layer{layerid}, t="+f"{t_all[t]}")
            plt.hist([pij[t, where_dead_minicols].flatten(), pij[t, where_alive_minicols].flatten()],
                     histtype="step", alpha=0.5, cumulative=False, color=["red", "green"], label=["dead", "alive"], bins=np.linspace(pij.min(), pij.max(), 100))
            plt.yscale("log")
            plt.xlim(pij.min()-1e-2, pij.max()+1e-2)
            plt.ylim(1, Hsrc*Msrc*Htrg*Mtrg)
            plt.legend()
            plt.savefig(f"layer{layerid}.pij.t{t:03}.png", dpi=200)
            if (SHOW): plt.show()
            plt.close()

        for t in range(len(pj)):
            
            plt.title(r"$w_{ij}$"+f", layer{layerid}, t="+f"{t_all[t]}")
            plt.hist([wij[t, where_dead_minicols].flatten(), wij[t, where_alive_minicols].flatten()],
                     histtype="step", alpha=0.5, cumulative=False, color=["red", "green"], label=["dead", "alive"], bins=np.linspace(wij.min(), wij.max(), 100))
            plt.yscale("log")
            plt.xlim(wij.min()-1e-2, wij.max()+1e-2)
            plt.ylim(1, Hsrc*Msrc*Htrg*Mtrg)
            plt.legend()
            plt.savefig(f"layer{layerid}.wij.t{t:03}.png", dpi=200)
            if (SHOW): plt.show()
            plt.close()
        
    return

def filters(offset=0, Nx=10, Ny=10):

    for layerid in range(1, nlayer):

        if layerid==1:
            Hsrc, Msrc, Nsrc = Hi, Mi, Hi*Mi
            Htrg, Mtrg, Ntrg = Hh, Mh, Hh*Mh
        else:
            Hsrc, Msrc, Nsrc = Hh, Mh, Hh*Mh
            Htrg, Mtrg, Ntrg = Hh, Mh, Hh*Mh
            
        cij = loadbin(datadir, f"init.cij.l{layerid}.bin", dtype=np.int32, shape=(Htrg, Hsrc))        
        wij = loadbin(datadir, f"init.wij.l{layerid}.bin", dtype=np.float32, shape=(Htrg, Mtrg, Hsrc, Msrc))
        wij = np.einsum('xy,xayb->xayb', cij, wij)
        wij = wij.reshape(Htrg, Mtrg, 28, 28, Msrc)

        fig, axs = plt.subplots(2, 16, figsize=(16, 2))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05, hspace=0.05)
        hs = [4, 13, 12, 8]
        for x in range(2):
            for y in range(16):
                h = hs[y//4]
                m = y%4 + 4*x
                w = wij[h, m, :, :, 0]
                axs[x, y].imshow(w, cmap="bwr", interpolation="None", vmin=-w.max(), vmax=w.max())
                axs[x, y].set_xticklabels([])
                axs[x, y].set_yticklabels([])
        plt.savefig("init.filter.png", dpi=200)
        # plt.savefig("init.filter.svg", dpi=200, format='svg')
        plt.close()
        
        cij = loadbin(datadir, f"learn.cij.l{layerid}.bin", dtype=np.int32, shape=(Htrg, Hsrc))
        wij = loadbin(datadir, f"learn.wij.l{layerid}.bin", dtype=np.float32, shape=(Htrg, Mtrg, 28, 28, Msrc))

        fig, axs = plt.subplots(2, 16, figsize=(16, 2))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05, hspace=0.05)
        hs = [4, 13, 12, 8]
        for x in range(2):
            for y in range(16):
                h = hs[y//4]
                m = y%4 + 4*x
                w = wij[h, m, :, :, 0]
                axs[x, y].imshow(w, cmap="bwr", interpolation="None", vmin=-w.max(), vmax=w.max())
                axs[x, y].set_xticklabels([])
                axs[x, y].set_yticklabels([])
        plt.savefig("learn.filter.png", dpi=200)
        # plt.savefig("learn.filter.svg", dpi=200, format='svg')
        plt.close()
        
    return

def simmat(offset=0):

    from sklearn.metrics.pairwise import cosine_similarity

    trnpat, tenpat = 1000, 1000
        
    # Load label dataset
    trlbl = loadbin(datadir, "Data/mnist/mnist_trlbl.bin", dtype=np.float32, shape=(-1, 10))
    trlbl = trlbl[:trnpat].argmax(axis=1)
    sortid = np.argsort(trlbl)
    trlbl = trlbl[sortid]
    
    for layerid in range(nlayer):

        if layerid == 0:
            H, M, N = Hi, Mi, Hi*Mi
        else:
            H, M, N = Hh, Mh, Hh*Mh
    
        tract = loadbin(datadir, f"predict.tract.l{layerid}.log", dtype=np.float32, shape=(-1, N))
        tract = tract[offset::nstep_per_pat+nstep_per_gap]
        tract = tract[sortid]

        simmat = cosine_similarity(tract)

        plt.imshow(simmat, cmap="bwr", vmin=0, vmax=1)
        plt.title(f"simmat, layer{layerid}, cosine")
        plt.colorbar()
        plt.savefig(datadir+"/"+f"simmat.l{layerid}.png", dpi=200)
        if (SHOW): plt.show()
        plt.close()

def show_energy():

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.8, wspace=0.3, hspace=0.3)

    for layerid in range(nlayer):

        if layerid == 0:
            H, M, N = Hi, Mi, Hi*Mi
        else:
            H, M, N = Hh, Mh, Hh*Mh

        binsize = 100
        
        energy = loadbin(datadir, f"energy.l{layerid}.log", dtype=np.float32, shape=(-1, binsize))

        axs[layerid].errorbar(x=range(len(energy)), y=energy.mean(axis=1), yerr=energy.std(axis=1), fmt="ro-")
        
        axs[layerid].set_xlabel(f"Iteration (binsize = {binsize})")
        axs[layerid].set_ylabel("Energy")
        axs[layerid].set_ylim(-0.15, 0.15)
        # axs[layerid].text(0.5, H*M*0.95, 'HxM', color="red")
        axs[layerid].set_title(f"Layer {layerid}")

    plt.suptitle("Energy")
    plt.savefig("energy.png", dpi=200)
    plt.show()

def show_usage():

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.8, wspace=0.3, hspace=0.3)
    
    for layerid in range(nlayer):

        H, M = param[f'H{layerid}'], param[f'M{layerid}']
        if layerid != 0: nconn = param[f'nconn{layerid}']

        tract = loadbin(datadir, f"predict.tract.l{layerid}.log", dtype=np.float32, shape=(-1, nstep_per_pat, H * M))
        tract = tract[:, -1, :]
        teact = loadbin(datadir, f"predict.teact.l{layerid}.log", dtype=np.float32, shape=(-1, nstep_per_pat, H * M))
        teact = teact[:, -1, :]
        
        rho_all = np.linspace(-0.01, 1.01, 100)
        num_dead_minicols = []
        for rho in rho_all:
            num = ((tract > rho).sum(axis=0)==0).sum()
            num_dead_minicols.append(num)
        axs[layerid].plot(rho_all, num_dead_minicols, "o-", color="gray", linewidth=1.5, label="train")
        num_dead_minicols = []
        for rho in rho_all:
            num = ((teact > rho).sum(axis=0)==0).sum()
            num_dead_minicols.append(num)
        axs[layerid].plot(rho_all, num_dead_minicols, "o-", color="black", linewidth=1.5, label="test")

        axs[layerid].legend(loc="center left")
        axs[layerid].set_xlabel(r"$\rho$")
        axs[layerid].set_ylabel("# dead units")
        axs[layerid].set_ylim(0, H*M*1.05)
        axs[layerid].hlines(H*M, xmin=0, xmax=1, color="red")
        axs[layerid].text(0.5, H*M*0.95, 'HxM', color="red")
        if layerid == 0:
            axs[layerid].set_title(f"Layer {layerid} "+f"{H}x{M}")
        else:
            axs[layerid].set_title(f"Layer {layerid} "+f"{H}x{M}@{nconn}")
        plt.suptitle("How many minicols ever cross "+r"$\rho$ "+f"during inference?\n" + \
                 r"MNIST, $N_{trpat}$=" +f"{len(tract)}, "+r" $N_{tepat}$=" + f"{len(teact)} "+r"$\tau_p$= "+f"{param['tau_p']}")
    filename = f"usage.{param['H1']}x{param['M1']}@{param['nconn1']:02d}-{param['H2']}x{param['M2']}@{param['nconn2']:02d}.png"
    plt.savefig(filename, dpi=200)
    print (filename)
    if (SHOW): plt.show()
    plt.close()

def parse_acc():
    logfilename = f"{datadir}/test.out"
    f = open(logfilename, "r")
    l1acc, l2acc = {}, {}
    for line in f:
        words = line.split()
        if len(words)>=10:
            if words[0]=="Layer" and words[1]=="1":
                l1acc["train"], l1acc["test"] = float(words[5]), float(words[10])
            if words[0]=="Layer" and words[1]=="2":
                l2acc["train"], l2acc["test"] = float(words[5]), float(words[10])
    return l1acc, l2acc

def barplot_acc():
            
    df["nconn1"].astype('int')
    df["nconn2"].astype('int')    

    df_sort = df.sort_values(["tau_p", "nconn1", "nconn2"]) 

    bar_width = 0.8
    fig, ax = plt.subplots(figsize=(14,6))
    offset = 0
    colors = ["red", "green", "blue"]
    for idx_tau_p, tau_p in enumerate(np.sort(df["tau_p"].unique())):
        df_tau_p = df[(df["tau_p"] == tau_p)]
        for idx_nconn1, nconn1 in enumerate(np.sort(df_tau_p["nconn1"].unique())):
            df_nconn1 = df_tau_p[(df_tau_p["nconn1"]==nconn1)]
            bars = ax.bar(offset, df_nconn1['l1_te_acc'], bar_width, label=tau_p, color="gray")
            ax.text(offset, 30, f"nconn1={nconn1}", color='white', ha='center', va='center', rotation=90, fontsize=12) 
            offset += 1
            for idx_nconn2, nconn2 in enumerate(np.sort(df_nconn1["nconn2"].unique())):
                df_nconn2 = df_nconn1[(df_nconn1["nconn2"]==nconn2)]
                bars = ax.bar(offset, df_nconn2['l2_te_acc'], bar_width, color=colors[idx_nconn2])
                ax.text(offset, 30, f"nconn2={nconn2}", color='white', ha='center', va='center', rotation=90, fontsize=12) 
                offset += 1
            offset += 1
        offset += 1
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 100)
    plt.show()
    
if __name__ == "__main__":

    SHOW = 1

    resultsdir = "results-emnist"

    df = pd.DataFrame()    

    # for paramdir in os.listdir(resultsdir):

    # for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):

    datadir = "." # f"{resultsdir}/{paramdir}/{timestampdir}"
    paramfilename = "apps/reprlearn/reprlearn.par"# "datadir+"/net.par"
            
    param = parseparam(paramfilename)
    
    Hi = param['H0']
    Mi = param['M0']
    Hh = param['H1']
    Mh = param['M1']
    nconn1 = param['nconn1']
    nconn2 = param['nconn2']
    nstep_per_pat = param["nstep_per_pat"]
    nlayer = 4 # TODO: this needs to be in par file
    taup = param['tau_p']
    
    # Plotting dimensions
    Ix, Iy = 28, 28
    Nx, Ny = 7, 7
    
    # raster()
    
    # traces()
    
    # simmat(offset=nstep_per_pat+nstep_per_gap-1)
    
    # Experiment 4.2.1
    # spkraster()
    
    # Experiment 4.2.2
    for offset in range(0,Hh,100):
        receptive_field(offset=offset, Nx=Nx, Ny=Ny)
    
    # for offset in range(0,Hh,100):
    #    filters(offset=offset, Nx=Nx, Ny=Ny)
    
    # show_energy()
    
    # show_usage()
    
    # l1acc, l2acc = parse_acc()
    
    #new_row = {'nconn1': param['nconn1'],
    #           'nconn2': param['nconn2'],
    #           'tau_p': param['tau_p'],
    #           'l1_tr_acc': l1acc["train"],
    #           'l1_te_acc': l1acc["test"],
    #           'l2_tr_acc': l2acc["train"],
    #           'l2_te_acc': l2acc["test"],
    #}
    
    # df = df.append(new_row, ignore_index=True)                

    # barplot_acc()
