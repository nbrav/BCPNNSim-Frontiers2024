import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import parseparam, loadbin
import os
import math
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def getpopinfo(popid):
    if popid==0:
        H = param["Hin"]
        M = param["Min"]
        popname = r"${INP}$"
    elif popid==1:
        H = param["Hhid"]
        M = param["Mhid"]
        popname = r"${HID}$"
    elif popid == 2:
        H = param["Hin"]
        M = param["Min"]
        popname = r"${INPRC}$ "
    else:
        print ("Invalid popid!")
    return H, M, popname

def getprjinfo(prjid):
    if prjid == "01":
        Hi, Mi = param["Hin"], param["Min"]
        Hj, Mj = param["Hhid"], param["Mhid"]
        prjname = r"$INP \rightarrow HID$ (feedforward)"
    elif prjid == "11":
        Hi, Mi = param["Hhid"], param["Mhid"]
        Hj, Mj = param["Hhid"], param["Mhid"]
        prjname = r"$HID \rightarrow HID$ (recurrent)"
    elif prjid == "12":
        Hi, Mi = param["Hhid"], param["Mhid"]
        Hj, Mj = param["Hin"], param["Min"]
        prjname = r"$HID \rightarrow INP$ (feedback)"
    else:
        print ("Invalid prjid!")
    return Hi, Mi, Hj, Mj, prjname

# def spkraster():
#     """ Plot the raster and firing rate of each population """    
#     Hshow = 4        
#     Tstart = 0 # 2 * nstep
#     Tshow = 3 * nstep
#     # Setup smoothing operation
#     kernel_window = 1000 # msec
#     gaussian_sigma = 20 # msec
#     moving_window = np.linspace(-kernel_window/2, kernel_window/2, kernel_window)
#     gaussian_kernel = np.exp(-(moving_window/gaussian_sigma)**2/2) / np.sqrt(2*np.pi*gaussian_sigma**2)
#     # Setup plots
#     fig, axs = plt.subplots(Hshow, 3, figsize=(9, 6))
#     plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)
#     # Fix the ticks
#     for ax in axs.flatten():
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.spines[['right', 'top', 'left', 'bottom']].set_linewidth(1)
#     # Set axis limits
#     for ax in axs.flatten():
#         ax.set_xlim(0, Tshow)
#         ax.set_ylim(0, 1)
#     # Iterate over all poopulations
#     for layer in range(0, 3):  
#         H, M, layername = getpopinfo(layer)
#         Hstart = H//3 #np.random.randint(H//3, H*2//3)
#         colors = plt.cm.jet(np.linspace(0,1,M))
#         # Load activity
#         spkact = loadbin(datadir=datadir, 
#                          filename=f"everystep.test.act.{layer}.bin", 
#                          dtype=np.float32, 
#                          shape=(-1, H, M), 
#                          verbose=1)
#         # Plot the spike rasters
#         axid = 0 
#         for h in range(Hstart, Hstart+Hshow):
#             # Iterate over all minicols and find spike timing
#             positions = []
#             for m in range(M): 
#                 # Find where there's a spike
#                 spktimes = np.argwhere(spkact[Tstart:Tstart+Tshow,h,m]).flatten() 
#                 positions.append(spktimes)
#             # Make event plot
#             axs[axid,layer].eventplot(positions, 
#                                       linewidth=1, # 2
#                                       lineoffsets=np.linspace(0,1,M+2)[1:-1], 
#                                       linelength=0.1, 
#                                       alpha=0.75,
#                                       colors=colors)  
#             axid += 1
#         # Plot firing rate
#         axid = 0
#         for h in range(Hstart, Hstart+Hshow):
#             for m in range(0, M):
#                 firingrate = np.convolve(spkact[:, h, m], gaussian_kernel, mode="same")
#                 firingrate *= 0.7 * 1000. / param['maxfq'] 
#                 axs[axid,layer].plot(firingrate[Tstart:Tstart+Tshow], 
#                                      linewidth=2, 
#                                      alpha=0.75, 
#                                      color=colors[m%M])
#             axid += 1
#         # Plot stimulus filling area
#         stimulus = np.zeros((Tshow//nstep, nstep)) # gap period
#         stimulus[:, nstep_gap:nstep_gap+nstep_pat] = 1 # pattern  duration
#         stimulus = stimulus.flatten()
#         for ax in axs[:,layer]:        
#             ax.fill_between(range(0, Tshow), y1=M+0.5, y2=-0.5, where=stimulus>0.5, facecolor='black', alpha=0.1)
#         # Set plot text
#         axs[0,layer].set_title(f"{layername}")
#     # Set ticks
#     axs[-1,0].set_xticks(ticks=np.arange(0, Tshow+nstep, step=nstep), labels=np.arange(0, Tshow+nstep, step=nstep))
#     axs[-1,0].set_xlabel("Time (ms)")
#     axs[-1,0].set_yticks(ticks=[0, 0.7], labels=[0, param['maxfq']])
#     axs[-1,0].set_ylabel("Firing rate (Hz)")
#     # Finalize plots
#     #plt.suptitle(f"Spike Raster")
#     plt.savefig(f"exp1a.png", dpi=400)
#     plt.savefig(f"exp1a.svg", format="svg", dpi=400)
#     if (SHOW): plt.show(block=True)
#     plt.close()

# def simmat():

#     from sklearn.metrics.pairwise import cosine_similarity

#     # Number of patterns to load and show for simmilarity
#     npat = 1000 
#     cmap = "jet"
    
#     # Load labels for later sorting
#     N = param['Hout'] * param['Mout']
#     telbl = loadbin(param['datadir'], param['telblfile'], dtype=np.float32, count=npat*N, shape=(-1, N), verbose=1)
#     telbl = telbl[:npat].argmax(axis=1)
#     sortid = np.argsort(telbl)
#     telbl = telbl[sortid]

#     # Start plots
#     fig, axs = plt.subplots(2, 3, figsize=(9, 6))
#     plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.85, wspace=0.15, hspace=0.15)

#     for ax in axs.flatten():
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.spines[['right', 'top', 'left', 'bottom']].set_linewidth(1)

#     # Print pop name on top
#     pop_name = [r"$INP$", r"$HID$",r"$INRC$"]
#     for colid, ax in enumerate(axs[0]):
#         axs[0,0].text(0.5, 1.3, 
#                       pop_name[colid], 
#                       transform=ax.transAxes, 
#                       color="black", 
#                       horizontalalignment="center", 
#                       verticalalignment="center",
#                       fontsize=20
#                       )

#     # Simmat for INP 
#     N = param['Hin'] * param['Min']
#     act = loadbin(datadir, f"predict.ffwd.test.zi.01.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
#     act = act[sortid] 
#     simmat = cosine_similarity(act)
#     axs[0,0].set_title(r"$T$=100ms", fontsize=15)
#     im = axs[0,0].imshow(simmat, cmap=cmap, vmin=0, vmax=1)
    
#     # Remove one plot
#     axs[1,0].axis('off')

#     # Simmat for HID (T=100)
#     N = param['Hhid'] * param['Mhid']
#     act = loadbin(datadir, f"predict.ffwd.test.zj.11.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
#     act = act[sortid] 
#     simmat = cosine_similarity(act)
#     axs[0,1].set_title(r"$T$=100ms", fontsize=15)
#     im = axs[0,1].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

#     # Simmat for HID (T=200)
#     N = param['Hhid'] * param['Mhid']
#     act = loadbin(datadir, f"predict.attractor.test.zj.11.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
#     act = act[sortid] 
#     simmat = cosine_similarity(act)
#     axs[1,1].set_title(r"$T$=200ms", fontsize=15)
#     im = axs[1,1].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

#     # Simmat for INPRC (T=100)
#     N = param['Hin'] * param['Min']
#     act = loadbin(datadir, f"predict.ffwd.test.zj.12.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
#     act = act[sortid] 
#     simmat = cosine_similarity(act)
#     axs[0,2].set_title(r"$T$=100ms", fontsize=15)
#     im = axs[0,2].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

#     # Simmat for INPRC (T=200)
#     N = param['Hin'] * param['Min']
#     act = loadbin(datadir, f"predict.attractor.test.zj.12.bin", dtype=np.float32, count=npat*N, shape=(npat, N), verbose=True)
#     act = act[sortid] 
#     simmat = cosine_similarity(act)
#     axs[1,2].set_title(r"$T$=200ms", fontsize=15)
#     im = axs[1,2].imshow(simmat, cmap=cmap, vmin=0, vmax=1)

#     # Compute label info and set ticks
#     labelname = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
#     labelstartid = np.zeros(11)
#     labelmiddleid = np.zeros(10)
#     for labelid in range(10):
#         labelstartid[labelid] = np.argwhere(telbl==labelid).flatten()[0]
#     labelstartid[-1] = npat
#     for labelid in range(1,11):
#         labelmiddleid[labelid-1] = labelstartid[labelid-1] + (labelstartid[labelid] - labelstartid[labelid-1])//2
#     axs[0,0].set_yticks(labelmiddleid)
#     axs[0,0].set_yticklabels(labelname)
#     axs[0,0].set_xlabel("Pattern Index (sorted)")

#     # Colorbars
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5]) # [left, bottom, width, height] 
#     fig.colorbar(im, cax=cbar_ax, label="Similarity")

#     # plt.suptitle("Representational Similarity")
#     plt.savefig("exp1b.png", dpi=400)
#     plt.savefig("exp1b.svg", format='svg', dpi=400)
#     if (SHOW): plt.show(block=True)
#     plt.close()

# def plot_rcpfld(nrow, ncol, iters, conn, Ix, Iy, title, prjid):
#     fig, axs = plt.subplots(nrow, ncol, figsize=(ncol/2, nrow/2))
#     plt.subplots_adjust(left=0.19, right=0.99, bottom=0.1, top=0.9, wspace=0., hspace=0.)
#     for row in range(nrow):
#         for col in range(ncol):
#             probe = row
#             hc = col
#             axs[row, col].imshow(conn[probe, hc].reshape(Ix, Iy), cmap="binary", interpolation="None")
#             axs[row, col].set_xticks([])
#             axs[row, col].set_yticks([])
#     fig.text(0.03, 0.5, 'Training (log steps)', va='bottom', ha='center', rotation='vertical')
#     for row in range(nrow):
#         axs[row, 0].set_ylabel(f"{iters[row]:d}", rotation=0, ha='right', va='center')
#     plt.suptitle(f"{title}")
#     plt.savefig(f"exp2a.{prjid}.png", dpi=400)
#     plt.savefig(f"exp2a.{prjid}.svg", format='svg', dpi=400)
#     if (SHOW): plt.show(block=True)
#     plt.close()

# def rcpfld_formation(Ix=28, Iy=28, num_hypercol_show=10):
#     print ("exp2a. rcpfld_formation")
#     ACTIVE, SILENT = 1, 0
#     probes = [numer*np.power(10, expon) for expon in range(10) for numer in range(1,10) ]
#     probes = np.array(probes)
#     probeid = [0, 4, 9, 13, 18, 22, ]# 27, 31, 36, 40, 45, 49, 54] # gets nice probes like 1000, 5000, 10000, etc. Shpuld be a better way though
#     probes = probes[probeid] 
#     print (probes)
#     for popid in range(1, param["nlayer"]):
#         # Feedforward receptive field
#         prjid = "01"
#         Hi, _, Hj, _, prjname = getprjinfo(prjid)
#         print (popid, Hi, Hj, prjname, prjid)
#         conn = loadbin(datadir, f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi))
#         conn = conn[probeid]
#         print (conn.shape)
#         plot_rcpfld(nrow=conn.shape[0], ncol=num_hypercol_show, iters=probes, conn=conn, Ix=Ix, Iy=Iy, title=prjname, prjid=prjid)
#         # Feedback receptive field
#         prjid = "12"
#         Hi, _, Hj, _, prjname = getprjinfo(prjid)
#         print (popid, Hi, Hj, prjname, prjid)
#         conn = loadbin(datadir, f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi))
#         conn = conn.transpose(0, 2, 1)
#         conn = conn[probeid]
#         print (conn.shape)
#         plot_rcpfld(nrow=conn.shape[0], ncol=num_hypercol_show, iters=probes, conn=conn, Ix=Ix, Iy=Iy, title=prjname, prjid=prjid)
#     return

# def rewiring_scores():
#     print ("exp2b. rewiring_scores")
#     # Create x-ticks
#     probes = [numer*np.power(10, expon) for expon in range(10) for numer in range(1,10) ]
#     probes = np.asarray(probes)
#     # Setup plots
#     fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
#     plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.1)
#     colid = 0
#     for prjid in ["01", "12"]: # 11
#         Hi, Mi, Hj, Mj, prjname = getprjinfo(prjid)
#         rowid = 0
#         for field in ["nswap", "nmi"]:
#             if field in ["nmi"]:
#                 cij = loadbin(datadir, filename=f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
#                 dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.float32, shape=(-1, Hj, Hi), verbose=1)
#                 dat = np.multiply(dat, cij)
#                 dat = dat.reshape(-1,Hj,Hi).sum(axis=2)
#             elif field in ["nswap"]:
#                 dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.int32, shape=(-1, Hj), verbose=1)
#                 dat = dat[1:] # first element is just zero
#             elif field in ["del_cij"]:
#                 cij = loadbin(datadir, filename=f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=1)
#                 dat = cij[1:] - cij[:-1]
#                 dat = dat.reshape(-1, Hj, Hi)
#                 dat = np.abs(dat).sum(axis=2)
#                 dat = dat / 2. # two conn changes is one swap
#             elif field in ["mean_del_pij"]:
#                 dat = loadbin(datadir, filename=f"learn.{field}.{prjid}.bin", dtype=np.float32, shape=(-1, 1), verbose=1)
#             # Create x- and y-lines
#             y_mean = dat.mean(axis=1)[22:]
#             y_std = dat.std(axis=1)[22:]
#             x = probes[22:][:len(y_mean)] #np.arange(len(y_mean)) # probes
#             # Plot stuff
#             ax = axs[rowid,colid]
#             ax.plot(x, y_mean, "ro-", linewidth=1.5)
#             ax.fill_between(x=x, y1=y_mean-y_std, y2=y_mean+y_std, alpha=0.25, color="red")
#             # Set x- and y-ticks and grid
#             axs[0,colid].set_title(f"{prjname}")
#             ax.set_xlim(400)
#             #ax.set_ylim(-0.1*np.max(dat), 1.1*np.max(dat))
#             #ax.set_ylim(-0.1*np.max(dat[23:]), 1.1*np.max(dat[23:]))
#             ax.set_ylim(0)
#             ax.set_xscale('log')
#             ax.grid(which='both')
#             ax.minorticks_on()
#             if field=="nswap":
#                 axs[rowid,0].set_ylabel(f"Num. of flips")
#             if field=="nmi":
#                 axs[rowid,0].set_ylabel(f"Usage score")                
#             rowid += 1
#         colid += 1
#     axs[-1, 0].set_xlabel("Training samples")
#     plt.savefig("exp2b.png", dpi=400)
#     plt.savefig("exp2b.svg", format='svg', dpi=400)
#     if (SHOW): plt.show(block=True)
#     plt.close()

# def filters():
#     print ("exp. filters")
#     Ix, Iy = 28, 28
#     fig, axs = plt.subplots(3, 3, figsize=(9, 9))
#     plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
#     timezone = -1
#     colid = 0
#     for prjid in ["01", "11", "12"]: 
#         Hi, Mi, Hj, Mj, prjname = getprjinfo(prjid)
#         c = loadbin(datadir, f"learn.cij.{prjid}.bin", dtype=np.int32, shape=(-1, Hj, Hi), verbose=True)
#         w = loadbin(datadir, f"learn.wij.{prjid}.bin", dtype=np.float32, shape=(-1, Hj*Mj, Hi*Mi), verbose=True)
#         wmax = np.max(np.abs(w))
#         axs[0,colid].imshow(c[timezone].T, aspect="auto", interpolation=None, cmap="binary", vmin=0, vmax=1)
#         axs[0,colid].set_title(f"{prjname} conns")
#         axs[0,0].set_ylabel(f"conn")
#         im = axs[1,colid].imshow(w[timezone].T, aspect="auto", interpolation=None, cmap="bwr", vmin=-wmax*0.25, vmax=wmax*0.25)
#         axs[1,0].set_ylabel(f"weight")
#         #fig.colorbar(im, ax=axs[1,colid])
#         axs[2,colid].hist(w[timezone].flatten(), bins=np.linspace(-wmax,wmax,200))
#         axs[2,colid].set_yscale("log")
#         axs[2,0].set_ylabel(f"w hist")
#         axs[0,colid].set_title(f"{prjname}")
#         colid += 1
#     plt.savefig("exp2c.png", dpi=300)
#     plt.savefig("exp2c.svg", format='svg', dpi=300)
#     if (SHOW): plt.show(block=True)
#     plt.close()
#     return

def parameter_explore():   
    import matplotlib.colors as mcolors
    df = pd.DataFrame()    
    resultdir = "results-hidassospk"
    for paramdir in os.listdir(resultdir):
        for seeddir in os.listdir(f"{resultdir}/{paramdir}"):
            tracc, teacc = parse_accuracy(f"{resultdir}/{paramdir}/{seeddir}/out.txt", "LSGD")
            paramfilename = f"{resultdir}/{paramdir}/{seeddir}/net.par"
            param = parseparam(paramfilename)
            new_row = {'maxfq': param['maxfq'],
                        'taum': param['taum'],
                        'tauz': param['tauz'],
                        'tracc': tracc,
                        'teacc': teacc 
            }
            df = df._append(new_row, ignore_index=True)
    df_new = df.sort_values(by=['maxfq', 'taum', 'tauz'])
    print (df_new)
    # Start plotting
    fig, axs = plt.subplots(2, 3, figsize=(6, 3.2), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.1, hspace=0.3)
    axid = 0
    for maxfq in sorted(df["maxfq"].unique(), reverse=True):
        ax = axs.flatten()[axid]
        all_taum = np.array(sorted(df["taum"].unique()))
        all_tauz = np.array(sorted(df["tauz"].unique()))
        all_teacc = df.loc[(df['maxfq']==maxfq)].sort_values(by=['taum', 'tauz'])["teacc"].to_numpy()
        all_teacc = all_teacc.reshape(len(all_taum), len(all_tauz))
        im = ax.imshow(all_teacc, vmin=0, vmax=100, cmap="jet")#, aspect="auto")
        ax.set_title(r"$f_{max}$="+f"{int(maxfq)} Hz", fontsize=12)
        axid += 1
    # Colorbars
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5]) # [left, bottom, width, height] 
    fig.colorbar(im, cax=cbar_ax, label="Accuracy (%)")
    # Labels
    axs[-1,0].set_xlabel(r"$\tau_z$ (ms)", fontsize=12)
    axs[-1,0].set_ylabel(r"$\tau_m$ (ms)", fontsize=12)
    # Ticks
    for ax in axs.flatten():
        ax.set_xticks(range(len(all_tauz)), [int(1000*x) for x in np.array(sorted(df["tauz"].unique()))], fontsize=8)
        ax.set_yticks(range(len(all_taum)), [int(1000*x) for x in np.array(sorted(df["taum"].unique()))], fontsize=8)
    plt.savefig("exp3.png", dpi=300)
    plt.savefig("exp3.svg", format="svg", dpi=300)

# def find_ptype(simmat, s_min = 0.1):
#     # Iterate to find prototype ids
#     ptyp_ids = []
#     pats_per_ptyp = {} # {key: value} = {prototype id: all pat id converging on prototype}
#     for pat_id in range(len(simmat)):
#         found_match = False
#         # Find closest prototype across all prototypes 
#         ptyp_ids_arr = np.asarray(ptyp_ids)
#         sim_to_ptyps = simmat[pat_id,ptyp_ids]
#         if (len(ptyp_ids)==0): # add a prototype if we're getting started
#             ptyp_ids.append(pat_id)
#         else: # check for closest prototype
#             closest_ptype_s = max(sim_to_ptyps)
#             closest_ptype_id = np.argwhere(sim_to_ptyps==closest_ptype_s)
#             if (closest_ptype_s > s_min):
#                 found_match = True
#             else:
#                 found_match = False
#             if found_match ==False:
#                 ptyp_ids.append(pat_id)                             
#     return ptyp_ids

# def plot_ptype():
#     # Start plots
#     nrow = 10
#     ncol = math.ceil(len(ptyps)/10)
#     fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
#     plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.85, wspace=0.1, hspace=0.1)
#     # Do plotting
#     for imgid in range(len(ptyps)):
#         ptyp_id = ptyps[imgid]
#         ax = axs.flatten()[imgid]
#         print (imgid, ptyp_id, ax)
#         ax.imshow(z_inprc[ptyp_id].reshape(28,28,2)[:,:,0], vmin=0, vmax=1, cmap="binary")
#     for ax in axs.flatten():
#         ax.set_xticks([])
#         ax.set_yticks([])
#     # Find top middle axis
#     if len(axs.shape)==1:
#         ax = axs[0]
#     else:
#         ax = axs[0,ncol//2]
#     fig.text(0.5, 1.2, r"$S_{min}=$"+f"{s_min}", fontsize=35, transform=ax.transAxes, ha='center')
#     plt.savefig(f"ptype.smin{s_min:1.2f}.png", dpi=400)
#     plt.show(block=True)

# def prototype():
    
#     print ("exp6. prototype")
#     from sklearn.metrics.pairwise import manhattan_distances, cosine_similarity
    
#     # Set params
#     npat = 1000 # numbers of patterns to use for simmat
#     s_min_all = np.arange(0, 1, 0.1) # find prototypes for these s_min's
#     s_min_plot = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] # plot these prototypes

#     # Load z-traces    
#     z_inp = loadbin(datadir, f"predict.attractor.test.zi.01.bin", dtype=np.float32, shape=(-1, param["Hin"]*param["Min"]), verbose=1)
#     z_hid = loadbin(datadir, f"predict.attractor.test.zj.11.bin", dtype=np.float32, shape=(-1, param["Hhid"]*param["Mhid"]), verbose=1)
#     z_inprc = loadbin(datadir, f"predict.attractor.test.zj.12.bin", dtype=np.float32, shape=(-1, param["Hin"]*param["Min"]), verbose=1)
    
#     # Load labels
#     telbl_1hot = loadbin(param['datadir'], param['telblfile'], dtype=np.float32, shape=(-1, param["Hout"]*param["Mout"]), verbose=1)
#     telbl_1hot = telbl_1hot[:npat]
#     telbl = telbl_1hot.argmax(axis=1)
#     sortid = np.argsort(telbl)
    
#     # Sort them by labels
#     telbl = telbl[sortid]
#     z_inp = z_inp[sortid]
#     z_hid = z_hid[sortid]
#     z_inprc = z_inprc[sortid]

#     # Compute simmat
#     simmat = cosine_similarity(z_hid, z_hid)
    
#     # Iterate to find prototype ids
#     nptyp_per_s_min = []
#     for s_min in s_min_all:
#         # Call and find ptypes
#         ptyps = find_ptype(simmat, s_min)
#         print (s_min, len(ptyps), telbl[ptyps])
#         nptyp_per_s_min.append(len(ptyps))
#         # Plot ptypes for some s_min 
#         if s_min in s_min_plot:
#             plot_ptyp(z_inprc, ptyps)
            
#     # Plot num. of prototype vs s_min
#     fig, ax = plt.subplots(1, 1, figsize=(6, 3))
#     plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.95, wspace=0.1, hspace=0.1)
#     ax.plot(np.arange(0, 1, 0.01), nptyp_per_s_min, "-", color="blue")
#     ax.set_xlabel(r"$S_{min}$")
#     ax.set_ylabel("Num. of prototypes")
#     plt.savefig("exp6-nptyp.png", dpi=400)
#     plt.close()

# def attractor_reconstruct():
#     # Set params
#     npat = 10000 
#     npatshow = 10 # number of patterns to show
#     timestep = 0.001 # ms
#     nzone = 10 # number of time zones to show (< nstep_pat)
#     nstep_zone = (nstep_pat+nstep_gap)//nzone # param["nstep_log"] # ms

#     for field in [ "zi.01","zj.12"]:
#         H, M = param["Hin"], param["Min"]
#         filename = f"everystep.test.{field}.bin"
#         # Load data files
#         print (datadir, filename)
#         dat = loadbin(datadir, 
#                       filename, 
#                       dtype=np.float32, 
#                       offset=(0*nstep*H*M*np.dtype(np.float32).itemsize),
#                       count=(npatshow*nzone*nstep_zone*H*M),
#                       shape=(npatshow,nzone,nstep_zone,H,M),
#                       verbose=0)
#         print (dat.shape)
#         dat = dat[:,:,-1,:,0].reshape(npatshow,nzone,28,28)
#         # Do plots
#         fig, axs = plt.subplots(dat.shape[0], dat.shape[1], figsize=(dat.shape[1]//2, dat.shape[0]//2))
#         plt.subplots_adjust(left=0.1, right=0.9, bottom=0.10, top=0.90, wspace=0, hspace=0.5)
#         for pat in range(dat.shape[0]):
#             for zone in range(dat.shape[1]):
#                 axs[pat,zone].imshow(dat[pat,zone], aspect="auto", cmap="binary", vmin=0, vmax=1)
#         for ax in axs.flatten():
#             ax.set_xticks([])
#             ax.set_yticks([])
#         axs[-1,0].set_xlabel("0")
#         for zone in range(dat.shape[1]):
#             t = (zone+1) * nstep_zone
#             if t%5==0:
#                 axs[-1,zone].set_xlabel(f"{t}")
#         fig.text(0.5, 0.02, 'Time since pattern onset (ms)', ha='center')
#         fig.text(0.02, 0.5, 'Pattern #', va='center', rotation='vertical')
#         if field=="zi.01":
#             plt.suptitle(r"$Pop_{INP}$")
#         elif field=="zj.12":
#             plt.suptitle(r"$Pop_{INPRC}$")
#         plt.savefig(f"exp4.{field}.png", dpi=300)
#         plt.savefig(f"exp4.{field}.svg", format="svg", dpi=300)
#         if (SHOW): plt.show(block=False)
#         else: plt.close()

# def parse_accuracy(filename, query):
#     with open(filename) as f:
#         for line in f.readlines():
#             if query in line:
#                 tr_acc = float(line.split()[6])
#                 te_acc = float(line.split()[11])
#                 return tr_acc, te_acc
#     return -1, -1

# def attractor_imagestrip(dataset="test"):
#     print ("attractor_imagestrip")
#     H, M = param["Hin"], param["Min"]
#     # Initialize variables
#     fieldnames = ["act_inp", "z_inp", "act_inprc", "z_inprc"]
#     fieldfilenames = [f"everystep.{dataset}.act.0.bin",
#                       f"everystep.{dataset}.zi.01.bin",
#                       f"everystep.{dataset}.act.2.bin",
#                       f"everystep.{dataset}.zj.12.bin"]
#     fieldprettynames = [r"${INP}$ spikes",
#                         r"${INP}$ $z$-traces",
#                         r"${INPRC}$ spikes",
#                         r"${INPRC}$ $z$-traces"]
#     # Params
#     nfield = len(fieldnames) # number of log variables to show
#     nrow = nfield
#     ncol = 15 # Time checkpoints to plot
#     vmin, vmax = 0, 1 # color min and max range
#     nshowstep = nstep_pat # gap period is boring to show
#     # Load all data file
#     dat = {} 
#     for fieldid in range(nfield):
#         fieldname = fieldnames[fieldid]
#         dat[fieldname] = loadbin(datadir=datadir, 
#                                  filename=fieldfilenames[fieldid], 
#                                  offset=(5000*nstep*784*2*np.dtype(np.float32).itemsize), 
#                                  count=(20*nstep*784*2), 
#                                  shape=(-1,nstep,784,2), 
#                                  verbose=1) 
#         dat[fieldname] = dat[fieldname]
#         print (dat[fieldname].shape)
#     # Do the plots for a few patterns
#     for patid in range(0, 20):
#         # Initialize plot
#         fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
#         plt.subplots_adjust(left=0.2, right=0.90, bottom=0.2, top=0.90, wspace=0.1, hspace=0.1)
#         for rowid in range(nrow):
#             fieldid = rowid
#             fieldname = fieldnames[fieldid]
#             for colid in range(ncol):
#                 # find time in simulation land to show
#                 t = nstep_gap + int(colid*nshowstep/ncol)
#                 imgdat = dat[fieldname][patid,t].reshape(28,28,M)[:,:,0]
#                 ax =  axs[rowid,colid]
#                 ax.imshow(imgdat, cmap="binary", vmin=vmin, vmax=vmax)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#         # Set x labels as times
#         for colid in range(ncol):
#             t = int(colid*nshowstep/ncol)
#             axs[-1,colid].set_xlabel(t) 
#         # Set x text as time
#         axs[-1,ncol//2].text(0, -1, 
#                       "Time since pattern onset (ms)", 
#                       transform=axs[-1,ncol//2].transAxes, 
#                       color="black", 
#                       horizontalalignment="center", 
#                       verticalalignment="center",
#                       fontsize=20
#                       )
#         # Set ylabel
#         for rowid in range(nrow):
#             ax = axs[rowid,0]
#             ax.set_ylabel(fieldprettynames[rowid], rotation="horizontal", va="center", ha="right", fontsize=20)
#         # Finalize plot
#         plt.savefig(f"exp4-attractor.{dataset}.pat{patid:02d}.png", dpi=400)
#         plt.close()

def attractor_video(npatshow=5):
    print ("attractor_video")
    import matplotlib.cm as cm
    import matplotlib.animation as animation
    import matplotlib.patches as mpatches
    from matplotlib import lines
    # Initialize variables
    H, M = param["Hin"], param["Min"]
    filename_all = [f"everystep.test.sup.2.bin",
                    f"everystep.test.act.2.bin",
                    f"everystep.test.zj.12.bin",
                    f"everystep.test.sup.1.bin",
                    f"everystep.test.act.1.bin",
                    f"everystep.test.zj.11.bin",
                    f"everystep.test.sup.0.bin",
                    f"everystep.test.act.0.bin",
                    f"everystep.test.zi.01.bin"]
    row_names = [r"${INPRC}$", r"${HID}$", r"${INP}$"]
    col_names = [r"$v_j$ (mem. volt.)", r"$s_j$ (spike)", r"$z_j$ ($Z$-traces)"]
    nfield = len(filename_all) # number of log variables to show
    # Load all data file
    dat = {} 
    for fieldid in range(nfield):
        if fieldid in [3, 4, 5]:
            dat[fieldid] = loadbin(datadir=datadir, 
                                   filename=filename_all[fieldid], 
                                   shape=(-1,100,100), 
                                   verbose=1)
        else:
            dat[fieldid] = loadbin(datadir=datadir, filename=filename_all[fieldid], shape=(-1,H,M), offset=0, verbose=1)
    # Function for updating all images
    def update_anim(frameid):
        t = frameid # ms
        if (t%100==0):
            print (f"Rendered t:{t:-5d}")
        imgs = []
        for axid, ax in enumerate(axs.flatten()):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            vmin, vmax = np.min(dat[axid]), np.max(dat[axid])
            if axid in [3, 4, 5]:
                img = ax.imshow(dat[axid][t].reshape(100,100), cmap="binary", vmin=vmin, vmax=vmax, animated=True)
            else:
                img = ax.imshow(dat[axid][t].reshape(28,28,M)[:,:,0], cmap="binary", vmin=vmin, vmax=vmax, animated=True)
            imgs.append(img)
        # Write on rows with population name
        for rowid in range(len(axs)):
            axs[rowid,0].set_ylabel(row_names[rowid])
        # Write on cols with field name
        for colid in range(len(axs[0])):
            axs[-1,colid].set_xlabel(col_names[colid]) 
        # Draw arrows
        axs[-1,1].annotate("",xy=(0.5, 1.3), xycoords='axes fraction',xytext=(0.5, 1), textcoords='axes fraction',arrowprops=dict(facecolor='gray', shrink=0.2),)        
        axs[-2,1].annotate("",xy=(0.5, 1.3), xycoords='axes fraction',xytext=(0.5, 1), textcoords='axes fraction',arrowprops=dict(facecolor='gray', shrink=0.2),)        
        time_text = axs[0,0].text(x=20, y=-5, s=f"Pattern: {t//nstep:-3d}      Time: {t%nstep-nstep_gap:-4d} ms")
        return [time_text, imgs]
    # Start figure
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    # Animate
    ani = animation.FuncAnimation(fig, update_anim, frames=nstep*npatshow, blit=False)
    GIFwriter = animation.PillowWriter(fps=10)
    ani.save('animation.gif',writer=GIFwriter)
    # FFwriter = animation.FFMpegWriter(fps=10)
    # ani.save('animation.mp4', writer = FFwriter)
    plt.savefig("animation.png")
    plt.show()

if __name__ == "__main__":

    SHOW = 1
    
    # matplotlib params
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size':15})

    datadir = "/cfs/klemming/scratch/n/nbrav/logs/test-mnist1k/"
    paramfilename = "apps/hidassospk/hidassospk.par" # f"{datadir}/net.par"
    param = parseparam(paramfilename)

    nstep_gap = param['nstep_gap']
    nstep_ffwd = param['nstep_ffwd']
    nstep_overlap = param['nstep_overlap']
    nstep_recr = param['nstep_recr']
    
    nstep_pat = nstep_ffwd + nstep_overlap + nstep_recr
    nstep = nstep_gap + nstep_pat
    
    ntrpat = param["ntrpat"]
    ntepat = param["ntepat"]

    # spkraster() # exp1a
    # simmat() # exp1b
    # rcpfld_formation() # exp2a
    # rewiring_scores() # exp2b
    # filters() # exp2c
    # parameter_explore() # exp3
    # attractor_video()
    # attractor_imagestrip("test") # exp1
    # attractor_imagestrip("complete") # exp4
    # attractor_imagestrip("distort") # exp4
    # attractor_imagestrip("rivalry") # exp4
    # prototype() # exp5
    # attractor_reconstruct() # exp?
    
    if (SHOW): plt.show()

    print ("Fin.")
