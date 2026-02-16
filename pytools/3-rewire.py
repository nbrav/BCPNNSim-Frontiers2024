import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import utils
import pandas as pd
from scipy import stats

datadir = "./"
param = utils.parseparam(f"{datadir}/apps/reprlearn/reprlearn.par")
cij = utils.loadbin(datadir, "learn.cij.l1.bin", shape=(-1, param['H1'], param['H0']), dtype=np.int32)
mi = utils.loadbin(datadir, "learn.mi.l1.bin", shape=(-1, param['H1'], param['H0']))
nmi = utils.loadbin(datadir, "learn.nmi.l1.bin", shape=(-1, param['H1'], param['H0']))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size':18})

fig, axs = plt.subplots(2, 1, figsize=(7,10))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.9, wspace=0.4, hspace=0.5)

# plot conn diff
cij_diff = np.abs(cij[1:] - cij[:-1])/2.
cij_diff = cij_diff.sum(axis=2)[1:]
y = cij_diff.mean(axis=1)
yerr = cij_diff.std(axis=1)
axs[0].plot(range(1,len(cij)-1), y, '-', color="black")
axs[0].fill_between(x=range(1,len(cij)-1), y1=y-yerr, y2=y+yerr, alpha=0.25, color="black")
axs[0].set_xlabel("N:o training steps")
axs[0].set_xscale('log')
#axs[0].set_xlim(9e-1, len(cij))
#ticks = np.asarray([1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3])
#axs[0].set_xticks(ticks=[1e0, 1e1, 1e2, 1e3])
#axs[0].set_xticklabels(["5x10$^2$", "5x10$^3$", "5x10$^4$", "5x10$^5$"])
#axs[0].set_ylabel("N:o swaps per\n(hidden) hypercolumn")
axs[0].grid(which="both")
axs[0].grid(visible=True, which='major', color='k', linestyle='-', alpha=0.25)
axs[0].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.25)

# plot usage score
score = np.multiply(mi, cij)
score = score.reshape(-1, param['H1'], param['H0']).sum(axis=2)
y = score.mean(axis=1)[1:]
yerr = score.std(axis=1)[1:] # stats.sem(nmi, axis=1) 
print (len(score), y.shape, yerr.shape)
axs[1].plot(range(1,len(score)), y, '-', color="black")
axs[1].fill_between(x=range(1,len(score)), y1=y-yerr, y2=y+yerr, alpha=0.25, color="black")
axs[1].set_xlabel("N:o training steps")
axs[1].set_xscale('log')
#axs[1].set_xlim(9e-1, len(cij))
#ticks = np.asarray([1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3])
#axs[1].set_xticks(ticks=[1e0, 1e1, 1e2, 1e3])
#axs[1].set_xticklabels(["5x10$^2$", "5x10$^3$", "5x10$^4$", "5x10$^5$"])
axs[1].set_ylabel("Usage score")
axs[1].grid(which="both")
axs[1].grid(visible=True, which='major', color='k', linestyle='-', alpha=0.25)
axs[1].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.25)

plt.text(-0.2, 1.2, 'B.', ha='left', va='top', transform=axs[0].transAxes, fontsize=20)
plt.text(-0.2, 1.2, 'C.', ha='left', va='top', transform=axs[1].transAxes, fontsize=20)
plt.savefig("3-swaps.png", dpi=200)
plt.savefig("3-swaps.svg", format='svg', dpi=200)
plt.show()
plt.close()

iters = [0, 1, 2, 5, 10, 20, 50, 100, 200]
true_iters = ["$0$", "5x10$^2$", "$1x10^3$", "$2x10^3$", "5x10$^3$", "$1x10^4$", "$2x10^4$", "5x10$^4$", "$1x10^5$", "$2x10^5$", "5x10$^5$", "1x10$^6$", "2x10$^6$", "5x10$^6$", "1x10$^7$"]
Nx, Ny = 8, 9 #len(true_iters)
fig, axs = plt.subplots(Nx, Ny, figsize=(10, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.92, wspace=0.05, hspace=0.05)
for x in range(Nx):
    for y in range(Ny):
        hc = x
        its = iters[y] 
        axs[x,y].imshow(cij[its,hc].reshape(28, 28), cmap="binary", interpolation="None")
        axs[x,y].set_xticklabels([])
        axs[x,y].set_yticklabels([])
        axs[x,y].tick_params(left=False, right=False, top=False, bottom=False)  # remove the ticks
plt.suptitle(f"Formation of receptive fields", fontsize=20)
#for y in range(Ny):
#    axs[-1,y].set_xlabel(f"{true_iters[y]}", fontsize=15)
plt.text(-0.5, 1.5, 'A.', ha='left', va='top', transform=axs[0,0].transAxes, fontsize=20)
plt.savefig("3-rcpfld.png", dpi=200)
plt.savefig("3-rcpfld.svg", format='svg', dpi=200)
plt.show()
plt.close()
      
