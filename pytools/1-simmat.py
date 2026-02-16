import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size':15})

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.90, wspace=0.15, hspace=0.35)
      
from sklearn.metrics.pairwise import cosine_similarity      

def get_simmat(act, lbl):
   # sort by labels
   lbl = lbl.argmax(axis=1)
   sortid = np.argsort(lbl)
   lbl = lbl[sortid]
   act = act[sortid]
   # compute and plot
   simmat = cosine_similarity(act)      
   return simmat

def get_orthoscore(simmat, lbl):
   lbl = lbl.argmax(axis=1)
   sortid = np.argsort(lbl)
   lbl = lbl[sortid]
   same_class = np.zeros((len(lbl), len(lbl)))
   for idx1 in range(len(lbl)):
      for idx2 in range(len(lbl)):
         same_class[idx1,idx2] = lbl[idx1]==lbl[idx2]
   same_sim = np.multiply(same_class==1, simmat).sum() / (same_class==1).sum()
   diff_sim = np.multiply(same_class==0, simmat).sum() / (same_class==0).sum()
   simscore = same_sim / simmat.mean()
   #print (f"Simscore: {simscore}")
   return simscore
   
trnpat, tenpat = 1000, 1000

simscore = []
resultsdir = "./results-mnist-logged/"
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      param = utils.parseparam(f"{datadir}/net.par")
      nstep_per_pat = param['nstep_per_pat']
      telbl = utils.loadbin(param['datadir'], param['telblfile'], shape=(-1, param['Mo']))[:trnpat]
      # load input data
      # teact = utils.loadbin(datadir, f"predict.teact.l0.log", shape=(-1, param['Hi']*param['Mi']))[nstep_per_pat-1::nstep_per_pat]
      teact = utils.loadbin(param['datadir'], param['teimgfile'], shape=(-1, param['Hi']))
      simmat = get_simmat(teact, telbl)
      axs[0,0].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
      print (teact.shape, telbl.shape)
      simscore.append( get_orthoscore(simmat, telbl) )
      # load hidden data
      teact = utils.loadbin(datadir, f"predict.teact.l1.log", shape=(-1, param['Hh']*param['Mh']))[nstep_per_pat-1::nstep_per_pat]
      simmat = get_simmat(teact, telbl)
      axs[0,1].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
print (np.asarray(simscore).mean(), np.asarray(simscore).std())
      
resultsdir = "./results-fmnist-logged/"
simscore = []
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      param = utils.parseparam(f"{datadir}/net.par")
      nstep_per_pat = param['nstep_per_pat']
      telbl = utils.loadbin(param['datadir'], param['telblfile'], shape=(-1, param['Mo']))[:trnpat]
      # load input data
      #teact = utils.loadbin(datadir, f"predict.teact.l0.log", shape=(-1, param['Hi']*param['Mi']))[nstep_per_pat-1::nstep_per_pat]
      teact = utils.loadbin(param['datadir'], param['teimgfile'], shape=(-1, param['Hi']))
      simmat = get_simmat(teact, telbl)
      axs[1,0].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
      print (teact.shape, telbl.shape, teact.min(), teact.max(), telbl.min(), telbl.max(), nstep_per_pat)
      simscore.append( get_orthoscore(simmat, telbl) )
      # load hidden data
      teact = utils.loadbin(datadir, f"predict.teact.l1.log", shape=(-1, param['Hh']*param['Mh']))[nstep_per_pat-1::nstep_per_pat]
      simmat = get_simmat(teact, telbl)
      im = axs[1,1].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
print (np.asarray(simscore).mean(), np.asarray(simscore).std())

simscore = []
resultsdir = "./results-svhn-logged/"
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      param = utils.parseparam(f"{datadir}/net.par")      
      telbl = utils.loadbin(param['datadir'], param['telblfile'], shape=(-1, param['Mo']))[:trnpat]
      # load input data
      # teact = utils.loadbin(datadir, f"predict.teact.l0.log", shape=(-1, param['Hi']*param['Mi']))
      teact = utils.loadbin(param['datadir'], param['teimgfile'], shape=(-1, param['Hi']))
      simmat = get_simmat(teact, telbl)
      axs[0,0].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
      simscore.append( get_orthoscore(simmat, telbl) )
      # load hidden data
      teact = utils.loadbin(datadir, f"predict.teact.l1.log", shape=(-1, param['Hh']*param['Mh']))
      simmat = get_simmat(teact, telbl)
      im = axs[0,1].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
print (np.asarray(simscore).mean(), np.asarray(simscore).std())

simscore = []
resultsdir = "./results-cifar10-logged/"
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      param = utils.parseparam(f"{datadir}/net.par")      
      telbl = utils.loadbin(param['datadir'], param['telblfile'], shape=(-1, param['Mo']))[:trnpat]
      # load input data
      # teact = utils.loadbin(datadir, f"predict.teact.l0.log", shape=(-1, param['Hi']*param['Mi']))
      teact = utils.loadbin(param['datadir'], param['teimgfile'], shape=(-1, param['Hi']))
      simmat = get_simmat(teact, telbl)
      axs[1,0].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
      simscore.append( get_orthoscore(simmat, telbl) )
      # load hidden data
      teact = utils.loadbin(datadir, f"predict.teact.l1.log", shape=(-1, param['Hh']*param['Mh']))
      simmat = get_simmat(teact, telbl)
      im = axs[1,1].imshow(simmat, cmap="bwr", vmin=0, vmax=1)
print (np.asarray(simscore).mean(), np.asarray(simscore).std())

axs[0,0].set_title("input layer\nMNIST")
axs[0,1].set_title("hidden layer\nMNIST")
axs[1,0].set_title("input layer\nFashion-MNIST")
axs[1,1].set_title("hidden layer\nFashion-MNIST")

for ax in axs.flatten(): ax.set_xticks([])

axs[0,0].set_yticks(np.arange(10)*100+50)
axs[0,0].set_yticklabels(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
axs[0,1].set_yticks(np.arange(10)*100+50)
axs[0,1].set_yticklabels(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
axs[1,0].set_yticks(np.arange(10)*100+50)
axs[1,0].set_yticklabels(['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot'])
axs[1,1].set_yticks(np.arange(10)*100+50)
axs[1,1].set_yticklabels(['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot'])

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.50])
fig.colorbar(im, cax=cbar_ax, ticks=[0,1], label="Similarity")

#plt.colorbar()
plt.savefig(f"1-simmat.png", dpi=200)
plt.savefig(f"1-simmat.svg", format='svg', dpi=200)
#plt.show()
plt.close()           
      
