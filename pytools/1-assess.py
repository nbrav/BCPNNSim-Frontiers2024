import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import pandas as pd

layerid = 1

resultsdir = "./results-mnist-big/"

ent_act = []
ent_pj = []

for paramdir in os.listdir(resultsdir):
   
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      paramfilename = [f"{datadir}/{filename}" for filename in os.listdir(datadir) if "par" in filename][0]
      
      param = utils.parseparam(paramfilename)
      
      act = utils.loadbin(datadir, f"predict.teact.l{layerid}.log", shape=(-1, param['H1']*param['M1']))
      pj = utils.loadbin(datadir, f"learn.pj.l{layerid}.bin", shape=(param['H1']*param['M1']))
      
      ent = - np.sum(np.multiply(act, np.log(act + 1e-9))) / len(act) / param['H1']
      ent_act.append( ent )
      
      print (f"Ent_act = {ent}")
      
      ent = - np.sum(np.multiply(pj, np.log(pj + 1e-9))) / param['H1']
      ent_pj.append( ent )
      
      print (f"Ent_usage = {ent}")
      
      plt.rcParams['font.family'] = 'serif'
      plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
      plt.rcParams.update({'font.size':15})
      
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
      plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.90, wspace=0.25, hspace=0.25)
      
      # activtivy histogram
      axs[0].hist(act[:1000].flatten(), bins=np.linspace(0, 1, 1000), facecolor="white", edgecolor="black", linewidth=2, histtype='stepfilled')
      axs[0].set_yscale("log")
      axs[0].set_ylim(1, len(act[:1000].flatten())*1) #)1e1, 1e7)
      axs[0].set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
      axs[0].set_yticklabels(["1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
      axs[0].set_ylabel("Density", fontsize=20)
      axs[0].set_xlabel(r"minicolumn activity", fontsize=20)
      axs[0].set_xlim(-0.05, 1.05)
      axs[0].grid(which="both")
      axs[0].grid(b=True, which='major', color='k', linestyle='-', alpha=0.25)
      axs[0].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
      
      # usage histogram
      axs[1].hist(pj.flatten(), bins=np.linspace(0, 1, 1000), facecolor="white", edgecolor="black", linewidth=2, histtype='stepfilled')
      axs[1].set_yscale("log")
      axs[1].set_ylim(1e0, len(pj.flatten()))
      axs[1].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4])
      axs[1].set_yticklabels(["1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
      axs[1].set_xlim(-0.05, 1.05)
      axs[1].set_ylabel("Density", fontsize=20)
      axs[1].set_xlabel(r"minicolumn usage", fontsize=20)
      axs[1].grid(which="both")
      axs[1].grid(b=True, which='major', color='k', linestyle='-', alpha=0.25)
      axs[1].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
      
      # inset histogram
      left, bottom, width, height = [0.7, 0.6, 0.2, 0.2]
      ax_inset = fig.add_axes([left, bottom, width, height])            
      ax_inset.hist(pj.flatten(), bins=np.linspace(0, 1, 1000), facecolor="white", edgecolor="black", linewidth=2, histtype='stepfilled')
      ax_inset.set_yscale("log")
      ax_inset.set_ylim(1e0, len(pj.flatten()))
      ax_inset.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4])
      ax_inset.set_yticklabels(["1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
      ax_inset.set_xlim(1./param['M1']-0.015, 1./param['M1']+0.01)
      axs[1].indicate_inset_zoom(ax_inset, edgecolor="black")
      
      print (datadir)
      plt.savefig("1-entropy.png", dpi=200)
      plt.savefig("1-entropy.svg", format='svg', dpi=200)
      #plt.show()
      
print (np.asarray(ent_act).mean(), np.asarray(ent_act).std())
print (np.asarray(ent_pj).mean(), np.asarray(ent_pj).std())

