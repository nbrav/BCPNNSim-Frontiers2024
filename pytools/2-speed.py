import matplotlib.pyplot as plt
import numpy as np
import os, utils
from utils import parseparam
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size':15})

fig, ax = plt.subplots(1, 1, figsize=(5.5,4))
plt.subplots_adjust(left=0.17, right=0.95, bottom=0.15, top=0.9)

# BCPNN

resultsdir = "./results-mnist/"
df = pd.DataFrame()
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      paramfilename = [f"{datadir}/{filename}" for filename in os.listdir(datadir) if "par" in filename][0]
      param = utils.parseparam(paramfilename)
      summaryfile = f"{datadir}/summary.txt"
      with open(summaryfile) as f:
          for line in f:
              its, tracc, teacc = line.strip('\n').split(',')
              new_row = {'its': int(its), 'tracc': float(tracc), 'teacc': float(teacc), 'seed': param['seed']}
              new_df = pd.DataFrame(new_row, index=[1,2,3])
              df = pd.concat([df, new_df])
print (df)
output = df.groupby(['its'], as_index=False).agg({'tracc': ['mean','std'], 'teacc': ['mean','std']})
x = output['its']
y = output[('teacc','mean')]
yerr = output[('teacc', 'std')]
#ax.plot(x, y, 'o-', color="red", label="BCPNN 400x100")
#ax.fill_between(x=x, y1=y-yerr, y2=y+yerr, alpha=0.25, color="red")
ax.errorbar(x=x, y=y, yerr=yerr, alpha=1, linewidth=1.5, elinewidth=1.5, capsize=2, color="red", label="BCPNN 400x100")


resultsdir = "./results-mnist-old-old/"
df = pd.DataFrame()
for paramdir in os.listdir(resultsdir):
   for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
      datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
      paramfilename = [f"{datadir}/{filename}" for filename in os.listdir(datadir) if "par" in filename][0]
      param = utils.parseparam(paramfilename)
      summaryfile = f"{datadir}/summary.txt"
      with open(summaryfile) as f:
          for line in f:
              its, tracc, teacc = line.strip('\n').split(',')
              new_row = {'its': int(its), 'tracc': float(tracc), 'teacc': float(teacc), 'seed': param['seed']}
              new_df = pd.DataFrame(new_row, index=[1,2,3])
              df = pd.concat([df, new_df])
print (df)
output = df.groupby(['its'], as_index=False).agg({'tracc': ['mean','std'], 'teacc': ['mean','std']})
x = output['its']
y = output[('teacc','mean')]
yerr = output[('teacc', 'std')]
#ax.plot(x, y, 'o-', color="orange", label="BCPNN 30x100")
#ax.fill_between(x=x, y1=y-yerr, y2=y+yerr, alpha=0.25, color="orange")
ax.errorbar(x=x, y=y, yerr=yerr, alpha=1, linewidth=1.5, elinewidth=1.5, capsize=2, color="orange", label="BCPNN 30x100")

# RBM
resultsdir = "../../other-nets/boltz/"
df = pd.DataFrame()
for filename in os.listdir(resultsdir):
   if "summary" in filename:   
      summaryfile = f"{resultsdir}/{filename}"
      with open(summaryfile) as f:
         seed = filename.split('-')[0]
         for line in f:
            its, tracc, teacc = line.strip('\n').split(',')
            new_row = {'its': int(its), 'tracc': float(tracc)*100, 'teacc': float(teacc)*100, 'seed': seed}
            new_df = pd.DataFrame(new_row, index=[1,2,3])
            df = pd.concat([df, new_df])
print (df)
output = df.groupby(['its'], as_index=False).agg({'tracc': ['mean','std'], 'teacc': ['mean','std']})
x = output['its']
y = output[('teacc','mean')]
yerr = output[('teacc', 'std')]
#ax.plot(x, y, 'o-', color="green", label="RBM")
#ax.fill_between(x=x, y1=y-yerr, y2=y+yerr, alpha=0.25, color="green")
ax.errorbar(x=x, y=y, yerr=yerr, alpha=1, linewidth=1.5, elinewidth=1.5, capsize=2, color="green", label="RBM")

# AE
resultsdir = "../../other-nets/autoenc/"
df = pd.DataFrame()
for filename in os.listdir(resultsdir):
   if "summary" in filename:   
      summaryfile = f"{resultsdir}/{filename}"
      with open(summaryfile) as f:
         seed = filename.split('-')[0]
         for line in f:
            its, tracc, teacc = line.strip('\n').split(',')
            new_row = {'its': int(its), 'tracc': float(tracc)*100, 'teacc': float(teacc)*100, 'seed': seed}
            new_df = pd.DataFrame(new_row, index=[1,2,3])
            df = pd.concat([df, new_df])
print (df)
output = df.groupby(['its'], as_index=False).agg({'tracc': ['mean','std'], 'teacc': ['mean','std']})
x = output['its']
y = output[('teacc','mean')]
yerr = output[('teacc', 'std')]
#ax.plot(x, y, 'o-', color="purple", label="AE")
#ax.fill_between(x=x, y1=y-yerr, y2=y+yerr, alpha=0.25, color="purple")
ax.errorbar(x=x, y=y, yerr=yerr, alpha=1, linewidth=1.5, elinewidth=1.5, capsize=2, color="purple", label="AE")

# MLP
resultsdir = "../../other-nets/mlp/"
df = pd.DataFrame()
for filename in os.listdir(resultsdir):
   if "summary" in filename:   
      summaryfile = f"{resultsdir}/{filename}"
      with open(summaryfile) as f:
         seed = filename.split('-')[0]
         print (seed)
         for line in f:
            its, tracc, teacc = line.strip('\n').split(',')
            new_row = {'its': int(its), 'tracc': float(tracc)*100, 'teacc': float(teacc)*100, 'seed': seed}
            new_df = pd.DataFrame(new_row, index=[1,2,3])
            df = pd.concat([df, new_df])
print (df)
output = df.groupby(['its'], as_index=False).agg({'tracc': ['mean','std'], 'teacc': ['mean','std']})
x = output['its']
y = output[('teacc','mean')]
yerr = output[('teacc', 'std')]
#ax.plot(x, y, 'o-', color="blue", label="MLP")
#ax.fill_between(x=x, y1=y-yerr, y2=y+yerr, alpha=0.25, color="blue")
ax.errorbar(x=x, y=y, yerr=yerr, alpha=1, linewidth=1.5, elinewidth=1.5, capsize=2, color="blue", label="MLP")

ax.set_ylim(85, 100)
ax.set_yticks([85, 90, 95, 100])
ax.set_yticks(np.arange(85, 101), minor=True)

ax.set_xscale('log')
ax.set_xlim(5e0, 2e7)
ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
ticks=[small*10**big for big in range (1,7) for small in range(1,10)]
ax.set_xticks(ticks, minor=True)

ax.grid(which='both')
ax.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5, alpha=0.25)
ax.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.5, alpha=0.25)

plt.xlabel("# training iterations")
plt.ylabel("Accuracy [%]")
plt.legend(fontsize=10)
plt.savefig('2-speed.png', dpi=200)
plt.show()
