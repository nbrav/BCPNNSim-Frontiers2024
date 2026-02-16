import matplotlib.pyplot as plt
import numpy as np
import os
from utils import parseparam
import pandas as pd

def parse_acc(df):
    logfilename = f"{datadir}/test.out"
    f = open(logfilename, "r")
    traccs, teaccs = {}, {}
    for line in f:
        words = line.split(' ')
        for wordidx, word in enumerate(words):
            if (word=="Layer"):
                layer = int(words[wordidx+1].replace(",", ""))
                tracc = float(words[wordidx+2].replace(",", ""))
                teacc = float(words[wordidx+3].replace(",", ""))
                traccs[f'l{layer}-tracc'] = tracc
                teaccs[f'l{layer}-teacc'] = teacc
    return traccs, teaccs

if __name__ == "__main__":

    # matplotlib config
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size':15})

    # pandas config
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame()
    
    resultsdir = "results-mnist-deep"
    # paramdir = "wta-again5"
    # timestampdir = "." # "2023-11-17_11:20:31:074028"
    # datadir = "." # f"{resultsdir}/{paramdir}/{timestampdir}"
    # paramfilename = "apps/deepassonet/deepassonet.par" # f"{datadir}/net.par"
    for paramdir in os.listdir(resultsdir):
        for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
            datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
            paramfilename = f"{datadir}/net.par"
            param = parseparam(paramfilename)
            traccs, teaccs = parse_acc(df)
            for layer in range(param['nlayer']):
                new_row = {'Hhid': param['Hhid'],
                           'Mhid': param['Mhid'],
                           'nconnih': param['nconnih'],
                           'nconnhh': param['nconnhh'],
                           'actfn': param['actfn'],
                           'layer': layer,
                           'tracc': traccs[f'l{layer}-tracc'],
                           'teacc': teaccs[f'l{layer}-teacc']
                }
                new_row_df = pd.DataFrame(new_row, index=[0]) # create a dataframe for new entry 
                df = pd.concat([df, new_row_df]) # concatenate with full dataframe
    df = df.sort_values(["actfn", "Hhid", "Mhid", "nconnih", "nconnhh"]) 

    nx = len(df["nconnih"].unique())
    ny = len(df["nconnhh"].unique())
    fig, axs = plt.subplots(nx, ny, figsize=(11, 8), sharex='all', sharey='all')
    plt.subplots_adjust(left=0.12, right=0.8, bottom=0.12, top=0.9, wspace=0.05, hspace=0.05)
    for x, nconnih in enumerate(np.sort(df["nconnih"].unique())):
        for y, nconnhh in enumerate(np.sort(df["nconnhh"].unique())):
            colors = ["green", "red"]
            for actfn_idx, actfn in enumerate(np.sort(df["actfn"].unique())):
                df_filt = df[(df["actfn"]==actfn) & (df["nconnih"]==nconnih) & (df["nconnhh"]==nconnhh)]
                axs[x][y].plot(df_filt["layer"].to_numpy(),
                               df_filt["tracc"].to_numpy(),
                               linestyle='-',
                               marker='o',
                               color=colors[actfn_idx],
                               linewidth=2,
                               alpha=0.7,
                               label=f"{actfn}-train")
                axs[x][y].plot(df_filt["layer"].to_numpy(),
                               df_filt["teacc"].to_numpy(),
                               linestyle='--',
                               marker='o',
                               color=colors[actfn_idx],
                               linewidth=2,
                               alpha=0.7,
                               label=f"{actfn}-test")
                axs[x][y].grid(True)
    # print xlabels as nconnih
    for x, nconnih in enumerate(np.sort(df["nconnih"].unique())):
        axs[x][-1].set_ylabel(r"$n_{ih}$="+f"{nconnih}", fontsize=15)
        axs[x][-1].yaxis.set_label_position("right") 
    # print ylabels as nconnhh
    for y, nconnhh in enumerate(np.sort(df["nconnhh"].unique())):
        axs[0][y].set_xlabel(r"$n_{hh}$="+f"{nconnhh}", fontsize=15)
        axs[0][y].xaxis.set_label_position("top") 
    axs[0][0].set_xticks(np.arange(0,4))
    plt.legend(bbox_to_anchor=(1.9, 2.5), ncol=1, fontsize=12)
    axs[-1][0].set_xlabel("Layer")
    axs[-1][0].set_ylabel("Accuracy")
    plt.suptitle("mnist")
    plt.show()
    #plt.savefig("acc.png", dpi=200) # show()
