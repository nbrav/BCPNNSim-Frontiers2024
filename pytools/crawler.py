import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import pandas as pd

def parse_accuracy(filename, query):
    with open(filename) as f:
        for line in f.readlines():
            if query in line:
                tr_acc = float(line.split()[5])
                te_acc = float(line.split()[10])
                return tr_acc, te_acc
    return -1, -1
                
def crawl(resultsdir):
    "crawl through resultsdir/dir/runs/*, load files, classifiy and retrun dataframe of results"
    df = pd.DataFrame()    
    for paramdir in os.listdir(resultsdir):
        for timestampdir in os.listdir(f"{resultsdir}/{paramdir}"):
            datadir = f"{resultsdir}/{paramdir}/{timestampdir}"
            paramfilename = [f"{datadir}/{filename}" for filename in os.listdir(datadir) if "par" in filename][0]
            param = utils.parseparam(paramfilename)
            relevant_param = {'H2': param['H2'],
                              'M2': param['M2'],
                              'nconn1': param['nconn1'],
                              'nconn2': param['nconn2'],
                              'again': param['again'],
                              'datadir': param['datadir'],
            }

            new_row = relevant_param.copy()
            _, new_row['l1_acc'] = parse_accuracy(f"{datadir}/test.out", "Layer 1")
            _, new_row['l2_acc'] = parse_accuracy(f"{datadir}/test.out", "Layer 2")
            _, new_row['l3_acc'] = parse_accuracy(f"{datadir}/test.out", "Layer 3")
            if new_row['l1_acc']!=-1:
                df = df.append(new_row, ignore_index=True)

    print (df)####.sort_values(by=['again']))

    df_new = df.loc[(df['again']==1.0)].sort_values(by=['nconni', 'nconnh'])
    print (df_new)
    df_new = df.loc[(df['again']==5.0)].sort_values(by=['nconni', 'nconnh'])
    print (df_new)
    df_new = df.loc[(df['again']==10.0)].sort_values(by=['nconni', 'nconnh'])
    print (df_new)

    #print (df.loc[(df['again']==1.0) & (df['nconni']==25.0) & (df['nconnh']==16.0)])

    # plt.plot()
    
    return df

if __name__ == "__main__":

    datadir = "./"
    paramfilename = "apps/reprlearnspk/reprlearnspk.par"
    param = utils.parseparam(paramfilename)
    crawl("./results-mnist-deep/")
