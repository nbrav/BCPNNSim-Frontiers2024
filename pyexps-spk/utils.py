import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size':15})

import numpy as np
import pandas as pd

import os
import math
import sys
import socket
import random
import string
import copy
import time
from datetime import datetime
import argparse
from itertools import product

def getpopinfo(popid, param):
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

def getprjinfo(prjid, param):
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

def parseparam(paramfilename) :
    # Parse parameter file and return dict of parameter name and value #
    pardtype = {"verbosity" : int,
                "seed" : int,
                "Hin": int,
                "Min": int,
                "Hhid": int,
                "Mhid": int,
                "Hout": int,
                "Mout": int,
                "binarize_in": int,
                "Hi" : int,
                "Mi" : int,
                "Hh" : int,
                "Mh" : int,
                "Ho" : int,
                "Mo" : int,
                "H0" : int,
                "M0" : int,
                "H1" : int,
                "M1" : int,
                "H2" : int,
                "M2" : int,
                "H3" : int,
                "M3" : int,
                "H4" : int,
                "M4" : int,
                "nconni" : int,
                "nconnh" : int,                
                "nconnih" : int,
                "nconnhh" : int,
                "nconnhi" : int,
                "nconn1" : int,
                "nconn2" : int,
                "nconn3" : int,
                "nconn4" : int,
                "nlayer": int,
                "updconn_interval" : int,
                "updconn_nswapmax" : int,
                "updconn_threshold" : float,        
                "biasreg": int,
                "tau_kb": float,
                "kb_half": float,
                "dt" : float,
                "taum" : float,
                "tauz" : float,
                "taup" : float,
                "eps": float,
                "maxfq" : int,
                "nampl" : float, 
                "actfn" : str,             
                "again" : int,             
                "nstep_gap" : int,
                "nstep_ffwd" : int,
                "nstep_overlap" : int,
                "nstep_recr" : int,
                "bgain" : float,
                "wgain" : float,
                "noise_amp" : float,
                "nusupepoch" : int,
                "nsupepoch" : int,
                "ntrpat" : int,
                "ntepat" : int,
                "nvalpat" : int,
                "datadir" : str,
                "trimgfile" : str,
                "teimgfile" : str,
                "trlblfile" : str,
                "telblfile" : str,
                "complete_teimgfile": str,
                "complete_telblfile": str,
                "rivalry_teimgfile": str,
                "rivalry_telblfile": str,
                "distort_teimgfile": str,
                "distort_telblfile": str,
                "logdir": str,
                "logmodelname": str
                }
    param = {}
    f = open(paramfilename, "r")
    for line in f:
        words = line.split()
        if (len(words)>=2):
            key, value = words[0], words[1]
            if (key not in pardtype.keys()):
                print(f"Warning! Parsing param {key} of undefined dtype. Assigning as string..")
                param[key] = value
            else:
                param[key] = pardtype[key](value)                
    return param

def stats(arr):
    return f"\tsize:{arr.shape} \tmin:{arr.min():.3f} \tmax:{arr.max():.3f} \tmean:{arr.mean():.3f} \tsum:{arr.sum():.3f} "

def loadbin(datadir, filename, shape, dtype=np.float32, offset=0, count=-1, verbose=False):
    dat = np.fromfile(datadir+"/"+filename, offset=offset, count=count, dtype=dtype)
    dat = dat.reshape(shape)
    if verbose:
        print (f"Loaded \t{filename} {stats(dat)}")
    return dat