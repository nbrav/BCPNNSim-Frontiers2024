#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import parseparam, loadbin
import os
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

if __name__ == "__main__":

	datadir = "Data/mnist/Raw/"
	
	trimg = loadbin(datadir, "mnist_trimg.bin", dtype=np.float32, shape=(-1, 28, 28))
	teimg = loadbin(datadir, "mnist_teimg.bin", dtype=np.float32, shape=(-1, 28, 28))
	
	fig, axs = plt.subplots(7, 7, figsize=(7, 7))
	axid = 0
	for patid in range(49):
		ax = axs.flatten()[patid]
		ax.imshow(trimg[patid], cmap="binary")
		axid += 1
	plt.savefig("trimg.pats.svg", format='svg', dpi=300)
	plt.savefig("trimg.pats.png", dpi=300)
	plt.show()

	print (trimg.shape, teimg.shape)
