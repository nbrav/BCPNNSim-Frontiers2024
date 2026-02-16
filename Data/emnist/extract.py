import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip

def download():
    '''
    Download files from url
    '''
    os.makedirs(f"{DownloadDir}", exist_ok=True)
    os.system(f"wget -nc -P {DownloadDir} {url}")
    os.system(f"unzip {DownloadDir}/gzip.zip")
    os.system(f"mv gzip/* {DownloadDir}")
    os.system(f"rm -r gzip")
    return

def plotgrid(img, lbl, label_names, savedir):
    img = (img-img.min())/(img.max()-img.min())
    for label in range(nclass):
        labelid = np.argwhere(label==lbl)
        fig, axs = plt.subplots(10, 10, figsize=(6,6))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        for x in range(10):
            for y in range(10):
                idx = labelid[x*10+y]
                axs[x,y].imshow(img[idx].reshape(28,28))
                axs[x,y].axis('off') 
        #plt.show()
        plt.savefig(f"{savedir}/{label_names[label]}.png", dpi=300)
        plt.close()
    return

def get_label_names():
    label_names = {}
    for k in range(0,nclass): label_names[k] = f'{k}'
    # TODO: use ord, chr function for conversion from ASCII decimal to symbol
    return label_names

def make_raw():

    f = gzip.open(f'{DownloadDir}/{filenames["trimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(ntrain*28*28)
    trimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(ntrain, 28, 28)/255.   
    trimg = trimg.transpose(0,2,1) 
    print (trimg.shape, trimg.min(), trimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["teimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(ntest*28*28)
    teimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(ntest, 28, 28)/255.   
    teimg = teimg.transpose(0,2,1)  
    print (teimg.shape, teimg.min(), teimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["trlbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(ntrain*1)
    trlbl = np.frombuffer(buf, dtype=np.uint8) 
    trlbl_onehot = np.eye(nclass)[trlbl]
    print (trlbl.shape, trlbl.min(), trlbl.max())

    f = gzip.open(f'{DownloadDir}/{filenames["telbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(ntest*1)
    telbl = np.frombuffer(buf, dtype=np.uint8)
    telbl_onehot = np.eye(nclass)[telbl]
    print (telbl.shape, telbl.min(), telbl.max())

    os.makedirs(f"{RawDir}", exist_ok=True)
    
    trimg.astype('float32').tofile(f"{RawDir}/emnist_trimg.bin")
    trlbl_onehot.astype('float32').tofile(f"{RawDir}/emnist_trlbl.bin")
    teimg.astype('float32').tofile(f"{RawDir}/emnist_teimg.bin")
    telbl_onehot.astype('float32').tofile(f"{RawDir}/emnist_telbl.bin")

    plotgrid(teimg, telbl, get_label_names(), savedir=RawDir)
        
    return

if __name__ == "__main__":

    DownloadDir = "Download"
    RawDir = "Raw"
    
    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

    filenames = {"trimg": "emnist-balanced-train-images-idx3-ubyte.gz",
                 "trlbl": "emnist-balanced-train-labels-idx1-ubyte.gz",
                 "teimg": "emnist-balanced-test-images-idx3-ubyte.gz",
                 "telbl": "emnist-balanced-test-labels-idx1-ubyte.gz"}

    ntrain = 112800
    ntest = 18800
    nclass = 47
    
    download()
    
    make_raw()
