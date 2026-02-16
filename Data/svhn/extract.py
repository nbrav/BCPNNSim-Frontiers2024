import numpy as np
#import matplotlib.pyplot as plt
import os
import struct
import gzip
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.io

def format_float(num):
    return np.format_float_positional(num, trim='-')

def download():
    '''
    From http://ufldl.stanford.edu/housenumbers/
    '''
    os.makedirs(f"{DownloadDir}", exist_ok=True) # Create directory
    url_tr = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    url_te = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    os.system(f"wget -nc -P {DownloadDir} {url_tr}") # Download from url
    os.system(f"wget -nc -P {DownloadDir} {url_te}") # Download from url

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_label_names():
    label_names = {}
    for k in range(0,10): label_names[k] = f'{(k+1)%10}'
    return label_names

def make_raw():

    trmat = scipy.io.loadmat(f"{DownloadDir}/train_32x32.mat")
    temat = scipy.io.loadmat(f"{DownloadDir}/test_32x32.mat")
    
    trimg = trmat['X']/255.
    trlbl = trmat['y'].flatten() - 1
    teimg = temat['X']/255.
    telbl = temat['y'].flatten() - 1

    trimg = trimg.transpose(3,0,1,2)
    teimg = teimg.transpose(3,0,1,2)
    
    print (trimg.shape, trimg.min(), trimg.max())
    print (trlbl.shape, trlbl.min(), trlbl.max())
    print (teimg.shape, teimg.min(), teimg.max())
    print (telbl.shape, telbl.min(), telbl.max())

    trlbl_onehot = np.eye(10)[trlbl]
    telbl_onehot = np.eye(10)[telbl]

    os.makedirs(f"{RawDir}", exist_ok=True)
    
    trimg.astype('float32').tofile(f"{RawDir}/svhn_trimg.bin")
    trlbl_onehot.astype('float32').tofile(f"{RawDir}/svhn_trlbl.bin")
    teimg.astype('float32').tofile(f"{RawDir}/svhn_teimg.bin")
    telbl_onehot.astype('float32').tofile(f"{RawDir}/svhn_telbl.bin")

    plotgrid(teimg, telbl, label_names, savedir=RawDir)
    
def make_kmeans(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    trimg = np.fromfile(f"{RawDir}/svhn_trimg.bin", dtype=np.float32).reshape(73257, 32*32, 3)
    teimg = np.fromfile(f"{RawDir}/svhn_teimg.bin", dtype=np.float32).reshape(26032, 32*32, 3)
    print (trimg.shape, trimg.min(), trimg.max())

    ntrain = len(trimg)
    ntest = len(teimg)
    npixel = 32*32
    
    # tract = np.zeros((ntrain, npixel, k))
    # teact = np.zeros((ntest, npixel, k))    
    # for pixelid in range(npixel):
    #     print (f"Pixel: {pixelid}")
    #     model = KMeans(n_clusters=k, init='random', n_init=3, verbose=0)  
    #     model.fit(trimg[:, pixelid])
    #     trpred = model.predict(trimg[:, pixelid])        
    #     tepred = model.predict(teimg[:, pixelid])        
    #     tract[:, pixelid, :] = np.eye(k)[trpred]
    #     teact[:, pixelid, :] = np.eye(k)[tepred]
    #     if pixelid%100==0: print (np.round(model.cluster_centers_, 3))

    trimg = trimg.reshape(-1,3)
    teimg = teimg.reshape(-1,3)

    model = KMeans(n_clusters=k, init='random', n_init=3, verbose=True)

    ntrpat = 1000000
    randidx = np.arange(len(trimg))
    np.random.shuffle(randidx)
    
    model.fit(trimg[randidx][:ntrpat])

    print ("\nKMeans cluster centers\n", np.round(model.cluster_centers_, 3))
    
    tract = model.predict(trimg)    
    tract = np.eye(k)[tract]
    tract = tract.reshape(-1,32*32,k)

    teact = model.predict(teimg)
    teact = np.eye(k)[teact]
    teact = teact.reshape(-1,32*32,k)

    os.makedirs(f"{KmeansDir}", exist_ok=True)
    os.makedirs(f"{KmeansDir}/k{k}", exist_ok=True)

    tract.astype('float32').tofile(f"{KmeansDir}/k{k}/svhn_trimg.bin")
    teact.astype('float32').tofile(f"{KmeansDir}/k{k}/svhn_teimg.bin")

def make_gmm(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    trimg = np.fromfile(f"{RawDir}/svhn_trimg.bin", dtype=np.float32).reshape(73257, 32*32, 3)
    teimg = np.fromfile(f"{RawDir}/svhn_teimg.bin", dtype=np.float32).reshape(26032, 32*32, 3)
    print (trimg.shape, trimg.min(), trimg.max())
    
    ntrain = len(trimg)
    ntest = len(teimg)
    npixel = trimg.shape[1]
        
    # tract = np.zeros((ntrain, npixel, k))
    # teact = np.zeros((ntest, npixel, k))
    # For pixelid in range(npixel):        
    #     print (f"Pixel: {pixelid}")
    #     model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=3, verbose=0)
    #     model.fit(trimg[:, pixelid])                
    #     trpred = model.predict_proba(trimg[:, pixelid])        
    #     tepred = model.predict_proba(teimg[:, pixelid])
    #     tract[:, pixelid, :] = trpred
    #     teact[:, pixelid, :] = tepred
    #     if pixelid%100==0: print (np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
            
    trimg = trimg.reshape(-1,3)
    teimg = teimg.reshape(-1,3)

    model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=10, verbose=True)

    ntrpat = 1000000
    randidx = np.arange(len(trimg))
    np.random.shuffle(randidx)
    
    model.fit(trimg[randidx][:ntrpat])

    print ("\nGMM cluster centers\n", np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
    
    tract = model.predict_proba(trimg)    
    tract= tract.reshape(-1,32*32,k)

    teact = model.predict_proba(teimg)
    teact = teact.reshape(-1,32*32,k)

    os.makedirs(f"{GmmDir}", exist_ok=True)
    os.makedirs(f"{GmmDir}/k{k}", exist_ok=True)

    tract.astype('float32').tofile(f"{GmmDir}/k{k}/svhn_trimg.bin")
    teact.astype('float32').tofile(f"{GmmDir}/k{k}/svhn_teimg.bin")

def make_whiten(eps=1e-2):
    '''
    Sources
    1. https://cs231n.github.io/neural-networks-2/
    2. https://www.kdnuggets.com/2018/10/preprocessing-deep-learning-covariance-matrix-image-whitening.html/3
    3. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''
    trimg = np.fromfile(f"{RawDir}/svhn_trimg.bin", dtype=np.float32).reshape(73257, 32*32*3)
    teimg = np.fromfile(f"{RawDir}/svhn_teimg.bin", dtype=np.float32).reshape(26032, 32*32*3)
    trlbl = np.fromfile(f"{RawDir}/svhn_trlbl.bin", dtype=np.float32).reshape(73257, 10)
    telbl = np.fromfile(f"{RawDir}/svhn_telbl.bin", dtype=np.float32).reshape(26032, 10)

    # normalize data
    trimg_mean = np.mean(trimg, axis=0)
    trimg_norm = trimg - trimg_mean
    teimg_norm = teimg - trimg_mean

    # compute covariance across data features
    #cov = np.dot(trimg_norm.T, trimg_norm) / len(trimg)
    cov = np.cov(trimg_norm, rowvar=False)
    
    # eigen decompose
    U, S, V = np.linalg.svd(cov)

    # find whitening matrix
    W = U.dot(np.diag(1.0/np.sqrt(S + eps))).dot(U.T)

    # whiten the data
    trimg_whiten = np.dot(W, trimg_norm.T).T
    teimg_whiten = np.dot(W, teimg_norm.T).T

    # normalize whitened data
    #trimg_whiten = (trimg_whiten-trimg_whiten.min())/(trimg_whiten.max()-trimg_whiten.min())
    #teimg_whiten = (teimg_whiten-teimg_whiten.min())/(teimg_whiten.max()-teimg_whiten.min())

    os.makedirs(f"{WhitenDir}", exist_ok=True)
    os.makedirs(f"{WhitenDir}/eps{format_float(eps)}", exist_ok=True)
    
    trimg_whiten.astype('float32').tofile(f"{WhitenDir}/eps{format_float(eps)}/svhn_trimg.bin")
    teimg_whiten.astype('float32').tofile(f"{WhitenDir}/eps{format_float(eps)}/svhn_teimg.bin")

    trlbl = np.argmax(trlbl, axis=1)
    telbl = np.argmax(telbl, axis=1)
    
    plotgrid(teimg_whiten.reshape(-1,32,32,3), telbl, label_names, savedir=f"{WhitenDir}/eps{format_float(eps)}/")

    print (f"Whiten eps={eps} done.")
    
    return

def make_whitenkmeans(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    for eps in all_eps:
        
        trimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/svhn_trimg.bin", dtype=np.float32).reshape(73257, 32*32, 3)
        teimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/svhn_teimg.bin", dtype=np.float32).reshape(26032, 32*32, 3)
       
        ntrain = len(trimg)
        ntest = len(teimg)
        npixel = trimg.shape[1]
            
        # tract = np.zeros((ntrain, npixel, k))
        # teact = np.zeros((ntest, npixel, k))        
        # for pixelid in range(npixel):            
        #     print (f"Pixel: {pixelid}")
        #     model = KMeans(n_clusters=k, init='random', n_init=3, verbose=0)  
        #     model.fit(trimg[:, pixelid])
        #     trpred = model.predict(trimg[:, pixelid])        
        #     tepred = model.predict(teimg[:, pixelid])        
        #     tract[:, pixelid, :] = np.eye(k)[trpred]
        #     teact[:, pixelid, :] = np.eye(k)[tepred]
        #     if pixelid%100==0: print (np.round(model.cluster_centers_, 3))
                
        trimg = trimg.reshape(-1,3)
        teimg = teimg.reshape(-1,3)
        
        model = KMeans(n_clusters=k, init='random', n_init=10, verbose=True)
        
        ntrpat = 1000000
        randidx = np.arange(len(trimg))
        np.random.shuffle(randidx)
        
        model.fit(trimg[randidx][:ntrpat])
        
        print ("\nKMeans cluster centers\n", np.round(model.cluster_centers_, 3))
        
        tract = model.predict(trimg)    
        tract = np.eye(k)[tract]
        tract = tract.reshape(-1,32*32,k)
        
        teact = model.predict(teimg)
        teact = np.eye(k)[teact]
        teact = teact.reshape(-1,32*32,k)

        os.makedirs(f"{WhitenKmeansDir}", exist_ok=True)
        os.makedirs(f"{WhitenKmeansDir}/eps{format_float(eps)}", exist_ok=True)
        os.makedirs(f"{WhitenKmeansDir}/eps{format_float(eps)}/k{k}", exist_ok=True)
        
        tract.astype('float32').tofile(f"{WhitenKmeansDir}/eps{format_float(eps)}/k{k}/svhn_trimg.bin")
        teact.astype('float32').tofile(f"{WhitenKmeansDir}/eps{format_float(eps)}/k{k}/svhn_teimg.bin")

        print (f"Whiten Kmeans eps={eps} k={k} done.")
    
def make_whitengmm(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    for eps in all_eps:
        
        trimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/svhn_trimg.bin", dtype=np.float32).reshape(73257, 32*32, 3)
        teimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/svhn_teimg.bin", dtype=np.float32).reshape(26032, 32*32, 3)
       
        ntrain = len(trimg)
        ntest = len(teimg)
        npixel = trimg.shape[1]
        
        # tract = np.zeros((ntrain, npixel, k))
        # teact = np.zeros((ntest, npixel, k))
        # for pixelid in range(npixel):            
        #     print (f"Pixel: {pixelid}")
        #     model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=3, verbose=0)  
        #     model.fit(trimg[:, pixelid])                
        #     trpred = model.predict_proba(trimg[:, pixelid])        
        #     tepred = model.predict_proba(teimg[:, pixelid])
        #     tract[:, pixelid, :] = trpred
        #     teact[:, pixelid, :] = tepred
        #     if pixelid%100==0: print (np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
                    
        trimg = trimg.reshape(-1,3)
        teimg = teimg.reshape(-1,3)
        
        # model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=3, verbose=True)
        
        # ntrpat = 1000000
        # randidx = np.arange(len(trimg))
        # np.random.shuffle(randidx)
        
        # model.fit(trimg[randidx][:ntrpat])
        
        # print ("\nGMM cluster centers\n", np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
        
        # tract = model.predict_proba(trimg)    
        # tract= tract.reshape(-1,32*32,k)
        
        # teact = model.predict_proba(teimg)
        # teact = teact.reshape(-1,32*32,k)
        
        os.makedirs(f"{WhitenGmmDir}", exist_ok=True)
        os.makedirs(f"{WhitenGmmDir}/eps{format_float(eps)}", exist_ok=True)
        os.makedirs(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}", exist_ok=True)
        
        tract = np.zeros((ntrain, npixel, k))
        teact = np.zeros((ntest, npixel, k))

        tract.astype('float32').tofile(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}/svhn_trimg.bin")
        teact.astype('float32').tofile(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}/svhn_teimg.bin")
            
        print (f"Whiten Gmm eps={eps} k={k} done.")
        
def plotgrid(img, lbl, label_names, savedir):
    # img = (img-img.min())/(img.max()-img.min())
    # for label in range(10):
    #     labelid = np.argwhere(label==lbl)
    #     fig, axs = plt.subplots(10, 10, figsize=(6,6))
    #     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    #     for x in range(10):
    #         for y in range(10):
    #             idx = labelid[x*10+y]
    #             axs[x,y].imshow(img[idx].reshape(32,32,3))
    #             axs[x,y].axis('off') 
    #     plt.savefig(f"{savedir}/{label_names[label]}.png", dpi=300)
    #     plt.close()
    return

if __name__ == "__main__":

    DownloadDir = "Download"
    RawDir = "Raw"
    KmeansDir = "Kmeans"
    GmmDir = "Gmm"
    WhitenDir = "Whiten"
    WhitenKmeansDir = "WhitenKmeans"
    WhitenGmmDir = "WhitenGmm"
    all_eps = [1e-1] #, 1e-2, 1e-3, 1e-4, 1e-5]:

    label_names = get_label_names()
    download()
    make_raw()
    # make_kmeans(k=20)
    # make_gmm(k=20)  
    #for eps in all_eps:
    #    make_whiten(eps=eps)
    # make_whitenkmeans(k=20)
    make_whitengmm(k=20)
    print ("Fin.")
