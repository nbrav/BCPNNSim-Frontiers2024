import numpy as np
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
    From https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    os.makedirs(f"{DownloadDir}", exist_ok=True) # Create directory
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    os.system(f"wget -nc -P {DownloadDir} {url}") # Download from url
    os.system(f"tar --skip-old-files -xzvf {DownloadDir}/cifar-10-python.tar.gz -C {DownloadDir} ") # Extract

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_label_names():

    TarDir = f"{DownloadDir}/cifar-10-batches-py"
    
    meta = unpickle(f"{TarDir}/batches.meta")
    label_names = [x.decode('ASCII') for x in meta[b'label_names']]

    return label_names

def make_raw():

    TarDir = f"{DownloadDir}/cifar-10-batches-py"
    
    trimg = np.zeros((50000, 32*32*3))
    trlbl = np.zeros((50000))
    teimg = np.zeros((10000, 32*32*3))
    telbl = np.zeros((10000))
    
    for batchid in range(1, 6):
        dict = unpickle(f"{TarDir}/data_batch_{batchid}")
        trimg[(batchid-1)*10000:(batchid)*10000] = dict[b'data']/255.
        trlbl[(batchid-1)*10000:(batchid)*10000] = np.asarray(dict[b'labels'])

    dict = unpickle(f"{TarDir}/test_batch")
    teimg = dict[b'data']/255.
    telbl = np.asarray(dict[b'labels'])

    trlbl = np.asarray(trlbl, dtype=np.int32)
    telbl = np.asarray(telbl, dtype=np.int32)

    trimg = trimg.reshape(50000, 3, 32*32).transpose(0, 2, 1)
    teimg = teimg.reshape(10000, 3, 32*32).transpose(0, 2, 1)
    trlbl_onehot = np.eye(10)[trlbl]
    telbl_onehot = np.eye(10)[telbl]

    os.makedirs(f"{RawDir}", exist_ok=True)
    
    trimg.astype('float32').tofile(f"{RawDir}/cifar10_trimg.bin")
    trlbl_onehot.astype('float32').tofile(f"{RawDir}/cifar10_trlbl.bin")
    teimg.astype('float32').tofile(f"{RawDir}/cifar10_teimg.bin")
    telbl_onehot.astype('float32').tofile(f"{RawDir}/cifar10_telbl.bin")

    plotgrid(teimg, telbl, label_names, savedir=RawDir)
    
def make_kmeans(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    trimg = np.fromfile(f"{RawDir}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32, 3)
    teimg = np.fromfile(f"{RawDir}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32, 3)
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

    tract.astype('float32').tofile(f"{KmeansDir}/k{k}/cifar10_trimg.bin")
    teact.astype('float32').tofile(f"{KmeansDir}/k{k}/cifar10_teimg.bin")

def make_gmm(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    trimg = np.fromfile(f"{RawDir}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32, 3)
    teimg = np.fromfile(f"{RawDir}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32, 3)
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

    tract.astype('float32').tofile(f"{GmmDir}/k{k}/cifar10_trimg.bin")
    teact.astype('float32').tofile(f"{GmmDir}/k{k}/cifar10_teimg.bin")

def make_whiten(eps=1e-2):
    '''
    Sources
    1. https://cs231n.github.io/neural-networks-2/
    2. https://www.kdnuggets.com/2018/10/preprocessing-deep-learning-covariance-matrix-image-whitening.html/3
    3. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''
    trimg = np.fromfile(f"{RawDir}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32*3)
    teimg = np.fromfile(f"{RawDir}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32*3)
    trlbl = np.fromfile(f"{RawDir}/cifar10_trlbl.bin", dtype=np.float32).reshape(50000, 10)
    telbl = np.fromfile(f"{RawDir}/cifar10_telbl.bin", dtype=np.float32).reshape(10000, 10)

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
    
    trimg_whiten.astype('float32').tofile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_trimg.bin")
    teimg_whiten.astype('float32').tofile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_teimg.bin")

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
        
        trimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32, 3)
        teimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32, 3)
       
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
        
        tract.astype('float32').tofile(f"{WhitenKmeansDir}/eps{format_float(eps)}/k{k}/cifar10_trimg.bin")
        teact.astype('float32').tofile(f"{WhitenKmeansDir}/eps{format_float(eps)}/k{k}/cifar10_teimg.bin")

        print (f"Whiten Kmeans eps={eps} k={k} done.")
    
def make_whitengmm(k=10):
    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
    for eps in all_eps:
        
        trimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32, 3)
        teimg = np.fromfile(f"{WhitenDir}/eps{format_float(eps)}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32, 3)
       
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
        
        model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=3, verbose=True)
        
        ntrpat = 1000000
        randidx = np.arange(len(trimg))
        np.random.shuffle(randidx)
        
        model.fit(trimg[randidx][:ntrpat])
        
        print ("\nGMM cluster centers\n", np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
        
        tract = model.predict_proba(trimg)    
        tract= tract.reshape(-1,32*32,k)
        
        teact = model.predict_proba(teimg)
        teact = teact.reshape(-1,32*32,k)
        
        os.makedirs(f"{WhitenGmmDir}", exist_ok=True)
        os.makedirs(f"{WhitenGmmDir}/eps{format_float(eps)}", exist_ok=True)
        os.makedirs(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}", exist_ok=True)
        
        tract = np.zeros((ntrain, npixel, k))
        teact = np.zeros((ntest, npixel, k))

        tract.astype('float32').tofile(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}/cifar10_trimg.bin")
        teact.astype('float32').tofile(f"{WhitenGmmDir}/eps{format_float(eps)}/k{k}/cifar10_teimg.bin")
            
        print (f"Whiten Gmm eps={eps} k={k} done.")

def dog(img, sigma1, sigma2):
    import cv2
    # Gaussian blurs
    blur1 = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma1, borderType=cv2.BORDER_REPLICATE)
    blur2 = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma2, borderType=cv2.BORDER_REPLICATE)
    # Compute difference
    img_dog = blur2 - blur1
    return img_dog

def make_dog(sigma1=1, sigma2=2):
    import cv2
    # Load Raw files
    trimg = np.fromfile(f"{RawDir}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32, 32, 3)
    teimg = np.fromfile(f"{RawDir}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32, 32, 3)
    trlbl = np.fromfile(f"{RawDir}/cifar10_trlbl.bin", dtype=np.float32).reshape(50000, 10)
    telbl = np.fromfile(f"{RawDir}/cifar10_telbl.bin", dtype=np.float32).reshape(10000, 10)
    # Convert RGB to Grayscale
    trimg_bw = .2126 * trimg[:,:,:,0] + .7152 * trimg[:,:,:,1] + .0722 * trimg[:,:,:,2]
    teimg_bw = .2126 * teimg[:,:,:,0] + .7152 * teimg[:,:,:,1] + .0722 * teimg[:,:,:,2]
    # Set parameters
    gamma = 0.2
    # gamma correction
    trimg_gamma = np.power(trimg_bw, gamma)
    teimg_gamma = np.power(teimg_bw, gamma)
    # Initialize dog data
    trimg_dog = np.zeros((len(trimg), 32, 32))
    teimg_dog = np.zeros((len(teimg), 32, 32))
    for patid in range(len(trimg)):
        # Read image
        img = trimg_gamma[patid]
        # DoG
        img_dog = dog(img, sigma1, sigma2)
        # normalize by the largest absolute value so range is -1 to 
        #img_dog = img_dog / np.amax(np.abs(img_dog))
        # Store Dogs
        trimg_dog[patid] = img_dog
    for patid in range(len(teimg)):
        # Read image
        img = teimg_gamma[patid]
        # DoG
        img_dog = dog(img, sigma1, sigma2)
        # normalize by the largest absolute value so range is -1 to 
        #img_dog = img_dog / np.amax(np.abs(img_dog))
        # Store Dogs
        teimg_dog[patid] = img_dog
    print (teimg_dog.min, teimg_dog.max())
    telbl = np.argmax(telbl, axis=1)
    plotgrid(teimg_dog.reshape(-1,32,32), telbl, label_names, savedir=f"{DogDir}/")
    # save results
    os.makedirs(f"{DogDir}", exist_ok=True)
    trimg_dog.astype('float32').tofile(f"{DogDir}/cifar10_trimg.bin")
    teimg_dog.astype('float32').tofile(f"{DogDir}/cifar10_teimg.bin")
        
def make_doggmm(k=10):

    '''
    Sources
    1. Coates, A., Ng, A.Y. (2012). Learning Feature Representations with K-Means.
    '''
        
    trimg = np.fromfile(f"{DogDir}/cifar10_trimg.bin", dtype=np.float32).reshape(50000, 32*32)
    teimg = np.fromfile(f"{DogDir}/cifar10_teimg.bin", dtype=np.float32).reshape(10000, 32*32)
    
    ntrain = len(trimg)
    ntest = len(teimg)
    npixel = trimg.shape[1]
    
    trimg = trimg.reshape(-1,1)
    teimg = teimg.reshape(-1,1)
    
    model = GaussianMixture(n_components=k, covariance_type="diag", tol=1e-3, n_init=3, verbose=True)
    
    ntrpat = 1000000
    randidx = np.arange(len(trimg))
    np.random.shuffle(randidx)
    
    model.fit(trimg[randidx][:ntrpat])
    
    print ("\nGMM cluster centers\n", np.round(model.weights_, 3), np.round(model.means_, 3), np.round(model.covariances_, 3))
    
    tract = model.predict_proba(trimg)    
    tract = tract.reshape(-1,32*32,k)
    
    teact = model.predict_proba(teimg)
    teact = teact.reshape(-1,32*32,k)
    
    os.makedirs(f"{DogGmmDir}", exist_ok=True)
    os.makedirs(f"{DogGmmDir}/k{k}", exist_ok=True)
    
    tract = np.zeros((ntrain, npixel, k))
    teact = np.zeros((ntest, npixel, k))
    
    tract.astype('float32').tofile(f"{DogGmmDir}/k{k}/cifar10_trimg.bin")
    teact.astype('float32').tofile(f"{DogGmmDir}/k{k}/cifar10_teimg.bin")
    
    print (f"Dog Gmm k={k} done.")
    
def plotgrid(img, lbl, label_names, savedir):
    img = (img-img.min())/(img.max()-img.min())
    for label in range(10):
        labelid = np.argwhere(label==lbl)
        fig, axs = plt.subplots(10, 10, figsize=(6,6))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        for x in range(10):
            for y in range(10):
                idx = labelid[x*10+y]
                axs[x,y].imshow(img[idx][0], cmap="binary")
                axs[x,y].axis('off') 
        plt.savefig(f"{savedir}/{label_names[label]}.png", dpi=300)
        #plt.show()
        plt.close()
    return

if __name__ == "__main__":

    DownloadDir = "Download"
    RawDir = "Raw"
    KmeansDir = "Kmeans"
    GmmDir = "Gmm"
    all_eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    WhitenDir = "Whiten"
    WhitenKmeansDir = "WhitenKmeans"
    WhitenGmmDir = "WhitenGmm"
    DogDir = "Dog"
    DogGmmDir = "DogGmm"

    download()
    label_names = get_label_names()
    #make_raw()
    #make_kmeans(k=20)
    #make_gmm(k=20)  
    #for eps in all_eps:
    #    make_whiten(eps=eps)
    #make_whitenkmeans(k=20)
    #make_whitengmm(k=20)
    #make_dog()
    #make_doggmm(k=20)
    #small_dog()
    print ("Fin.")
