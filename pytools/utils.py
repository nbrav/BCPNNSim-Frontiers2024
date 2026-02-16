import numpy as np

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

def getclassifier(h_train, o_train, h_test, o_test, h_valid=None, o_valid=None, NEPOCH=25, BATCH_SIZE=100, PATIENCE=5):
    import tensorflow as tf
    # set shapes
    ni = h_train.shape[1]
    no = o_train.shape[1]
    optimizer='adam'
    # building the network
    layer1 = tf.keras.layers.Input(shape=(ni,))
    outputs = tf.keras.layers.Dense((no), activation='softmax')(layer1)
    # train classifier        
    classifier = tf.keras.Model(inputs=layer1, outputs=outputs)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    classifier.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['categorical_accuracy'])    
    history = classifier.fit(h_train, o_train, batch_size=BATCH_SIZE, epochs=NEPOCH, shuffle=True, validation_split=0.1, callbacks=[callback])
    tr_loss, tr_acc = classifier.evaluate(h_train, o_train)
    te_loss, te_acc = classifier.evaluate(h_test, o_test)
    if not h_valid == None and not o_valid: val_loss, val_acc = classifier.evaluate(h_valid, o_valid)
    del classifier
    return tr_acc, te_acc

