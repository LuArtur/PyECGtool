#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:13:37 2019

@author: aasl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import DBSCAN
from scipy.io.matlab import loadmat
from os.path import dirname, join

refractory_period = 54


def mode(vec):
    return np.argmax(np.bincount(vec))


def filter_output_fir(detect, prediction):

    ##Filtering Stage 1
    Ladj = len(detect)
    start_sz = np.int32(np.round(100*np.sum(detect)/Ladj))
#    start_sz = 8
    remainder = start_sz % 2
    
    if remainder == 1:
        filter_size = np.max([start_sz, 3])
    else:
        filter_size = np.max([start_sz-1, 3])
    
    b = 1/filter_size * np.ones(filter_size)
    
    fpred = signal.filtfilt(b, 1, detect) 
    
#    plt.figure()
#    plt.stem(fpred)
#    plt.show()
#    
    fpred2 = fpred ** 2
    
    thr = ((np.floor(filter_size/2) - 1) / filter_size) ** 2
    
    detect_unpad = (fpred2 > thr)
    
    detect2 = detect_unpad
    
#    plt.figure()
#    plt.stem(detect2)
#    plt.show()
    
    #Padding
    
    for u  in range(2):
        pad_val = np.pad(detect2, (1, 1), 'constant', constant_values=0)
        detect2 = np.array(pad_val[2:]) + np.array(pad_val[0:-2]) + detect2
    
    
#    plt.figure()
#    plt.plot(detect2)
#    plt.show()
    
    ##Filtering Stage 2
            
    b2 = np.array([-1, 0, 1])
    
    detect2_filt = signal.filtfilt(b2, 1, detect2) 
    
#    plt.figure()
#    plt.plot(detect2_filt)
#    plt.show()
    
    detect6 = np.array(detect2_filt == 1)
    detect4 = np.array(-detect2_filt == 1)
    detect5 = np.pad(detect4, (1, 1), 'constant', constant_values=0)
    detect7 = np.array(detect5[0:-2]) & detect6 
    detect8 = np.array(detect5[2:]) & detect6
    detect3 = detect7 | detect8
    
    positions = np.array(np.nonzero(detect3)).squeeze()
    
#    plt.figure()  este en el original
#    plt.plot(positions)
#    plt.show()
    
    return positions


def filter_output_dbscan(prediction, hw_size):
    
    detect = np.array(prediction != 2, dtype='int').squeeze()

    detect2 = np.array(np.nonzero(detect)).squeeze()

    detect3 = detect2.reshape(-1, 1)
    db = DBSCAN(eps=10, min_samples=hw_size, metric='cityblock')
    clusters = db.fit_predict(detect3)
    
    top = np.max(clusters) + 1
    
    class_out = np.zeros(top-1)
    pos_out = np.zeros(top-1, dtype='int')
    
    last_pos = 0
    
    for i in range(top-1):
        idx1 = clusters == i
        val = np.array(detect2[idx1])
        
        if (val[0] - last_pos) < refractory_period and last_pos != 0:
            continue
        else:
            last_pos = val[0]
        
        pos_out[i] = int(np.floor(np.median(val + 64)))
        class_out[i] = mode(prediction[val])
    
    return (class_out[np.nonzero(pos_out)], pos_out[np.nonzero(pos_out)])


def process_positions(positions, prediction):

    diffA = positions[1] - positions[0]
    diffB = positions[2] - positions[1]
    
    if diffA > diffB:
        positions = np.array(positions[1:])
        
    
    N = len(positions)
    
    starts = np.array([positions[i] for i in range(0, N, 2)])
    
    ends  = np.array([positions[i] for i in range(1, N, 2)])
    
    out_list = list()
    pos_list = list()
    
    last_start = 0
    
    for (i, j) in zip(starts, ends):
        
        if (i - last_start) < refractory_period and last_start != 0:
            continue
        else:
            last_start = i
            
        segm = np.array(prediction[i:j])
        mode = np.argmax(np.bincount(segm[segm != 2]))
        pos = 64 + i + ((j - i) >> 1)
        out_list.extend([mode])
        pos_list.extend([pos])

    return (out_list, pos_list)


def load_record(filename, mask_val, debug=False, Len_s=1):
    
    matfile = loadmat(join(dirname(__file__),'slmit_' + filename + '.mat'))
    targets = np.array(matfile['trg']).squeeze() - 1;
    trg_pos = np.array(matfile['peakpos']).squeeze() - 1;
    
    if debug == True:
        s = np.array(matfile['s']).squeeze();
        top = Len_s*360
#        plt.figure()
#        plt.plot(s[0:top])
#        plt.show()
        print(targets[trg_pos < top])
        print(trg_pos[trg_pos < top])
        
    targets[targets == 1] = 0
    targets[targets == 2] = 1
    targets[targets == 3] = 1
    targets[targets == 4] = 2
    targets[targets == 5] = 2
    
    return (targets, trg_pos)


def match_output_targets(pos_val, out_val, trg_pos, targets):
    
    H = len(trg_pos)
    
    loc_qrs = np.zeros(H)
    val_qrs = np.zeros(H)
     
    for i in range(H-1):
        temp_pos = np.argmin(np.abs(pos_val - trg_pos[i]))
        loc_qrs[i] = pos_val[temp_pos]
        val_qrs[i] = out_val[temp_pos]


    valid_ind = (np.ediff1d(loc_qrs, to_begin=1) != 0)
    invalid_ind = np.max([len(pos_val) - len(valid_ind), 0])

    return (val_qrs, loc_qrs, valid_ind, invalid_ind)


def compute_performance(val_qrs, loc_qrs, targets, trg_pos, valid_ind, invalid_ind):
    
    total_qrs = len(trg_pos) 

    fn_qrs = total_qrs - np.sum(valid_ind)
    fp_qrs = np.sum(invalid_ind)
    
    error_qrs = fn_qrs+fp_qrs
    error_qrsp = (fn_qrs+fp_qrs) / total_qrs
    
    detected_qrs = np.sum(valid_ind)
    
    acc_qrs = np.mean(valid_ind)
    
    tp = np.sum((val_qrs[valid_ind] == targets[valid_ind]) & (val_qrs[valid_ind] == 1))
    tn = np.sum((val_qrs[valid_ind] == targets[valid_ind]) & (val_qrs[valid_ind] == 0))
    
    fp = np.sum((val_qrs[valid_ind] != targets[valid_ind]) & (val_qrs[valid_ind] == 1))
    fn = np.sum((val_qrs[valid_ind] != targets[valid_ind]) & (val_qrs[valid_ind] == 0))
    
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tn + tp + fp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    conf_mtx = {'TP':tp, 'FP':fp, '+P':ppv, 
                'FN':fn, 'TN':tn, '-P':npv,
                'Se':se, 'Sp':sp, 'Acc':acc}
    
    qrs_data = {'Total':total_qrs, 'FN':fn_qrs, 'FP':fp_qrs, 
                'Error':error_qrs, 'ErrorR':error_qrsp, 
                'Detected':detected_qrs, 'Acc':acc_qrs}
    
    return (conf_mtx, qrs_data)

