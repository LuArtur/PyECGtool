# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:29:37 2020

@author: Artur
"""

import numpy as np
from keras.models import load_model
#from scipy import signal
#from scipy.io.matlab import loadmat
from mitdbseq import MITDBSequence
#from filter_variants import filter_output_dbscan, load_record 
from filter_variants import filter_output_fir, load_record, process_positions 
from filter_variants import match_output_targets, compute_performance
from os.path import dirname, join
from keras.utils import data_utils
data_utils._SEQUENCE_COUNTER = 0


hw_size=5
vector_size=128
batch_size=360
def clasificarecg():
    model = load_model(join(dirname(__file__),'data_mitdb2_52_01.hdf5'),
                       custom_objects=None,
                       compile=False)

    print('Cargando modelo... data_mitdb2_52_01')

    mask_val = np.array([0, 0, 1, 1, 2, 2])

    #global_conf_mtx = {'TP':0, 'FP':0, 'FN':0, 'TN':0}
    #
    #global_qrs_data = {'Total':0, 'FN':0, 'FP':0, 'Detected':0}

    filename='200'

    print('Processing record: '+ filename)

    test_gen = MITDBSequence(filename, vector_size, batch_size, hw_size, mask_val)

    prediction = np.argmax(model.predict_generator(test_gen, test_gen.len, verbose=1), 1)

    detect = np.array(prediction != 2)

    positions = filter_output_fir(detect, prediction)

    out_list, pos_list = process_positions(positions, prediction)

    pos_val = np.array(pos_list)
    out_val = np.array(out_list)

    #out_val, pos_val = filter_output_dbscan(prediction, hw_size)

    targets, trg_pos = load_record(filename, mask_val)

    val_qrs, loc_qrs, valid_ind, invalid_ind = match_output_targets(pos_val,
                                                                    out_val,
                                                                    trg_pos,
                                                                    targets)

    conf_mtx, qrs_data = compute_performance(val_qrs, loc_qrs,
                                             targets, trg_pos,
                                             valid_ind, invalid_ind)

    print('------- Statistics summary for record: ' + filename + ' -------')

    print('QRS detection')
    print('-------------')
    print('Total QRS: '+ str(qrs_data['Total']) + ' Detected: ' +
          str(qrs_data['Detected']) + ' Accuracy: '+ str(qrs_data['Acc']))
    print('FP: ' + str(qrs_data['FP']) + ' FN: '+ str(qrs_data['FN']))
    print('Error: ' + str(qrs_data['Error']) + ' Relative Error: '+
          str(qrs_data['ErrorR']))

    print('VEB detection')
    print('-------------')
    print('TP: ' + str(conf_mtx['TP']) + ' FP: '+ str(conf_mtx['FP']))
    print('FN: ' + str(conf_mtx['FN']) + ' TN: '+ str(conf_mtx['TN']))
    print('Se: ' + str(conf_mtx['Se']) + ' Sp: '+ str(conf_mtx['Sp']))
    print('+P: ' + str(conf_mtx['+P']) + ' Acc: '+ str(conf_mtx['Acc']))

