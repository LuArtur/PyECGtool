#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:34:46 2020

@author: luis-arturo
"""

from scipy.io.matlab import loadmat
import numpy as np
import math
#from keras.models import load_model


from os.path import dirname, join


#model = load_model(join(dirname(__file__),'data_mitdb2_53_01.hdf5'))


def ecgsignal(direccion,value_min):
    signal=loadmat(str(direccion))
    ecg=signal['s']
    ecg_red=ecg[1:21600*value_min]
    ecg_red=np.hstack(ecg_red)



    return ecg_red


