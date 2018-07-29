#!/usr/bin/env python3
"""
Created on 2018-07-29

AUTHORS: Enrico Sanna - Unversita' degli Studi Guglielmo Marconi - Rome (IT)

PURPOSE: This script contains utility for read e manipulate signals data of the PHYSIONET / CinC Challenge 2018
         Developed from the Example script submitted by Mohammad M. Ghassemi, Benjamin E. Moody

REQUIREMENTS: We assume that you have downloaded the data from
              https://physionet.org/physiobank/database/challenge/2018/#files
"""

import os
import numpy as np
import pandas as pd
from pylab import find
import scipy.io
from statistics import mode
import matplotlib
import warnings
#import glob
#import datetime

#from sklearn.externals import joblib

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

p_DATASET_DIR='/opt/PHYSIONET/ch2018_mini'
#'/media/ensanna/Disco Dati 2/PHYSIONET_DATASET/ch_2018'
# -----------------------------------------------------------------------------
# returns a list of the training and testing file locations for easier import
# -----------------------------------------------------------------------------
def get_files():
    print("eseguo phic get files")
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    #rootDir = '.' esanna
    rootDir = p_DATASET_DIR
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):  #sorted(
        if dirName != '.' and dirName != './test' and dirName != './training':
            if dirName.startswith(rootDir+'/training/'):
                is_training.append(True)

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

            elif dirName.startswith(rootDir+'/test/'):
                is_training.append(False)
                arousal_loc.append('')

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

    # combine into a data frame
    data_locations = {'header':      header_loc,
                      'arousal':     arousal_loc,
                      'signal':      signal_loc,
                      'is_training': is_training
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))
    te_ind = list(find(df.is_training.values == False))

    training_files = df.loc[tr_ind, :]
    testing_files  = df.loc[te_ind, :]

    return training_files, testing_files

# -----------------------------------------------------------------------------
# import the outcome vector, given the file name.
# e.g. /training/tr04-0808/tr04-0808-arousal.mat
# -----------------------------------------------------------------------------
def import_arousals(file_name):
    import h5py
    import numpy
    try:
        f = h5py.File(file_name, 'r')
        arousals = numpy.array(f['data']['arousals'])
        arousals = arousals[220000:230000]
    except:
        arousals=None
    return arousals

def import_signals(file_name):
    return np.transpose(scipy.io.loadmat(file_name)['val'])[220000:230000]

# -----------------------------------------------------------------------------
# Take a header file as input, and returns the names of the signals
# For the corresponding .mat file containing the signals.
# -----------------------------------------------------------------------------
def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

# -----------------------------------------------------------------------------
# Get a given subject's data
# -----------------------------------------------------------------------------
def get_subject_data(arousal_file, signal_file, signal_names):
    this_arousal   = import_arousals(arousal_file)
    this_signal    = import_signals(signal_file)
    #if arousal file is not found
    #print("this_signal shape: " + str(this_signal.shape))
    if this_arousal is None:
        shape_x, shape_y=this_signal.shape
        this_arousal = np.zeros((shape_x,1))
        #, dtype=np.float32
    #print("this_arousal shape: " + str(this_arousal.shape))
    this_data      = np.append(this_signal, this_arousal, axis=1)
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data

def get_subject_data_test(signal_file, signal_names):
    this_signal    = import_signals(signal_file)
    this_data      = this_signal
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data

# -----------------------------------------------------------------------------
#  Transform data to matrix of windows_size_length
# -----------------------------------------------------------------------------
def signalsToMatrix(data, arousal,recordLength,WINDOW_SIZE):
    #recordLength = data.size
    # the number of samples is the entire part of the division between window_size and record length
    #print(str(recordLength))
    a=int(recordLength)
    b = int(WINDOW_SIZE)
    samples = int(int(recordLength) / int(WINDOW_SIZE))
    trainset = np.zeros((samples, WINDOW_SIZE))
    traintarget = np.zeros((samples, 3))

    for i in range(1, samples):
        sample_from = i * WINDOW_SIZE;
        trainset[i] = data[sample_from:sample_from + WINDOW_SIZE][0]
        arousalArray = arousal[sample_from:sample_from + WINDOW_SIZE]
        arousalLabels = np.zeros((WINDOW_SIZE, 3))

        arousalArray= arousalArray[:,0]
        # prendo come elemento il valore più diffuso nel campione target
        try:
            arousalArray = mode(arousalArray)
        except:
            try:
                arousalArray = np.amax(arousalArray)
            except:
                #if 50% of two values i choose the first character
                arousalArray=arousalArray[0]

        #print("arousalArray "+str(arousalArray))
        #print("arousalArray shape "+str(arousalArray.shape))

        if arousalArray == 0:
            arousalLabels[0, 0] = 1  # print("messo a zero")
        if arousalArray == 1:
            arousalLabels[0, 1] = 1
        if arousalArray == -1:
            arousalLabels[0, 2] = 1  # print("messo a  - uno")

        traintarget[i]=arousalLabels[0]
        #print("traintarget ["+str(i)+"]: "+ str(traintarget[i]))

    return trainset, traintarget