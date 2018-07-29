#!/usr/bin/env python3
"""
Created on 2018-07-29

AUTHORS: Enrico Sanna - Unversita' degli Studi Guglielmo Marconi - Rome (IT)

PURPOSE: Script that create a CNN Resnet Model to predict values of PYSIONET / CinC Challenge 2018
         Data records are splitted into NxM matrix array, where M is the WINDOW accepted by the network,
         an N is then number of M WINDOWS in the record length.
         Arousal file are splitted in Nx1 array, retrieving max value in M interval.
         Predictions are then repeated M time to create a 1 x record lenght predictions array

         I developed this model from this work, adapting for variable record lenght and multi channel signal
               Andreotti, F., Carr, O., Pimentel, M.A.F., Mahdi, A., & De Vos, M. (2017). Comparing Feature Based
               Classifiers and Convolutional Neural Networks to Detect Arrhythmia from Short Segments of ECG. In
               Computing in Cardiology. Rennes (France).

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import gc
import itertools
from sklearn.metrics import confusion_matrix
import sys
sys.path.insert(0, './preparation')
import os
import glob
import shutil
# Keras imports
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger,TensorBoard
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback,warnings
import my_logger as L

import physionetchallenge2018_lib as phyc
from score2018 import Challenge2018Score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
************************************************************************************
MODEL PARAMETERS
************************************************************************************
Fs=200               frequenza di campionamento segnali
p_WINDOW_SIZE=3*Fs   dimensione della serie storica passata al modello
p_INPUT_FEAT=13      numero di segnali in input
p_OUTPUT_CLASS=3     # 1,0,-1 (total of 3) - numero di classi contenute nel tracciato di target y (arousals)
p_BATCH_SIZE=1000    numero di campioni per volta passati al modello    
p_EPOCHS=75          epoche, numero di volte che lo stesso report viene fatto ripassare.
p_DATASET_DIR        directory del dataset
p_MODEL_FILE         file dove salvo i pesi del modello
p_LOG_FILE           log testuale in CSV
p_TENSORBOARD_LOGDIR directory di log per Tensorboard
************************************************************************************
"""
Fs=200
p_WINDOW_SIZE=3*Fs  # padding window for CNN
p_INPUT_FEAT=1
p_OUTPUT_CLASS=3  # 1,0,-1 (total of 3)
p_BATCH_SIZE=64
p_EPOCHS=1 #75
p_MODEL_NAME="RESNET"
p_MODEL_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)
p_LOG_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+".log"
p_KERAS_LOG_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+"_Keras.log"
p_ENTRY_ZIP_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+"_entry.zip"
p_TENSORBOARD_LOGDIR=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
seed = 7
np.random.seed(seed)

"""
Inzializzo il modello e stampo i parametri
"""
def init():
    # Create the 'models' subdirectory and delete any existing model files
    try:
        os.mkdir('models')
    except OSError:
        pass
    # Create the 'tensorboard' subdirectory
    try:
        os.mkdir('tensorboard')
    except OSError:
        pass

    for f in glob.glob('models/*.hdf5'):
        os.remove(f)
    for f in glob.glob('tensorboard/*'):
        shutil.rmtree(f, ignore_errors=True)

    stringInit = "";
    stringInit += str("\r\n*************************** init ***********************************")
    stringInit += str("\r\nFs (frequenza di campionamento segnali): " + str(Fs))
    stringInit += str(
        "\r\np_WINDOW_SIZE=x*Fs   dimensione della serie storica passata al modello: " + str(p_WINDOW_SIZE))
    stringInit += str("\r\np_INPUT_FEAT=13      numero di segnali in input: " + str(p_INPUT_FEAT))
    stringInit += str(
        "\r\np_OUTPUT_CLASS=3     # 1,0,-1 (total of 3) - numero di classi contenute nel tracciato di target y (arousals):  " + str(
            p_OUTPUT_CLASS))
    stringInit += str("\r\np_BATCH_SIZE=1000    numero di campioni per volta passati al modello " + str(p_BATCH_SIZE))
    stringInit += str(
        "\r\np_EPOCHS=75          epoche, numero di volte che lo stesso report viene fatto ripassare: " + str(p_EPOCHS))
    stringInit += str("\r\np_MODEL_FILE - file dove salvo i pesi del modello:" + str(p_MODEL_FILE))
    stringInit += str("\r\np_DATASET_DIR - directory del dataset:" + str(phyc.p_DATASET_DIR))
    stringInit += str("\r\np_LOG_FILE - log testuale in CSV:" + str(p_LOG_FILE))
    stringInit += str("\r\np_KERAS_LOG_FILE - log testuale in CSV:" + str(p_KERAS_LOG_FILE))
    stringInit += str("\r\np_TENSORBOARD_LOGDIR - directory di log per Tensorboard:" + str(p_TENSORBOARD_LOGDIR))
    stringInit += str("\r\n********************************************************************")
    L.log_info(stringInit)

def finish():
    pass

###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):    
    '''
   # Arguments
       monitor: quantity to be monitored.
       patience: number of epochs with no improvement
           after which training will be stopped.
       verbose: verbosity mode.
       mode: one of {auto, min, max}. In 'min' mode,
           training will stop when the quantity
           monitored has stopped decreasing; in 'max'
           mode it will stop when the quantity
           monitored has stopped increasing.
   '''
    def __init__(self, monitor='val_loss', patience=0,verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__() 
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
 
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'
 
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
 
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
 
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0 
            self.wait += 1


#####################################
## Model definition              ##
## ResNet based on Rajpurkar    ##
################################## 
def ResNet_model():
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    WINDOW_SIZE=p_WINDOW_SIZE
    INPUT_FEAT = p_INPUT_FEAT # da 1 a 13 enrico
    OUTPUT_CLASS = p_OUTPUT_CLASS    # output classes da 4 a 3 enrico

    k = 1    # increment every 4th residual block
    p = True # pool toggle every other residual block (end with 2^8)
    convfilt = 64
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5
    
    # Modelling with Functional API
    #input1 = Input(shape=(None,1), name='input')
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')

    # instantiate model
   # model = Sequential()
    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)
    #enrico aggiunto
    #sx.trainable=False
    #enrico aggiunto
    x = Activation('relu')(x)  
    
    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)      
    x1 = BatchNormalization()(x1)    
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1,x2
    
    ## Main loop
    p = not p 
    for l in range(15):
        
        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
             # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x        
        # Left branch (convolutions)
        # notice the ordering of the operations has changed        
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        if p:
            x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)                

        # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
        else:
            x2 = xshort  # pool or identity            
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p # toggle pooling

    
    # Final bit    
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Flatten()(x)
    #x = Dense(1000)(x)
    #x = Dense(1000)(x)
    out = Dense(OUTPUT_CLASS, activation='softmax')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    #plot_model(model, to_file='model.png')
    return model

###########################################################
## Function to perform K-fold Crossvalidation on model  ##
##########################################################
def model_eval(X,y, p_INPUT_FEAT, p_OUTPUT_CLASS, record_name, record_length):
    batch =p_BATCH_SIZE
    epochs = p_EPOCHS   # reduced from 120
    rep = 1         # K fold procedure can be repeated multiple times
    Kfold = 3   # ENRICO reduced from 5
    #Ntrain = int(record_length/9000) #8528 # number of recordings on training set
    Ntrain = int(record_length/600)
    Nsamp = int(Ntrain/Kfold) # number of recordings to take as validation #/10
    X_train=X

    # Need to add dimension for training
    X = np.expand_dims(X, axis=2)
    classes = ['1', '0', '-1']
    Nclass = len(classes)
    print("Nclass"+str(Nclass))
    # ENRICO la matrice di confusione serve a predirre i valori veri vs trovati
    cvconfusion = np.zeros((Nclass,Nclass,Kfold*rep))
    cvscores = []       
    counter = 0
    # repetitions of cross validation
    for r in range(rep):
        print("Rep %d"%(r+1))
        # cross validation loop
        for k in range(Kfold):
            print("Cross-validation run %d"%(k+1))
            # Callbacks definition
            callbacks = [
                # Early stopping definition
                EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                # Decrease learning rate by 0.1 factor
                AdvancedLearnignRateScheduler(monitor='val_loss', patience=1,verbose=1, mode='auto', decayRatio=0.1),            
                # Saving best model
                ModelCheckpoint('models/'+p_MODEL_FILE+'_k{}_r{}.hdf5'.format(k,r), monitor='val_loss', save_best_only=True, verbose=1),
                CSVLogger('logs/' + p_KERAS_LOG_FILE, separator=',', append=False),
                TensorBoard(log_dir='tensorboard/' + str(p_TENSORBOARD_LOGDIR), histogram_freq=0, batch_size=32,
                            write_graph=True,
                            write_grads=False, write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None),
                ]
            #print("loading model with window_size: "+str(WINDOW_SIZE))
            # Load model
            model = ResNet_model()
            
            # split train and validation sets
            idxval = np.random.choice(Ntrain, Nsamp,replace=False)
            idxtrain = np.invert(np.in1d(range(X_train.shape[0]),idxval))
            ytrain = y[np.asarray(idxtrain),:]
            Xtrain = X[np.asarray(idxtrain),:,:]         
            Xval = X[np.asarray(idxval),:,:]
            yval = y[np.asarray(idxval),:]
            
            # Train model
            #model.fit(Xtrain, ytrain,
            model.fit(X, y,
                      validation_data=(Xval, yval),
                      epochs=epochs, batch_size=batch,callbacks=callbacks)
            
            # Evaluate best trained model
            model.load_weights('models/'+p_MODEL_FILE+'_k{}_r{}.hdf5'.format(k,r))
            #enrico addedd  https://github.com/matterport/Mask_RCNN/issues/588
            model._make_predict_function()
            #ypred = model.predict(Xval)
            ypred = model.predict(X)
            yval=y
            #print("yval SHAPE" + str(yval.shape))
            #print ("ypred SHAPE"+str(ypred.shape)  )

            #confronto la previsione della colonna 0 (quella con  valore 1)
            arousals=yval[:,0]
            predictions=ypred[:,0]

            K.clear_session()
            gc.collect()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True            
            sess = tf.Session(config=config)
            K.set_session(sess)
            counter += 1

    # Saving cross validation results 
    scipy.io.savemat('xval_results.mat',mdict={'cvconfusion': cvconfusion.tolist()})  
    return model,arousals, predictions


###########################
## Function to load data ##
###########################
def loaddata(record_name):
    L.log_info("Loading record: " + str(record_name))
    header_file = record_name + '.hea'
    signal_file = record_name + '.mat'
    arousal_file = record_name + '-arousal.mat'

    # Get the signal names from the header file
    signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
    signal_names = list(np.append(signal_names, 'arousals'))
    print("signal_names: " + str(signal_names))
    print("Fs: " + str(Fs))
    print("n_samples: " + str(n_samples))
    # Convert this subject's data into a pandas dataframe
    this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)

    # ----------------------------------------------------------------------
    # Generate the Features for the classificaition model - variance of SaO2
    # ----------------------------------------------------------------------

    # For the baseline, let's only look at how SaO2 might predict arousals

    SaO2 = this_data.get(['SaO2']).values
    arousals_originale = this_data.get(['arousals']).values
    recordLength = SaO2.size
    SaO2, arousals = phyc.signalsToMatrix(SaO2, arousals_originale, recordLength, p_WINDOW_SIZE);
    return SaO2, arousals,recordLength,arousals_originale


def preprocess_record(record_name):
    (X_train, y_train, recordLength,arousals_originale) = loaddata(record_name)
    model, arousals, predictions = model_eval(X_train, y_train, p_INPUT_FEAT, p_OUTPUT_CLASS, record_name, recordLength)


def classify_record(record_name):
    (X_train, y_train, recordLength, arousals_originale) = loaddata(record_name)

    #print("X shape: " + str(X_train.shape))
    #print("Y shape: " + str(y_train.shape))

    #predictions = model_eval_test(X_train, y_train, p_INPUT_FEAT, p_OUTPUT_CLASS, record_name, recordLength)

    model = ResNet_model()
    k = 0
    r = 0
    model.load_weights('models/' + p_MODEL_FILE + '_k{}_r{}.hdf5'.format(k, r))
    # Need to add dimension for training
    X_train = np.expand_dims(X_train, axis=2)
    predict_classes = model.predict(X_train)
    #Seleziono le probabilita della colonna 1 (quella di arousal ==1
    predictions = predict_classes[:, 1]
    predictions = np.repeat(predictions, p_WINDOW_SIZE)

    #print("X_train: " + str(X_train))
    print("X_train.shape: " + str(X_train.shape))

    #print("y_train: " + str(y_train))
    print("y_train.shape: " + str(y_train.shape))

    #print("arousals_originale: "+str(arousals_originale))
    print("arousals_originale.shape: " + str(arousals_originale.shape))

    #print("predictions: " + str(predictions))
    print("predictions.shape: "+str(predictions.shape))

    #print("predict_classes: " + str(predict_classes))
    print("predict_classes.shape: " + str(predict_classes.shape))

    if (predictions.size < arousals_originale.size):
        #print("correggo differenza")
        differenza = arousals_originale.size - predictions.size
        print("correggo differenza : " + str(differenza))
        predictions = np.pad(predictions, (0, differenza), 'median')
    return predict_classes, predictions


#####################
# Main function   ##
###################
#if __name__ == '__main__':
#    trainModel()