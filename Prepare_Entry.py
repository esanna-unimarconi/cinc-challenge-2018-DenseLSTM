#!/usr/bin/env python3
"""
Created on 2018-07-29

AUTHORS: Enrico Sanna - Unversita' degli Studi Guglielmo Marconi - Rome (IT)

PURPOSE: This script prepares an entry for the physionet 2018 Challenge.
         Developed from the Example script submitted by Mohammad M. Ghassemi, Benjamin E. Moody

REQUIREMENTS: We assume that you have downloaded the data from
              https://physionet.org/physiobank/database/challenge/2018/#files
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import os
import sys
import glob
import physionetchallenge2018_lib as phyc
from score2018 import Challenge2018Score
from pylab import find
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import datetime

import my_logger as L
import train_model_Resnet as T

#import run_tf_Sanna_classifier as R

# -----------------------------------------------------------------------------
# Generate the data to train the classifier
# -----------------------------------------------------------------------------
def train():
    T.init()

    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()

    # For each subject in the training set...
    for i in range(0, np.size(tr_files, 0)):
        gc.collect()
        print('Preprocessing training subject: %d/%d'
              % (i + 1, np.size(tr_files, 0)))
        record_name = tr_files.header.values[i][:-4]
        T.preprocess_record(record_name)

    #T.trainModel()

    T.finish()

# -----------------------------------------------------------------------------
# Run the classifier on each training subject, and compute the mean performance
# -----------------------------------------------------------------------------
def score_training_set():

    try:
        os.mkdir('training_output')
    except OSError:
        pass

    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()

    score = Challenge2018Score()
    for i in range(0, np.size(tr_files, 0)):
        gc.collect()
        sys.stdout.write('Evaluating training subject: %d/%d'
                         % (i + 1, np.size(tr_files, 0)))
        sys.stdout.flush()
        record_name = tr_files.header.values[i][:-4]
        predictions, pred_arousal_probabilities = T.classify_record(record_name)

        arousals = phyc.import_arousals(tr_files.arousal.values[i])
        arousals = np.ravel(arousals)

        print("arousals.shape: "+str(arousals.shape))
        print("predictions.shape: "+str(predictions.shape))

        score.score_record(arousals, pred_arousal_probabilities, record_name)
        auroc = score.record_auroc(record_name)
        auprc = score.record_auprc(record_name)
        L.log_info(' AUROC:%f AUPRC:%f' % (auroc, auprc))

        # save also training predictions to evaluate voting solutions between different models
        output_file = "training_output/" + os.path.basename(record_name) + '.vec'
        L.log_info("Salvo i files esito del training in " + str(output_file))
        np.savetxt(output_file, pred_arousal_probabilities, fmt='%.3f')

    L.log_info("\n\r")
    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    L.log_info('Training AUROC Performance (gross): %f' % auroc_g)
    L.log_info('Training AUPRC Performance (gross): %f' % auprc_g)
    L.log_info("\n\r")

# -----------------------------------------------------------------------------
# Run the classifier on each test subject, and save the predictions
# for submission
# -----------------------------------------------------------------------------
def evaluate_test_set():
    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()
    try:
        os.mkdir('test_output')
    except OSError:
        pass

    for f in glob.glob('models/*.vec'):
        os.remove(f)

    for i in range(0, np.size(te_files, 0)):
        gc.collect()
        print('Evaluating test subject: %d/%d' % (i+1, np.size(te_files, 0)))
        record_name = te_files.header.values[i][:-4]
        output_file = "test_output/"+os.path.basename(record_name) + '.vec'
        print("Salvo i files esito del test in "+str(output_file))
        predictions, pred_arousal_probabilities = T.classify_record(record_name)
        np.savetxt(output_file, pred_arousal_probabilities, fmt='%.3f')

# -----------------------------------------------------------------------------
# Build a zip file for submission to the Challenge
# -----------------------------------------------------------------------------
def package_entry():
    try:
        os.mkdir('test_output')
    except OSError:
        pass

    for f in glob.glob('models/*.zip'):
        os.remove(f)

    with ZipFile('test_output/' + str(T.p_ENTRY_ZIP_FILE), 'w', ZIP_DEFLATED) as myzip:
        for dirName, subdirList, fileList in os.walk('.'):
            for fname in fileList:
                if ('.vec' in fname[-4:] or '.py' in fname[-3:]
                        or '.pkl' in fname[-4:] or '.txt' in fname[-4:]
                        or '.log' in fname[-4:] or '.hdf5' in fname[-5:]):
                    myzip.write(os.path.join(dirName, fname))

# -----------------------------------------------------------------------------
#    MAIN FUNCTION
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    L.init_logger(T.p_LOG_FILE)
    dtInizioElaborazione = datetime.datetime.now()
    L.log_info("Execution starts at: " + str(dtInizioElaborazione))
    try:
        L.log_info("############################## TRAIN ##############################")
        train()
        L.log_info("###################### SCORE TRAINING SET #########################")
        score_training_set()
        L.log_info("###################### EVALUATE TEST SET  #########################")
        evaluate_test_set()
        L.log_info("####################### PACKAGE ENTRY #############################")
        package_entry()
        dtFineElaborazione = datetime.datetime.now()
        elapsedTime = dtFineElaborazione - dtInizioElaborazione
        L.log_info("Execution stops at: " + str(dtInizioElaborazione))
        L.log_info(" Elapsed Time: " + str(elapsedTime))
        print("Launching Tensorboard...")
        os.system('tensorboard --logdir=tensorboard/' + str(T.p_TENSORBOARD_LOGDIR))
    except:
        dtFineElaborazione = datetime.datetime.now()
        elapsedTime = dtFineElaborazione - dtInizioElaborazione
        L.log_info("Execution stops with errors at: " + str(dtInizioElaborazione))
        L.log_info(" Elapsed Time: " + str(elapsedTime))
        raise
