#!/usr/bin/env python3
"""
Created on 2018-07-29

AUTHORS: Enrico Sanna - Unversita' degli Studi Guglielmo Marconi - Rome (IT)

PURPOSE: Script to manage shared file logging between modules

"""

import logging
import glob
import os

def init_logger(filename):
    try:
        os.mkdir('logs')
    except OSError:
        pass
    for f in glob.glob('logs/*'):
        os.remove(f)
    logging.basicConfig(filename="logs/" + str(filename),
                    level=logging.DEBUG,
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

def log_info(message):
    print(message)
    logging.info(message)

def log_debug(message):
    print(message)
    logging.debug(message)