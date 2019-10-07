#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math
import wave
import array
import joblib
import glob

from numba import jit
import numpy as np
import numpy.random as rd
from scipy.signal import lfilter

from settings import settings

import os

# Low Pass Filter for de-emphasis
# @jit
def de_emph(y, preemph=0.95):
    if preemph <= 0:
        return y
    return lfilter([1],[1, -preemph], y)

# Dataset loader
def data_loader(test=False, preemph=0.95, need_length=False):
    """
    Read wav files or Load pkl files
	"""
    lendata         = {
        'name'   : [],
        'length' : []
    }


    ## Sub function : wav read & data shaping
    def wavloader(filename, length, name='wav', get_lendata=False):

        # Error
        num = len(filename)
        if num == 0:
            print('Dataset Error : no wave files.')

        i = 1
        filedata = []
        for filename_ in filename:
            file_ = wave.open(filename_, 'rb')
            wavdata = np.frombuffer(file_.readframes(-1), dtype='int16')
            if get_lendata:
                lendata['name'].append(filename_)
                lendata['length'].append(len(wavdata))
            filedata.append(wavdata)
            file_.close()
            print(' Loading {0} wav... #{1} / {2}'.format(name, i, num))
            i+=1

        filedata = np.concatenate(filedata, axis=0)             # Serializing
        filedata = filedata - preemph * np.roll(filedata, 1)    # Pre-enphasis
        filedata = filedata.astype(np.float32)                  # Data Compressing (float64 -> float32)
        L = length // 2                                         # Half of Input Size (init: 8192 samples)
        D = len(filedata) // L                                  # No. of 0.5s blocks
        if len(filedata) % (D*L) != 0:
            fdata = []
            for f in filedata:
                fdata.append(f)
            zeros = np.zeros(shape=(len(filedata) - L*D), dtype=np.float32)
            for z in zeros:
                fdata.append(z)
            filedata = np.array(fdata, dtype=np.float32)
        filedata = filedata[:D * L].reshape(D, L)               # Split data for each half of input size : (1,:) --> (D, 8192)
        return filedata


	# Load settings
    args = settings()

    # Make folder
    if not os.path.exists(args.model_save_path):    # Folder of model
        os.makedirs(args.model_save_path)

    if not os.path.exists(args.train_pkl_path):     # Folder of train pkl
        os.makedirs(args.train_pkl_path)

    if not os.path.exists(args.test_pkl_path):      # Folder of test pkl
        os.makedirs(args.test_pkl_path)

    # File name
    if not test:
        wav_clean   = args.clean_train_path + '/*.wav'
        wav_noisy   = args.noisy_train_path + '/*.wav'
        pkl_clean   = args.train_pkl_path + '/' + args.train_pkl_clean
        pkl_noisy   = args.train_pkl_path + '/' + args.train_pkl_noisy
    else:
        wav_clean   = args.clean_test_path + '/*.wav'
        wav_noisy   = args.noisy_test_path + '/*.wav'
        pkl_clean   = args.test_pkl_path + '/' + args.test_pkl_clean
        pkl_noisy   = args.test_pkl_path + '/' + args.test_pkl_noisy


    ##   No pkl files -> read wav + create pkl files
    ## -------------------------------------------------
    if not (os.access(pkl_clean, os.F_OK) and os.access(pkl_noisy, os.F_OK)):

        ##  Wav files
        print(' Load wav file...')

	    # Get file path
        cname = glob.glob(wav_clean)
        nname = glob.glob(wav_noisy)

        # Get wave data
        cdata = wavloader(cname, args.len, name='clean', get_lendata=True)  # Clean wav
        ndata = wavloader(nname, args.len, name='noisy')  # Noisy wav

        ##  Pkl files
        print(' Create Pkl file...')

		# Create clean pkl file
        with open(pkl_clean, 'wb') as f:
            joblib.dump(cdata, f, protocol=-1,compress=3)

        # Create noisy pkl file
        with open(pkl_noisy, 'wb') as f:
            joblib.dump(ndata, f, protocol=-1,compress=3)

        #if (not os.access(pkl_length, os.F_OK)) and test:
        #    # Create length pkl file
        #    with open(pkl_length, 'wb') as f:
        #        joblib.dump(lendata, f, protocol=-1,compress=3)

	##  Pkl files exist -> Load
    ## -------------------------------------------------
    else:
        # Load clean pkl file
        print(' Load Clean Pkl...')
        with open(pkl_clean, 'rb') as f:
            cdata = joblib.load(f)

        # Load noisy pkl file
        print(' Load Noisy Pkl...')
        with open(pkl_noisy, 'rb') as f:
            ndata = joblib.load(f)


    return cdata, ndata




class create_batch:
    """
    Creating Batch Data for training
    """

    ## 	Initialization
    def __init__(self, clean_data, noisy_data, batches):

        # Normalization
        def normalize(data):
            return (1. / 32767.) * data  # [-32768 ~ 32768] -> [-1 ~ 1]

        # Data Shaping
        self.clean = np.expand_dims(normalize(clean_data),axis=1)     # (D,8192,1) -> (D,1,8192)
        self.noisy = np.expand_dims(normalize(noisy_data),axis=1)     # (D,8192,1) -> (D,1,8192)

        # Random index ( for data scrambling)
        ind = np.array(range(len(clean_data)-1))
        rd.shuffle(ind)

        # Parameters
        self.batch = batches
        self.batch_num = math.ceil(len(clean_data)/batches)         # Batch num for each 1 Epoch
        self.rnd = np.r_[ind,ind[:self.batch_num*batches-len(clean_data)+1]] # Reuse beggining of data when not enough data
        self.len = len(clean_data)                                  # Data length
        self.index = 0                                              # Start Position for data loading

    def shuffle(self):
        ind = np.array(range(self.len - 1))
        rd.shuffle(ind)
        self.rnd = np.r_[ind,ind[:self.batch_num*self.batch-self.len+1]]

    ## 	Pop batch data
    def next(self, i):

        # Index of extracting data
        index = self.rnd[ i * self.batch : (i + 1) * self.batch ]

        # Reconstructing clean & noisy batch : (*, 1,8192) -> (*, 1,16384)
        return np.concatenate((self.clean[index],self.clean[index+1]),axis=2), \
               np.concatenate((self.noisy[index],self.noisy[index+1]),axis=2)


class create_batch_test:
    """
    Creating Batch Data for test
    """

    ## 	Initialization
    def __init__(self, clean_data, noisy_data, start_frame=None, stop_frame=None):

        def normalize(data):
            return (1. / 32767.) * data  # [-32768 ~ 32768] -> [-1 ~ 1]

        # Processing range
        if start_frame is None:             # Start frame position
            start_frame  = 0
        if stop_frame is None:              # Stop frame position
            stop_frame   = clean_data.shape[0]

        # Parameters
        f_len = clean_data.shape[1] * 2     # Inuput size : 8192*2 = 16384
        stop_frame = 2 * math.floor((stop_frame-start_frame)/2) # Truncate protruded frame
        self.clean = np.expand_dims(normalize(clean_data[start_frame:stop_frame]).reshape(-1, f_len), axis=1)
        self.noisy = np.expand_dims(normalize(noisy_data[start_frame:stop_frame]).reshape(-1, f_len), axis=1)
        self.len = len(clean_data)


def wav_write(filename, x, fs=16000):

    x = de_emph(x)      # De-emphasis using LPF

    x = x * 32767       # denormalized
    x = x.astype('int16')  # cast to int
    w = wave.Wave_write(filename)
    w.setparams((1,     # channel
                 2,     # byte width
                 fs,    # sampling rate
                 len(x),  # #. of frames
                 'NONE',
                 'not compressed' # no compression
    ))
    w.writeframes(array.array('h', x).tobytes())
    w.close()

    return 0
