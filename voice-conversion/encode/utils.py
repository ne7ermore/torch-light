import yaml
import logging
import h5py
from math import ceil
from termcolor import colored
from pysptk import sptk

from scipy import signal
import numpy as np
from numpy.random import RandomState
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler
from librosa.filters import mel


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def read_hdf5(hdf5_name, hdf5_path):
    hdf5_file = h5py.File(hdf5_name, "r")
    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data

def set_logger(context, verbose=False, usefile=None):
    logger = logging.getLogger(colored(context, 'yellow'))
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt='%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if usefile:
        file_handler = logging.FileHandler(usefile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def butter_highpass(cutoff, sr=24000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result) 

def get_config(cfg_file="../pretrained_model/config.yml"):
    tmp = open(cfg_file)
    config = yaml.load(tmp, Loader=yaml.Loader)
    tmp.close()

    return config

def get_generater_scalar(stats="../pretrained_model/stats.h5"):
    scalar = StandardScaler()
    scalar.mean_ = read_hdf5(stats, "mean")
    scalar.scale_ = read_hdf5(stats, "scale")
    scalar.n_features_in_ = scalar.mean_.shape[0]

    return scalar

def amp_to_db(x, min_level_db=-100):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def normalize(S, bias=-16, min_level_db=-100):
    return np.clip((S + bias - min_level_db) / -min_level_db, 0, 1)  

def denormalize(S, bias=-16, min_level_db=-100):
    return ((np.clip(S, 0, 1) * -min_level_db) + min_level_db - bias) / 20

def get_mel_bias_transpose(sr=24000):
    return mel(sr, 1024, fmin=90, fmax=7600, n_mels=80).T  

def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    #index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0

def speaker_f0(wav, sr=24000, lo=100, hi=600):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, sr, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)    

    return f0_norm    