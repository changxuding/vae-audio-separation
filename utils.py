"""
Created on 09.09.2019
@author: Changxu Ding
"""

import numpy as np
import librosa
import json
import logging
import os

def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
    return y

def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)

def mix_tracks(data_dict):
    """
    Mixt mehrere Signale zu einem Gesamtsignal. Dieses hat dann die Länge des längsten Signals.
    Es wird der Mix sowie die aufgefüllten Signale zurückgegeben
    """
    # Ermittle maximale Lämge für Zero-Padding
    max_length = 0
    for instr in data_dict:
        max_length = np.maximum(max_length, data_dict[instr].size)

    for instr in data_dict:
        # Zero-Padding
        data_dict[instr] = np.append(data_dict[instr], np.zeros(max_length - data_dict[instr].size))

        # Normierung auf jeweiliges Betragsmaximum
        data_dict[instr] = data_dict[instr] / np.max(np.abs(data_dict[instr]))

    # Summe der Daten
    data_mix = np.sum([data_dict[instr] for instr in data_dict], axis=0)

    # Normierung auf Maximum des Mixes
    norm = np.max(np.abs(data_mix))
    for instr in data_dict:
        data_dict[instr] = data_dict[instr] / norm

    return data_dict, data_mix / norm

def data_aug(data):
    delta = np.random.uniform(0.7,1.3,1)
    return data*delta


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('new folder')

