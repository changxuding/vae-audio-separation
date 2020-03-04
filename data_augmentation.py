"""
Created on 15.01.2020
@author: Changxu Ding
"""

import utils
import os
import numpy as np
import librosa
import shutil

source_map = {
    'mix': 0,
    'bn': 1,
    'cl': 2,
    'db': 3,
    'fl': 4,
    'hn': 5,
    'ob': 6,
    'sax': 7,
    'tba': 8,
    'tbn': 9,
    'tpt': 10,
    'va': 11,
    'vc': 12,
    'vn': 13,
}

class Data_aug():
    """
         augmente audio data with desired instruments 
    """
    def __init__(self):
            self.outputs = ['vn','tpt','fl']
            self.train_data_path = 'H:/URMP/train'
            self.aug_num = 50

            self.tracklist = {'mix': [], 'bn': [], 'cl': [], 'db': [], 'fl': [], 'hn': [], 'ob': [], 'sax': [],
                                'tba': [], 'tbn': [], 'tpt': [], 'va': [], 'vc': [], 'vn': []}

    def get_wav(self):
        # get path of each source
        # Iterate through each music
        for folder in os.listdir(self.train_data_path):
            # 1st must be mix  + 13 individual sources
            track_sources = [0 for i in range(14)]
            # save path in list
            for filename in os.listdir(os.path.join(self.train_data_path, folder)):
                if filename.endswith(".wav"):
                    if filename.startswith("AuMix"):
                        mix_path = os.path.join(self.train_data_path, folder, filename)
                        track_sources[0] = mix_path
                    else:
                        source_name = filename.split('_')[2]
                        source_idx = source_map[source_name]
                        source_path = os.path.join(self.train_data_path, folder, filename)
                        track_sources[source_idx] = source_path

            for instr in list(source_map.keys()):
                idex = source_map[instr]
                if track_sources[idex] != 0:
                    self.tracklist[instr].append(track_sources[idex])
        return self

    def augmentation(self):
        # augmente audio with fix number
        for num in range(self.aug_num):
            mix_name = 'AuMix_' + str(num) + '_'+ str(num) + 'song_' + 'vn_tpt_fl.wav'
            file_name = self.train_data_path+'/'+ str(num) + '_'+ str(num) + 'song_' + 'vn_tpt_fl'

            print('augmente Nr.'+ str(num) + 'Music')
            length = 0
            sample_aug=[]
            utils.mkdir(file_name)
            for stem in self.outputs:
                index = np.random.randint(0, len(self.tracklist[stem]),1)
                sample,_ = librosa.load(self.tracklist[stem][index[0]])
                if length >= len(sample):
                    length = len(sample)
                sample_aug.append(utils.data_aug(sample))
                # shutil.copy(self.tracklist[stem][index[0]], file_name)
            mix_length = min(len(sample_aug[0]), len(sample_aug[1]), len(sample_aug[2]))
            mix = sample_aug[0][:mix_length] + sample_aug[1][:mix_length] +sample_aug[2][:mix_length]
            librosa.output.write_wav(file_name +'/' + mix_name,
                                     np.float32(mix),
                                     sr=22050)
            for i in range(3):
                stem_name = 'AuSep_' + str(i+1)+'_' + self.outputs[i] +str(num) +'_song.wav',
                librosa.output.write_wav(file_name +'/' + stem_name[0] ,
                                     np.float32(sample_aug[i][:mix_length]),
                                     sr=22050)
            print('finish Nr.'+ str(num) + 'Music')


if __name__ == '__main__':
    data = Data_aug().get_wav()
    data.augmentation()