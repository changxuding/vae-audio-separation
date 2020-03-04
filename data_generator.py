"""
Created on 30.09.2019 
@author: Changxu Ding
"""

import librosa
import os
import numpy as np
import utils

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
train_source_map = ['vn', 'tpt', 'fl']

def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
    return y

def read_wav(instr, mix_path, stem_path):
    sequences = {}
    sequences['mix'] = load(mix_path)
    sequences[instr] = load(stem_path)
    return sequences

def read_npy_file(item):
        data = np.load(item.decode())
        return data.astype(np.float32)

class Dataset_URMP():
    # prepare URMP dataset
    def __init__(self, config):
        self.tracklist = {'train': {'mix': [],'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []},
                          'val': {'mix': [],'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []}}
        self.sequences = {'train':{'mix': [],'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []},
                          'val':{'mix': [],'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []}}

        self.label = { 'train': {'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []},
                       'val' : {'bn': [], 'cl': [],'db': [],'fl': [],'hn': [],'ob': [],'sax': [],
                                    'tba': [],'tbn': [],'tpt': [],'va': [],'vc': [],'vn': []}
        }
        self.config = config
        self.silence_path = self.config["dataset"]["silence_path"]
        self.train_path = self.config["dataset"]["train_path"]
        self.val_path = self.config["dataset"]["val_path"]
        self.batch_size =self.config["dataset"]["batch_size"]
        self.input_length = self.config["dataset"]["input_length"]
        self.hamming = False
        self.labels = True

    def get_wav(self,set):

        if set == 'train':
            database_path = self.train_path
        else:
            database_path = self.val_path
        # Iterate through each track
        for folder in os.listdir(database_path):
            track_sources = [0 for i in range(14)]  # 1st index must be mix source + 13 individual sources

            # Create Sample object for each instrument source files present
            for filename in os.listdir(os.path.join(database_path, folder)):
                if filename.endswith(".wav"):
                    if filename.startswith("AuMix"):
                        mix_path = os.path.join(database_path, folder, filename)

                        track_sources[0] = mix_path
                    else:
                        source_name = filename.split('_')[2]
                        source_idx = source_map[source_name]
                        source_path = os.path.join(database_path, folder, filename)

                        track_sources[source_idx] = source_path

            for i, track in enumerate(track_sources):
                if track == 0:
                    track_sources[i] = self.silence_path

            for instr in list(source_map.keys()):
                idex = source_map[instr]
                self.tracklist[set][instr].append(track_sources[idex])
        return self

    def load_songs(self,set):
        mix_length = []
        for track in self.tracklist[set]['mix']:
            mix_path = track
            mix_sequence = load(mix_path)
            self.sequences[set]['mix'].append(mix_sequence)
            mix_length.append(len(mix_sequence))

        for instr in list(source_map.keys())[1:]:
            for i, track in enumerate(self.tracklist[set][instr]):
                stem_path = track
                stem_sequence= load(stem_path)
                if len(stem_sequence) != mix_length[i]:
                    stem_sequence = stem_sequence[:mix_length[i],:]
                self.sequences[set][instr].append(stem_sequence)

    def get_label(self, set):
        for instr in list(source_map.keys())[1:]:
            for path in self.tracklist[set][instr]:
                if path == self.silence_path:
                    label = 0
                else:
                    label = 1
                self.label[set][instr].append(label)

    def batch_generator(self, set):
        if set not in ['train', 'val']:
            raise ValueError('Argument set must be train or val')

        while True:
            #random choose samples
            sample_indices = np.random.randint(0, len(self.sequences[set]['mix']), self.batch_size)

            batch_inputs = []
            batch_outputs = []
            batch_labels = []
            for i, sample_index in enumerate(sample_indices):

                output = []
                label = []
                mix_sequence = self.sequences[set]['mix'][sample_index]
                #random choose start point
                off_set = np.squeeze(np.random.randint(0,len(mix_sequence)-self.input_length+1,1))
                mix_sample = mix_sequence[off_set:off_set+self.input_length,:]
                for instr in train_source_map:
                    stem_sequence = self.sequences[set][instr][sample_index]
                    stem_sample = stem_sequence[off_set:off_set+self.input_length,:]
                    #if use hanning window
                    if self.hamming:
                        stem_sample = np.hamming(M=self.input_length) * np.squeeze(stem_sample, axis=-1)
                        output.append(np.expand_dims(stem_sample, axis=1))
                    else:
                        output.append(stem_sample)
                    label.append(self.label[set][instr][sample_index])

                if self.hamming:
                    mix_hamming = np.hamming(M=self.input_length) * np.squeeze(mix_sample, axis=-1)
                    batch_inputs.append(np.expand_dims(mix_hamming, axis=1))
                else:
                    batch_inputs.append(mix_sample)
                batch_labels.append(label)
                output = np.concatenate((output),axis=-1)
                batch_outputs.append(output)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs = np.array(batch_outputs, dtype='float32')
            batch_labels = np.array(batch_labels, dtype='int32')
            #for VAE with additional label information in latent dimension
            if self.labels:
                batch = {'input':batch_inputs, 'label': batch_labels} , {'output':batch_outputs}
            else:
                batch = {'input':batch_inputs} , {'output':batch_outputs}

            yield batch

class Dataset_from_Alex():
    #prepare data from Alexander
    #some part function from code of Alex
    def __init__(self):
        self.tracklist = {'train': {'mix': [], 'fl': [],'tpt': [], 'vn': []},
                          'val': {'mix': [], 'fl': [],'tpt': [], 'vn': []}}
        self.sequences = {'train': {'mix': [], 'fl': [],'tpt': [], 'vn': []},
                          'val': {'mix': [], 'fl': [],'tpt': [], 'vn': []}}

        self.train_path = 'H:/frames_data/train'
        self.val_path = 'H:/frames_data/val'
        self.raw_path = 'H:/frames_data/'
        self.batch_size = 16
        self.input_length = 16384
        self.augumentation = False

    def norm_music(self, set):
        # this part comes from Alex
        # for normierung
        train_files = [*["0" + str(i) for i in np.arange(1, 9, 1)]]
        train_files = [file + "_train_1" for file in train_files]
        test_files = ['01_test_0', '01_test_1']
        if set == 'train':
            files = train_files
        elif set == 'val':
            files = test_files
        for file in files:
            print(file + ":")
            data_dict = {}
            # load all instruments
            for instr in train_alex_source_map:
                print("Lade " + instr + "...")
                data_dict[instr], sr = librosa.load(self.raw_path + file + "_" + instr + ".wav", sr=22050)
            # mix audiodata
            data_dict, data_mix = utils.mix_tracks(data_dict)
            data_dict["mix"] = data_mix
            # Abspeichern
            for instr in data_dict:
                data_dict[instr] = np.float32(data_dict[instr])
                if set == 'train':
                    out_path = self.train_path
                elif set == 'val':
                    out_path = self.val_path
                librosa.output.write_wav(out_path + file + "_" + instr + "_norm.wav", data_dict[instr], sr)
            del data_dict

    def get_wav(self, set):
        #get music path
        if set == 'train':
            database_path = self.train_path
        else:
            database_path = self.val_path
        # Iterate through each track
        for folder in os.listdir(database_path):
            track_sources = [0 for i in range(14)]

            # Create Sample object for each instrument source files present
            for filename in os.listdir(os.path.join(database_path, folder)):
                if filename.endswith(".wav"):
                    if filename.startswith("AuMix"):
                        mix_path = os.path.join(database_path, folder, filename)

                        track_sources[0] = mix_path
                    elif filename.startswith("AuSep"):
                            source_name = filename.split('_')[2]
                            source_idx = source_map[source_name]
                            source_path = os.path.join(database_path, folder, filename)
                            track_sources[source_idx] = source_path

            self.tracklist[set]['mix'].append(track_sources[0])
            for track in ['fl','tpt','vn']:
                idex = source_map[track]
                if track_sources[idex] == 0:
                   track_sources[idex] = self.silence_path
                self.tracklist[set][track].append(track_sources[idex])
        return self

    def load_songs(self, set):
        mix_length = []
        for track in self.tracklist[set]['mix']:
            mix_path = track
            mix_sequence = load(mix_path)
            self.sequences[set]['mix'].append(mix_sequence)
            mix_length.append(len(mix_sequence))

        for instr in list(self.tracklist[set].keys())[1:]:
            for i, track in enumerate(self.tracklist[set][instr]):
                stem_path = track
                stem_sequence= load(stem_path)
                if len(stem_sequence) != mix_length[i]:
                    stem_sequence = stem_sequence[:mix_length[i],:]
                self.sequences[set][instr].append(stem_sequence)

    def batch_generator(self, set):
        #set: 'trian' or 'val'
        #instr: list contains instruments to be trained('vn','tpt','fl')

        if set not in ['train', 'val']:
            raise ValueError('Argument set must be train or val')

        while True:
            sample_indices = np.random.randint(0, len(self.sequences[set]['mix']), self.batch_size)

            batch_inputs = []
            batch_outputs_1 = []
            batch_outputs_2 = []
            batch_outputs_3 = []

            for i, sample_index in enumerate(sample_indices):

                mix_sequence = self.sequences[set]['mix'][sample_index]
                # for instr in instrs:
                vn_sequence = self.sequences[set]['vn'][sample_index]
                tpt_sequence = self.sequences[set]['tpt'][sample_index]
                fl_sequence = self.sequences[set]['fl'][sample_index]

                off_set = np.squeeze(np.random.randint(0,len(mix_sequence)-self.input_length+1,1))
                mix_sample = mix_sequence[off_set:off_set+self.input_length]
                vn_sample = vn_sequence[off_set:off_set+self.input_length]
                tpt_sample = tpt_sequence[off_set:off_set+self.input_length]
                fl_sample = fl_sequence[off_set:off_set+self.input_length]

                batch_inputs.append(mix_sample)
                batch_outputs_1.append(vn_sample)
                batch_outputs_2.append(tpt_sample)
                batch_outputs_3.append(fl_sample)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs_1 = np.array(batch_outputs_1, dtype='float32')
            batch_outputs_2 = np.array(batch_outputs_2, dtype='float32')
            batch_outputs_3 = np.array(batch_outputs_3, dtype='float32')

            batch = {'input':batch_inputs},{'output_1': batch_outputs_1,
                                            'output_2': batch_outputs_2,
                                            'output_3': batch_outputs_3}

            yield batch

if __name__ == '__main__':
    config = utils.load_config("config.json")
    dataset = Dataset_URMP(config)
    dataset.get_wav('train')
    dataset.get_label("train")
    train_generator = dataset.batch_generator('train')
    for i in range(1):
        b=next(train_generator)
        print(b)

