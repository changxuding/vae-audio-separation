"""
Created on 22.12.2019
@author: Changxu Ding
"""
import librosa
import mir_eval

import numpy as np
import sys
# sys.path.append('H:/checkpoints/mode14_aug/')
import model

import os
import pandas as pd

source_map = {
    'mix': 0,
    'vn': 1,
    'tpt': 2
}
def load(path, sr=22050, mono=True, offset=0.0, duration=None, output_2d=True,  dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio if output2d
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if output_2d:
        return np.expand_dims(y, axis=1)
    else:
        return y

def save_wav(prediction, source_path, sample_rate=22050 ):

    librosa.output.write_wav(source_path,
                             np.float32(prediction),
                             sr=sample_rate)

def predict(mix_path, model_weight_path, save_path):
    # mix_path : path of mix music for predict
    # model_weight_path: weight of model ready for loading
    # save_path: list of paths for saving

    sep_input_shape = (1,16384,1)
    print("start predicting......")
    mix_sequence = load(mix_path)
    assert (len(mix_sequence.shape) == 2)

    # Preallocate source predictions (same shape as input mixture)
    sample_length = mix_sequence.shape[0]
    #the num of zero matrix depends on num of output source
    vn_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument
    tpt_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument
    fl_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument

    input_length = sep_input_shape[1]
    #load model
    vae = model.VAE().build_models()
    vae.load_weights(model_weight_path)
    # Iterate through total length
    for source_pos in range(0, sample_length, input_length):
        # If last segment small than input_length, then take very end segment instead
        if source_pos + input_length > sample_length:
            source_pos = sample_length - input_length

        mix_part = mix_sequence[source_pos:source_pos + input_length,:]
        #let the shape of input same as shape of training process, set batch to 1
        mix_part = np.expand_dims(mix_part, axis=0)

        predict_source = np.squeeze(vae.predict(mix_part,batch_size=1), axis=0)


        # Save predictions for concate
        vn_pre[source_pos:source_pos + input_length, 0] = predict_source[:, 0]
        tpt_pre[source_pos:source_pos + input_length, 0] = predict_source[:, 1]
        fl_pre[source_pos:source_pos + input_length, 0] = predict_source[:, 2]


    save_wav(vn_pre,  save_path[0])
    # print("finish fl predict......")
    save_wav(tpt_pre,  save_path[1])
    save_wav(fl_pre,  save_path[2])

def predict_hanning(mix_path, model_weight_path, save_path):
    # mix_path : path of mix music for predict
    # model_weight_path: weight of model ready for loading
    # save_path: list of paths for saving

    sep_input_shape = (1,16384,1)
    print("start predicting......")
    mix_sequence = load(mix_path)
    assert (len(mix_sequence.shape) == 2)

    # Preallocate source predictions (same shape as input mixture)
    sample_length = mix_sequence.shape[0]
    #the num of zero matrix depends on num of output source
    vn_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument
    tpt_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument
    fl_pre = np.zeros(mix_sequence.shape, np.float32) #prediction for one instrument

    input_length = sep_input_shape[1]
    #load model
    vae = model.VAE().build_models()
    vae.load_weights(model_weight_path)
    # Iterate through total length
    for source_pos in range(0, sample_length, int(input_length/2)):
        # If last segment small than input_length, then take very end segment instead
        if source_pos + input_length > sample_length:
            source_pos = sample_length - input_length

        mix_part = mix_sequence[source_pos:source_pos + input_length,:]
        mix_part = np.expand_dims((np.hamming(M=input_length) * np.squeeze(mix_part, axis=-1)),axis=-1)
        #let the shape of input same as shape of training process, set batch to 1
        mix_part = np.expand_dims(mix_part, axis=0)

        predict_source = np.squeeze(vae.predict(mix_part,batch_size=1), axis=0)

        # Save predictions for concate
        vn_pre[source_pos:source_pos + input_length, 0] += predict_source[:, 0]
        tpt_pre[source_pos:source_pos + input_length, 0] +=  predict_source[:, 0]
        fl_pre[source_pos:source_pos + input_length, 0] +=  predict_source[:, 0]


    save_wav(vn_pre,  save_path[0])
    # print("finish fl predict......")
    save_wav(tpt_pre,  save_path[1])
    save_wav(fl_pre,  save_path[2])
def evaluation(true_path, predict_path):
    # true_path & predict_path :
    # path list of grundtrue and prediction
    horn_true = load(true_path[0], output_2d=False)
    tpt_true = load(true_path[1], output_2d=False)
    fl_true = load(true_path[2], output_2d=False)

    horn_predict = load(predict_path[0], output_2d=False)
    tpt_predict = load(predict_path[1], output_2d=False)
    fl_predict = load(predict_path[2], output_2d=False)
    sdr, sir, sar, per = mir_eval.separation.bss_eval_sources(np.array([horn_true, tpt_true,fl_true]),
                                                              np.array([horn_predict,tpt_predict,fl_predict]),
                                                               compute_permutation=True)
    return sdr, sir, sar

def get_path_alex(path_dict, folder, train=True):
    for num in os.listdir(folder):
        path = os.path.join(folder,num)
        for next_folder in os.listdir(path):
            if train:
                instr = next_folder.split('_')[3]
            else:
                instr = next_folder.split('_')[0]
            if instr != 'mix':
                path_dict[num].append(os.path.join(path, next_folder))
    return path_dict

def metrics_caculate(true_path, predict_path, metrics_path):
    df_sdr = pd.DataFrame(columns=['vn', 'tpt','fl'])
    df_sir = pd.DataFrame(columns=['vn', 'tpt','fl'])
    df_sar = pd.DataFrame(columns=['vn', 'tpt','fl'])

    sdr,sir,sar = evaluation(true_path,predict_path)
    true_path = get_path_alex(true_path,'H:/test')
    predict_path = get_path_alex(predict_path,'H:/pre_test', train=False)

    for i, num in enumerate(true_path):

        vn_true = load(true_path[0], output_2d=False)
        tpt_true = load(true_path[1], output_2d=False)
        fl_true = load(true_path[2], output_2d=False)

        vn_predict = load(predict_path[0], output_2d=False)
        tpt_predict = load(predict_path[1], output_2d=False)
        fl_predict = load(predict_path[2], output_2d=False)

        for i, source_pos in enumerate(range(0, len(vn_true), 22050 * 6)):
            if source_pos + 22050 * 6 > len(vn_true):
                source_pos = 22050 * 6 - len(vn_true)
            vn_true_sample = vn_true[source_pos:source_pos + 22050 * 6]
            tpt_true_sample = tpt_true[source_pos:source_pos + 22050 * 6]
            fl_true_sample = fl_true[source_pos:source_pos + 22050 * 6]
            vn_pre_sample = vn_predict[source_pos:source_pos + 22050 * 6]
            tpt_pre_sample = tpt_predict[source_pos:source_pos + 22050 * 6]
            fl_pre_sample = fl_predict[source_pos:source_pos + 22050 * 6]
            sdr, sir, sar, per = mir_eval.separation.bss_eval_sources(np.array([vn_true_sample, tpt_true_sample,fl_true_sample]),
                                                                      np.array(
                                                                          [vn_pre_sample, tpt_pre_sample,fl_pre_sample]),
                                                                      compute_permutation=True)

            df_sdr.loc[i + 1] = sdr
            df_sir.loc[i + 1] = sir
            df_sar.loc[i + 1] = sar

    df_sdr.to_csv(metrics_path[0], sep=";")
    df_sir.to_csv(metrics_path[1], sep=";")
    df_sar.to_csv(metrics_path[2], sep=";")


if __name__ == '__main__':
    test_path = "H:/test/18_Nocturne_vn_fl_tpt/AuMix_18_Nocturne_vn_fl_tpt.wav"
    predict_path = ['H:/prediction/deep_3_pre/vn.wav',
                    'H:/prediction/deep_3_pre/tpt.wav',
                    'H:/prediction/deep_3_pre/fl.wav']
    predict(test_path, "H:/aug/3_instr_4096_epoch_145.h5", predict_path)

    metrics_path = ["sdr_deep3.csv","sir_deep3.csv","sar_deep3.csv"]

    true_path = ['H:/test/18_Nocturne_vn_fl_tpt/AuSep_1_vn_18_Nocturne.wav',
                'H:/test/18_Nocturne_vn_fl_tpt/AuSep_3_tpt_18_Nocturne.wav',
                 'H:/test/18_Nocturne_vn_fl_tpt/AuSep_2_fl_18_Nocturne.wav'
                 ]

    metrics_caculate(true_path,predict_path)


