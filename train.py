"""
Created on 09.11.2019
@author: Changxu Ding
"""

import tensorflow as tf
import os
import tensorflow.keras.callbacks as tfc
import matplotlib.pyplot as plt

import data_generator
import model
import utils


class SaveWeights(tfc.Callback):
    """
    Dieser Callback speichert die Gewichte nach jeder geüwnschten Anzahl an Epochen ab.
    """

    def __init__(self, path, string, epochs = 5):
        """
        :param path: Speicherort der Gewichte
        :param string: Name der Gewichte
        :param epochs: Nach wie vielen Epochen wird abgespeichert
        """
        self.path = path
        self.string = string
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.epochs == 0:
            self.model.save_weights(self.path + self.string + "_epoch_" + str(epoch) + ".h5")


class LossHistory(tfc.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()

def train(config):
    #prepare dataset
    dataset = data_generator.Dataset_URMP()
    dataset.get_wav('train')
    # dataset.get_label('train')
    dataset.load_songs('train')
    train_generator = dataset.batch_generator('train')
    dataset.get_wav('val')
    # dataset.get_label('train')
    dataset.load_songs('val')
    val_generator = dataset.batch_generator('val')

    vae = model.VAE().build_models()
    #path for saving model weights
    ckpt_path = config["train"]["path"]

    #callback_1 to save best model
    file_name1 = 'vn_tpt__fl_ckp_{epoch}.h5'
    file_path1 = os.path.join(ckpt_path, file_name1)
    callbacks_1 = tfc.ModelCheckpoint(
            filepath=file_path1,
            save_weight_only=True)

    # callback_2 to visualize loss during training
    file_path2 = os.path.join(ckpt_path, 'log')
    callbacks_2 = tfc.TensorBoard(
            log_dir=file_path2,
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq='epoch' # How often to write logs (default: once per epoch)
     )

    # callbacks_3 to save weights after every epoch
    callbacks_3 = SaveWeights(ckpt_path, '4_instr', epochs=5)

    #callbacks 4
    callbacks_4=tfc.EarlyStopping(patience=["train"]["early_stopping_epoch"], verbose=1,
                                              monitor='loss')
    #callbacks_5 to plot loss after training
    history = LossHistory()


    GPU_Memory = False
    #allocate a subset of the available memory
    if GPU_Memory:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)


    print('training starts......')
    vae.fit_generator(
        train_generator,
        steps_per_epoch=["train"]["num_steps_train"],
        epochs=["train"]["num_epochs"],
        verbose=["train"]["verbosity"],
        callbacks=[callbacks_3, callbacks_4, history],
        validation_data=val_generator,
        validation_steps=["train"]["num_steps_val"]

    )

    history.loss_plot('epoch')

if __name__ == '__main__':
    config = utils.load_config("config.json")
    train(config)
