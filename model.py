"""
Created on 15.10.2019 
@author: Changxu Ding
based on code from Alexander
"""

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, Concatenate, UpSampling1D, Cropping1D
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU, BatchNormalization, Layer, Flatten, Dense, Dropout
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import utils


def sampling(args):
    """
    sampling trick in latent space
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = list(K.int_shape(z_mean))
    dim[0] = batch
    rand_norm = K.random_normal(shape=tuple(dim))
    return z_mean + K.exp(0.5 * z_log_var) * rand_norm

def mse(y_true, y_pred):
    loss = K.sum(K.square(K.batch_flatten(y_true) - K.batch_flatten(y_pred)), axis=-1)
    return K.mean(loss)


def Conv1DTranspose(input, filters, kernel_size, name, padding='same'):
    """
    transposed conv1d function as alternative methode in upsampling
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=1))(input)
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(1, kernel_size),
        strides=(1, 1),
        padding=padding,
        activation=None,
        use_bias=False,
        name=name,
    )(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x


class KLDivergenceLayer(Layer):
    """
    Identity transform layer that adds KL divergence to the final model loss.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        self.vae_beta = kwargs.get('vae_beta', 1.0)
        del kwargs['vae_beta']
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)


    def call(self, inputs):
        assert isinstance(inputs, list)
        mu, log_var = inputs
        kl_batch = -0.5 * (1 + log_var - K.square(mu) - K.exp(log_var))
        # Since we have multiple encoding levels, with very different dimensions, summing all dimensions except batch
        # will lead to very different impact on loss. The difference between the optimal value of VAE_BETA_FONT and
        # VAE_BETA_CHAR would likely be big.
        # Therefore, perhaps we should just take the mean over all dims, instead of summing some.
        while len(kl_batch.shape) >= 2:
            kl_batch = K.mean(kl_batch, axis=-1)  # Sum over all axes, except the batch axis
        self.add_loss(self.vae_beta * K.mean(kl_batch), inputs=inputs)

        return inputs

class Addlabel_layer(Layer):
    """
     add additional label information of instruments into latent feature maps
    """

    def __init__(self, source_num, latent,  **kwargs):
        self.source_num = source_num
        self.latent = latent
        super(Addlabel_layer, self).__init__(**kwargs)

    def call(self, inputs):
        # multiply label matrix with latent feature map
        assert isinstance(inputs, list)
        x, label = inputs
        label = tf.tile(label, [1,self.latent])
        x = tf.expand_dims(x, axis=-1)
        x = tf.multiply(label, x)
        x = tf.reshape(x,(-1, self.latent*self.latent*self.source_num))
        return x


class VAE():

    def __init__(self, config, model_type="shallow_vae"):
        """
        config : .json file contains parameter and hyperparameter
        model_type: "shallower_vae" or "deeper_vae"
        """
        self.config = config
        self.model_type = model_type
        self.input_length = self.config["model"]["input_length"]
        self.latent_dim = self.config["model"]["latent"]
        self.sources_num = self.config["model"]["source_num"]
        self.model = self.config["model"][self.model_type]
        self.dense = self.config["model"]["dense"]
        self.input_labels = False
        self.alpha = self.config["model"]["alpha_relu"]
        self.dropout = self.config["model"]["drop_out"]
        self.lr = self.config["model"]["lr"]
        self.l2 = self.config["model"]["l2"]
        
    def build_models(self, verbose=True):
        x_mix = Input(shape=(self.input_length, 1), name='input')
        if self.input_labels:
            label = Input(shape=(self.sources_num,), name='label')
            
        x = x_mix
        concat = list()
        for i in range(len(self.model)):
            #encoder block
            x = Conv1D(filters=self.model[i]["filter"],
                        kernel_size=self.model[i]["encoder"],
                        strides=1,
                        dilation_rate=1,
                        padding='same',
                        activation=None,
                        use_bias=False,
                        kernel_regularizer=l2(self.l2))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.alpha)(x)
            
            #save feature map after BN for concanating 
            concat.append(x)

            if self.model[i]["pool"] > 1:
                x = MaxPooling1D( pool_size=self.model[i]["pool"], strides=self.model[i]["pool"], padding='same')(x)

         # return shape of tensor before flattened
        shape = K.int_shape(x)
        
        x = Flatten()(x)
        #Dense Layer
        x = Dense(self.config["model"]["dense"])(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = Dropout(self.dropout)(x)

        #mu and log sigma 
        mean = Dense(self.latent_dim, name="mean")(x)
        mean = Dropout(self.dropout)(mean)
        log_var = Dense(self.latent_dim, name="log_var")(x)
        log_var = Dropout(self.dropout)(log_var)

        #weight for kl loss
        kl_params = {'vae_beta': 1.0}
        mean, log_var = KLDivergenceLayer(name='KLD_VAE', **kl_params)([mean, log_var])
        #sampling for latent feature map
        x= Lambda(sampling, name='z')([mean, log_var])
        #add label
        if self.input_labels:
            x = Addlabel_layer(self.sources_num, self.latent_dim, name='add_label')([x,label])

        x = Dense(shape[1] * shape[2])(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout)(x)

        x = Reshape((shape[1], shape[2]))(x)

        for i in reversed(range(len(self.model))):
            #linear upsampling
            if self.model[i]["pool"] > 1:
                x = UpSampling1D(size=self.model[i]["pool"])(x)
                x = AveragePooling1D(pool_size=self.model[i]["pool"], strides=1, padding='same')(x)

            #skip connection of encoder and decoder before convolution
            x = Concatenate()([x, concat[i]])

            x = Conv1D( filters=self.model[i]["filter"],
                        kernel_size=self.model[i]["decoder"],
                        strides=1,
                        dilation_rate=1,
                        padding='same',
                        activation=None,
                        use_bias=False,
                        kernel_regularizer=l2(self.l2))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.alpha)(x)

        #connect last layer with original sample
        x = Concatenate()([x, x_mix])

        # output with desired source num
        # shape(batch, input_length, source_num)
        stems = list()
        for stem in range(self.sources_num):
            x_stem = Conv1D(filters=1,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            activation='tanh' )(x)
            stems.append(x_stem)
        if self.sources_num > 1:
            x_sources = Concatenate(name='output')(stems)
        elif self.sources_num ==1:
            x_sources = stems[0]
            
        if self.input_labels:
            vae = Model([x_mix,label], x_sources, name='vae')
        else:
            vae = Model(x_mix, x_sources, name='vae')
        vae.compile(
            optimizer=Adam(lr=self.lr, amsgrad=True, epsilon=1e-8),
            loss=mse)
        if verbose:
            vae.summary()
        return vae

if __name__ == '__main__':
    config = utils.load_config("config.json")
    vea=VAE(config).build_models()
