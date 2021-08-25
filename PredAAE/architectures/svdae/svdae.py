"""

Library of a collection of encoders and decoders that can readily be imported
and used by the SVD autoencoder model.

"""

from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPool1D, MaxPool2D,\
                         Flatten, UpSampling2D, UpSampling1D, Reshape,\
                         BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def build_dense_encoder(latent_dim, initializer, info=False,
                        act='relu', dropout=0.6):
    encoder = Sequential()
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


def build_conv_encoder_decoder(input_dim, latent_dim, initializer, info=False,
                               act='relu', dense_act='relu', dropout=0.6, 
                               final_act="linear"):
    """
    These show terrible performance
    """
    encoder = Sequential()
    encoder.add(Conv1D(16, 3, kernel_initializer=initializer,
                       input_shape=input_dim, activation=act, padding="same"))
    encoder.add(MaxPool1D(padding="same"))
    encoder.add(Conv1D(8, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    encoder.add(MaxPool1D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    decoder = Sequential()
    decoder.add(Dense(64, input_dim=latent_dim, activation=dense_act))
    decoder.add(Reshape((encoder.layers[4].input_shape[2], 8)))
    decoder.add(Conv1D(8, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    decoder.add(UpSampling1D())
    decoder.add(Conv1D(16, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    decoder.add(UpSampling1D())
    decoder.add(Conv1D(1, 1, strides=1, activation=final_act))

    return encoder, decoder


def build_wider_conv_encoder_decoder(input_dim, latent_dim, initializer,
                                     info=False, act='relu', dense_act='relu',
                                     dropout=0.6, final_act="linear"):
    """
    These show terrible performance
    """
    encoder = Sequential()
    encoder.add(Conv1D(64, 3, kernel_initializer=initializer,
                       input_shape=(1, input_dim), activation=act,
                       padding="same"))
    encoder.add(MaxPool1D(padding="same"))
    encoder.add(Conv1D(32, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    encoder.add(MaxPool1D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    decoder = Sequential()
    decoder.add(Dense(32*8, input_dim=latent_dim, activation=dense_act))
    decoder.add(Reshape((encoder.layers[4].input_shape[2], 8)))
    decoder.add(Conv1D(32, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    decoder.add(UpSampling1D())
    decoder.add(Conv1D(64, 3, kernel_initializer=initializer, activation=act,
                       padding="same"))
    decoder.add(UpSampling1D())
    decoder.add(Conv1D(1, 1, strides=1, activation="linear"))

    return encoder, decoder


# We make the decoder model
def build_dense_decoder(input_dim, latent_dim, initializer, info=False,
                        act='relu', dropout=0.6, final_act="linear"):
    decoder = Sequential()
    decoder.add(Dense(1000, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder


def build_vinicius_encoder_decoder(input_dim, latent_dim, initializer,
                                   info=False, act='elu', dense_act='elu',
                                   dropout=0.6, reg=1e-3, batchnorm=True,
                                   final_act="linear"):

    encoder = Sequential()
    encoder.add(Dense(8*8*128, activation=act,
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Reshape((8, 8, 128)))
    encoder.add(Conv2D(128, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear",
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    decoder = Sequential()
    decoder.add(Dense(2*2*64, activation=dense_act, input_shape=(latent_dim,),
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Reshape((2,
                         2, 64)))
    decoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Conv2D(128, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Flatten())
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_regularizer=l2(reg),  bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary(), decoder.summary())

    return encoder, decoder


def build_deriv_encoder_decoder(input_dim, latent_dim, initializer,
                                   info=False, act='elu', dense_act='elu',
                                   dropout=0.6, reg=1e-3, batchnorm=True,
                                   final_act="linear"):

    encoder = Sequential()
    encoder.add(Dense(8*8*128, activation=act,  
                      kernel_regularizer=l2(reg), 
                      kernel_initializer=initializer,
                      use_bias=False,))
    encoder.add(Dropout(dropout))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Reshape((8, 8, 128)))
    encoder.add(Conv2D(128, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       kernel_initializer=initializer,
                       use_bias=False))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       kernel_initializer=initializer,
                       use_bias=False))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="sigmoid",
                      kernel_regularizer=l2(reg), 
                      kernel_initializer=initializer,
                      use_bias=False))

    decoder = Sequential()
    decoder.add(Dense(2*2*64, activation=dense_act, input_shape=(latent_dim,),
                      kernel_regularizer=l2(reg), 
                      kernel_initializer=initializer,
                      use_bias=False))
    decoder.add(Dropout(dropout))
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Reshape((2,
                         2, 64)))
    decoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       kernel_initializer=initializer, 
                       use_bias=False))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Conv2D(128, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       kernel_initializer=initializer,
                       use_bias=False))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Flatten())
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_regularizer=l2(reg), 
                      kernel_initializer=initializer,
                      use_bias=False))

    if info:
        print(encoder.summary(), decoder.summary())

    return encoder, decoder


def build_slimmer_vinicius_encoder_decoder(input_dim, latent_dim, initializer,
                                           info=False, act='elu',
                                           dense_act='elu',
                                           dropout=0.6, reg=1e-3,
                                           batchnorm=True, final_act="linear"):

    encoder = Sequential()
    encoder.add(Dense(8*8*64, activation=act,
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Reshape((8, 8, 64)))
    encoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Conv2D(32, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear",
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    decoder = Sequential()
    decoder.add(Dense(2*2*32, activation=dense_act, input_shape=(latent_dim,),
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Reshape((2,
                         2, 32)))
    decoder.add(Conv2D(32, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Flatten())
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_regularizer=l2(reg),  bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary(), decoder.summary())

    return encoder, decoder


def build_smaller_vinicius_encoder_decoder(input_dim, latent_dim, initializer,
                                           info=False, act='elu',
                                           dense_act='elu',
                                           dropout=0.6, reg=1e-3,
                                           batchnorm=True, final_act="linear"):

    encoder = Sequential()
    encoder.add(Dense(4*4*64, activation=act,
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Reshape((4, 4, 64)))
    encoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Conv2D(32, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    if batchnorm:
        encoder.add(BatchNormalization())
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear",
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    decoder = Sequential()
    decoder.add(Dense(1*1*32, activation=dense_act, input_shape=(latent_dim,),
                      kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Reshape((1,
                         1, 32)))
    decoder.add(Conv2D(32, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Conv2D(64, (3, 3), strides=(1, 1), activation=act,
                       padding='same', kernel_regularizer=l2(reg),
                       bias_regularizer=l2(reg),
                       kernel_initializer=initializer,
                bias_initializer=initializer))
    decoder.add(UpSampling2D())
    if batchnorm:
        decoder.add(BatchNormalization())
    decoder.add(Flatten())
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_regularizer=l2(reg),  bias_regularizer=l2(reg),
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary(), decoder.summary())

    return encoder, decoder


# We make the encoder model
def build_wider_dense_encoder(latent_dim, initializer, info=False,
                              act='relu', dropout=0.6):
    encoder = Sequential()
    encoder.add(Dense(1500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(2000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_wider_dense_decoder(input_dim, latent_dim, initializer, info=False,
                              act='relu', dropout=0.6, final_act="linear"):
    decoder = Sequential()
    decoder.add(Dense(1500, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(2000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder


# We make the encoder model
def build_slimmer_dense_encoder(latent_dim, initializer, info=False,
                                act='relu', dropout=0.6):
    encoder = Sequential()
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer, activation="linear"))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_slimmer_dense_decoder(input_dim, latent_dim, initializer, info=False,
                                act='relu', dropout=0.6, final_act="linear"):
    decoder = Sequential()
    decoder.add(Dense(500, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder


# We make the encoder model
def build_deeper_dense_encoder(latent_dim, initializer, info=False,
                               act='relu', dropout=0.6):
    encoder = Sequential()
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dropout(dropout))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_deeper_dense_decoder(input_dim, latent_dim, initializer, info=False,
                               act='relu', dropout=0.6, final_act="linear"):
    decoder = Sequential()
    decoder.add(Dense(1000, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dropout(dropout))
    decoder.add(Dense(input_dim, activation=final_act,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder
