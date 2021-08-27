"""

Library of a collection of generators that can readily be imported
and used by the GAN model.

"""

from tensorflow.keras.layers import Dense, Conv2DTranspose, Reshape, \
                                    BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def make_generator_model_deriv_1(latent_space=100):
    model = Sequential()

    model.add(Dense(3*3*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((3, 3, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              output_padding=[0, 0], use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              use_bias=False, activation='tanh'))

    return model


def make_generator_model_deriv_2(latent_space=100):
    """
    This one has good performance
    """
    model = Sequential()

    model.add(Dense(4*4*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((4, 4, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False, 
                              activation='tanh'))

    return model


def make_generator_model_deriv_3(latent_space=100):
    """
    This one has best performance
    """
    model = Sequential()

    model.add(Dense(5*5*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((5, 5, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[1, 1], use_bias=False, 
                              activation='tanh'))

    return model


def make_generator_model_deriv_4(latent_space=100):
    model = Sequential()

    model.add(Dense(7*7*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False, 
                              activation='tanh'))

    return model


def make_generator_model_deriv_5(latent_space=100):
    model = Sequential()

    model.add(Dense(8*8*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((8, 8, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[1, 1], use_bias=False, 
                              activation='tanh'))

    return model


def make_generator_model_deriv_6(latent_space=100):
    model = Sequential()

    model.add(Dense(9*9*256, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((9, 9, 256)))

    model.add(Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                              use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', 
                              output_padding=[0, 0], use_bias=False, 
                              activation='tanh'))

    return model
