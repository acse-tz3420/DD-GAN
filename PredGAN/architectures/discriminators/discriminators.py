"""

Library of a collection of discriminators that can readily be imported
and used by the GAN models.

"""

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def make_discriminator_model(ntimes=20, ninput=20, d_dropout=0.3):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=[ntimes, ninput, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(d_dropout))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(d_dropout))

    model.add(Flatten())
    model.add(Dense(1))

    return model
