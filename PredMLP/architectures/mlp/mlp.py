"""

Library of a collection of MLP models.

"""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def make_mlp_model(n_input, n_output):
    model = Sequential()
    #model.add(Flatten(input_shape=X_train.shape))
    model.add(Dense(units=100, activation="relu", input_dim=n_input))    #units of hidden layer is 100
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=n_output))    #output layer, dim=n_output
    
    return model
