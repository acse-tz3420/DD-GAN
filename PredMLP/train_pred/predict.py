"""

Predicting process of MLP.

"""

import numpy as np

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def predict_next_time_level(mlp, scaler, x_test, x_inv, nstep, npredict):
    y_predall = []
    for i in range(npredict):
        yhat = mlp.predict(x_test)
        yhat_inv = np.concatenate((x_test, yhat), axis=1)
        yhat_inv = scaler.inverse_transform(yhat_inv)
        yhat_inv = yhat_inv[:, 5:]
        y_predall.append(yhat_inv[-1])

        y_next = np.concatenate((yhat_inv, x_inv), axis=1)
        y_next = scaler.transform(y_next)
        y_next = y_next[:, :5]
        x_test = np.concatenate((x_test, [y_next[-1]]), axis=0)[-nstep:, :]
        x_inv = np.concatenate((x_inv, [yhat_inv[-1]]), axis=0)[-nstep:, :]

    return np.array(y_predall)


def predict_next_time_level_deriv(mlp, scaler, step, x_test, x_inv, 
                                  nstep, npredict):
    y_predall = []
    for i in range(npredict):
        yhat = mlp.predict(x_test)
        yhat_inv = np.concatenate((x_test, yhat), axis=1)
        yhat_inv = scaler.inverse_transform(yhat_inv)
        yhat_inv = yhat_inv[:, 5:]
        y_pred = step * yhat_inv + x_inv
        y_predall.append(y_pred[-1])

        y_next = np.concatenate((y_pred, x_inv), axis=1)
        y_next = scaler.transform(y_next)
        y_next = y_next[:, :5]
        x_test = np.concatenate((x_test, [y_next[-1]]), axis=0)
        x_inv = np.concatenate((x_inv, [y_pred[-1]]), axis=0)
        x_test = np.concatenate((x_test, [y_next[-1]]), axis=0)[-nstep:, :]
        x_inv = np.concatenate((x_inv, [y_pred[-1]]), axis=0)[-nstep:, :]

    return np.array(y_predall)
