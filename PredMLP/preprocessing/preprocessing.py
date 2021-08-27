"""

Preprocessing before creating training dataset.

"""

from pandas import DataFrame
from pandas import concat

__author__ = "Tianyi Zhao"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Reconstruct the time series into a supervised learning dataset.
    input:
    data, list or array, sequence of observations
    n_in, int, length of lagging observations (X)
    n_out, int, length of observations (y)
    dropnan, boolean, discard rows with NaN values
    return:
    Pandas DataFrame, reconstructed dataset
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input series (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # predict series (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # concat column name with data
    agg = concat(cols, axis=1)
    agg.columns = names
    # discard rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def set_deriv(df, step):
    """
    Calculate the time derivative by (a_n+1 - a_n)/step
    """
    for i in range(1, 6):
        df["var%d(deriv)" % i] = (1 / step) * (
            df["var%d(t)" % i] - df["var%d(t-1)" % i])
        df = df.drop(columns=["var%d(t)" % i])
    return df
