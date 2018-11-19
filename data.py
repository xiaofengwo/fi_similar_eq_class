import pandas as pd
import numpy as np
from configuration import Config
import math
from sklearn.model_selection import train_test_split


# The data format in the csv files.
COLUMS = Config.COLUMS
FEATURES = Config.FEATURES
LABELS = Config.LABELS


def load_data_from_csv(file_name):

    """
    Call the load_data_from_csv(file_name) methods, return the batch_data
    :param file_name:
    :returns:
    x_train, x_test, y_train, y_test
    shape[batch_size, dim]
    """

    df = pd.read_csv(file_name)
    raw_data = df.dropna()
    total_data_num = raw_data.shape[0]
    print("raw data loaded.")

    x_all = raw_data[:][Config.FEATURES]
    y_all = raw_data[:][Config.LABELS]


    x_all = raw_data[:][Config.FEATURES].astype(np.uint64)
    y_all = raw_data[:][Config.LABELS].astype(np.uint64)

    # split raw_data into trainning set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_all.values, y_all.values, test_size=Config.test_size)

    print('Data Loaded:')
    print('x_train shape:' + str(np.shape(x_train)))
    print('x_test shape:' + str(np.shape(x_test)))
    print('y_train shape:' + str(np.shape(y_train)))
    print('y_test shape:' + str(np.shape(y_test)))

    return x_train, x_test, y_train, y_test
