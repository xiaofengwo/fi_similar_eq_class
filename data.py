import pandas as pd
import numpy as np
from configuration import Config
import math
from sklearn.model_selection import train_test_split


# The data format in the csv files.
COLUMS = Config.COLUMS
FEATURES = Config.FEATURES
LABELS = Config.LABELS


def merge_result_and_machine_states(result_filename, machine_states_filename, machine_states_with_results_filename):
    df_results = pd.read_csv(result_filename, sep=' ')
    df_machine_states = pd.read_csv(machine_states_filename, sep=',')
    df_machine_states_with_results = pd.merge(df_machine_states, df_results, on='dyn')
    df_machine_states_with_results.to_csv(machine_states_with_results_filename)
    return df_machine_states_with_results

def merge_result_and_machine_states_and_prop_his(result_filename, machine_states_filename, prop_his_filename, machine_states_with_prop_his_with_results_filename):
    df_results = pd.read_csv(result_filename, sep=' ')
    df_machine_states = pd.read_csv(machine_states_filename, sep=',')
    df_prop_his = pd.read_csv(prop_his_filename, sep=',')
    df_machine_states_with_prop_his = pd.merge(df_machine_states, df_prop_his, on='dyn')
    df_machine_states_with_prop_his_with_results = pd.merge(df_machine_states_with_prop_his, df_results, on='dyn')
    df_machine_states_with_prop_his_with_results.to_csv(machine_states_with_prop_his_with_results_filename)
    return df_machine_states_with_prop_his_with_results

def load_data_from_csv(file_name, max_size=None):

    """
    Call the load_data_from_csv(file_name) methods, return the batch_data
    :param file_name:
    :returns:
    x_train, x_test, y_train, y_test
    shape[batch_size, dim]
    """

    df = pd.read_csv(file_name)
    df_raw_data = df.dropna(axis=1)
    total_data_num = df_raw_data.shape[0]
    print("raw data loaded.")

    # drop columns which have same values in all rows
    cols = list(df_raw_data)
    unique_count = df_raw_data.apply(pd.Series.nunique)
    cols_to_drop = unique_count[unique_count == 1].index
    df_raw_data_remove_trivial = df_raw_data.drop(cols_to_drop, axis=1)
    df_raw_data_remove_trivial.to_csv('df_raw_data_remove_trivial.csv')
    print(df_raw_data_remove_trivial.shape)

    # x_all = df_raw_data[:][Config.FEATURES]
    x_all = df_raw_data_remove_trivial.drop(Config.LABELS, axis=1).astype(np.uint64)
    y_all = df_raw_data_remove_trivial[:][Config.LABELS].astype(np.uint64)

    # if max_size if given, drop the others
    if max_size is not None:
        x_all = x_all[: max_size]
        y_all = y_all[: max_size]

    # split raw_data into trainning set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_all.values, y_all.values, test_size=Config.test_size)

    print('Data Loaded:')
    print('x_train shape:' + str(np.shape(x_train)))
    print('x_test shape:' + str(np.shape(x_test)))
    print('y_train shape:' + str(np.shape(y_train)))
    print('y_test shape:' + str(np.shape(y_test)))

    return x_train, x_test, y_train, y_test
