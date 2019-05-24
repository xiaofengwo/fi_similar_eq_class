import pandas as pd
import numpy as np
import os
import shutil
import math

from fi_similar_eq_class.config import Config
from sklearn.model_selection import train_test_split

# The data format in the csv files.
COLUMS = Config.COLUMS
FEATURES = Config.FEATURES
LABELS = Config.LABELS


def merge_machine_states_and_result(result_filename, machine_states_filename, machine_states_with_results_filename):

    df_machine_states = pd.read_csv(machine_states_filename, sep=',')
    df_machine_states = df_machine_states.dropna(axis=1)
    df_results = pd.read_csv(result_filename, sep=' ')
    df_results_columns = df_results.columns.difference(df_machine_states.columns)
    df_machine_states_with_results = pd.merge(df_machine_states, df_results, on='dyn')

    # Calculate length and insert it before result column
    right_column = df_machine_states_with_results['right']
    left_column = df_machine_states_with_results['left']
    length_column = right_column - left_column
    df_machine_states_with_results['length'] = length_column

    output_dir = os.path.dirname(machine_states_with_results_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df_machine_states_with_results.to_csv(machine_states_with_results_filename, index=False)
    return df_machine_states_with_results


def merge_machine_states_and_prop_his_and_result(result_filename, machine_states_filename, prop_his_filename, machine_states_with_prop_his_with_results_filename):
    df_results = pd.read_csv(result_filename, sep=' ')
    df_machine_states = pd.read_csv(machine_states_filename, sep=',')
    df_machine_states = df_machine_states.dropna(axis=1)
    df_prop_his = pd.read_csv(prop_his_filename, sep=',')
    df_machine_states_with_prop_his = pd.merge(df_machine_states, df_prop_his, on='dyn')
    df_machine_states_with_prop_his_with_results = pd.merge(df_machine_states_with_prop_his, df_results, on='dyn')

    output_dir = os.path.dirname(machine_states_with_prop_his_with_results_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df_machine_states_with_prop_his_with_results.to_csv(machine_states_with_prop_his_with_results_filename, index=False)
    return df_machine_states_with_prop_his_with_results


def prepare_data(program):
    # prepare data
    if Config.using_prop_his is True:
        if Config.need_remerge_tables:
            df_results_with_machine_states = merge_machine_states_and_prop_his_and_result(program.results_path,
                                                                                          program.machine_states_path,
                                                                                          program.prop_his_path,
                                                                                          program.machine_states_with_prop_his_with_results_path)
        x_train, x_test, y_train, y_test, blockid_index, prop_his_index = load_data_from_csv(
            program.machine_states_with_prop_his_with_results_path, program.df_raw_data_remove_trivial_path,
            Config.max_size)
    else:
        if Config.need_remerge_tables:
            df_results_with_machine_states = merge_machine_states_and_result(program.results_path,
                                                                             program.machine_states_path,
                                                                             program.results_with_machine_states_path)
        x_train, x_test, y_train, y_test = load_data_from_csv(program.results_with_machine_states_path,
                                                              program.df_raw_data_remove_trivial_path,
                                                              Config.max_size)
    X_train_rows_count, NDIM = x_train.shape
    return x_train, x_test, y_train, y_test, X_train_rows_count, NDIM


def load_data_from_csv(file_name, removed_trivial_file_name, max_size=None):

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

    if Config.using_selected_features:
        df_raw_data = df_raw_data[:][Config.FEATURES]

    # drop columns which have same values in all rows
    cols = list(df_raw_data)
    unique_count = df_raw_data.apply(pd.Series.nunique)
    cols_to_drop = unique_count[unique_count == 1].index
    df_raw_data_remove_trivial = df_raw_data.drop(cols_to_drop, axis=1)

    # df_raw_data_remove_trivial_filename = Config.df_raw_data_remove_trivial_path
    df_raw_data_remove_trivial_filename = removed_trivial_file_name

    output_dir = os.path.dirname(df_raw_data_remove_trivial_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df_raw_data_remove_trivial.to_csv(df_raw_data_remove_trivial_filename, index=False)
    print(df_raw_data_remove_trivial.shape)

    df_sample = df_raw_data_remove_trivial
    # if max_size if given, drop the others
    if (max_size is not None) and (len(df_raw_data_remove_trivial) > max_size):
        df_sample = df_raw_data_remove_trivial.sample(max_size)
    
    x_all = df_sample.drop(Config.LABELS, axis=1).astype(np.uint64)
    y_all = df_sample[:][Config.LABELS].astype(np.uint64)

    if Config.using_prop_his:
        blockid_index = list(x_all.columns).index(Config.BLOCK_ID_LABEL)
        prop_his_index = list(x_all.columns).index(Config.HISTORY_ID_LABEL)

    # split raw_data into trainning set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_all.values, y_all.values, test_size=Config.test_size)

    print('Data Loaded:')
    print('x_train shape:' + str(np.shape(x_train)))
    print('x_test shape:' + str(np.shape(x_test)))
    print('y_train shape:' + str(np.shape(y_train)))
    print('y_test shape:' + str(np.shape(y_test)))
    if Config.using_prop_his:
        return x_train, x_test, y_train, y_test, blockid_index, prop_his_index
    else:
        return x_train, x_test, y_train, y_test

