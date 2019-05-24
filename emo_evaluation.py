import numpy as np
from fi_similar_eq_class.config import Config
from collections import defaultdict
from collections import Counter
from ordered_set import OrderedSet
import fi_similar_eq_class.data_structure as ds
import random
import time


def evaluate_with_dataflow(individual, x, y, df_equal, static_dynamic_dict):
    start = time.time()
    # init all definite equvalence set
    similarity_division = set()
    de_dict = {}

    npmask = np.array(individual, dtype=np.uint64)

    masked_x = x & npmask

    # individual[-1] = np.uint64(
    #     0xFFFFFFFFFFFFFFFF)  # the last element of individual is mask of 'length', which needs to remain out
    # individual[-2] = np.uint64(
    #     0xFFFFFFFFFFFFFFFF)  # the second last element of individual is mask of 'reg'

    # projection similarity division
    projection_dict = defaultdict(list)
    for i in range(masked_x.shape[0]):
        projection_tuple = tuple(masked_x[i].tolist())
        dyn = x[i][0]
        ip_x = x[i][1]
        reg = x[i][2]
        bit = x[i][-2]
        length = x[i][-1]
        result = y[i][0]
        de = ds.DefiniteEquvalanceSet(dyn, ip_x, reg, bit, length, result)
        de_dict[de.id] = de
        projection_dict[projection_tuple].append(de)

    for key in projection_dict:
        ss = ds.SimilaritySet()
        ss.id = key
        for de in projection_dict[key]:
            ss.add(de)
        similarity_division.add(ss)

    # dataflow similarity division
    for index, dataflow_pair in df_equal.iterrows():
        p = dataflow_pair['is_equal']
        if random.random() > 0.5:
            static1 = dataflow_pair['static1']
            register1 = dataflow_pair['register1']

            dyn2 = dataflow_pair['dyn2']
            static2 = dataflow_pair['static2']
            register2 = dataflow_pair['register2']

            dyn1_list = static_dynamic_dict[static1]
            dyn2_list = static_dynamic_dict[static2]

            # assert (len(dyn1_list) == len(dyn2_list))
            if len(dyn1_list) != len(dyn2_list):
                print('=============================')
                print('dyn1_list:', dyn1_list)
                print('dyn2_list:', dyn2_list)
                continue

            for i in range(len(dyn2_list)):
                dyn1 = dyn1_list[i]
                dyn2 = dyn2_list[i]
                id1 = str(dyn1) + ',' + str(static1) + ',' + str(register1)

                id2 = str(dyn2) + ',' + str(static2) + ',' + str(register2)

                if id1 in de_dict and id2 in de_dict:
                    ss1 = de_dict[id1].belongs_to
                    ss2 = de_dict[id2].belongs_to

                    if ss1.id != ss2.id:
                        for de in ss2.de_set:
                            ss1.add(de)
                        similarity_division.remove(ss2)

    # calculate accuracy, class_nums
    right_predicted = 0
    total_count = 0
    for ss in similarity_division:
        count_list = [0, 0, 0, 0, 0, 0]
        for de in ss.de_set:
            count_list[de.fi_result] = count_list[de.fi_result] + de.length
        right_predicted = right_predicted + max(count_list)
        total_count += sum(count_list)
    eq_accuracy = right_predicted / total_count
    num_eq_class = len(similarity_division)

    print('evaluation time: ', time.time() - start)

    return num_eq_class, eq_accuracy


def projvec_evaluate_with_def_use_length_only_merge_same_register(individual, x, y):
    id_label_dict = defaultdict(list)
    # individual[-1] = np.uint64(
    #     0xFFFFFFFFFFFFFFFF)  # the last element of individual is mask of 'length', which needs to remain out
    individual[-2] = np.uint64(
        0xFFFFFFFFFFFFFFFF)  # the second last element of individual is mask of 'reg'

    npmask = np.array(individual, dtype=np.uint64)

    masked_x = x & npmask

    for i in range(masked_x.shape[0]):
        # id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
        count_list = [0, 0, 0, 0, 0, 0]
        list_tuple = tuple(masked_x[i].tolist())
        if list_tuple in id_label_dict:
            count_list = id_label_dict[list_tuple]

        count_list[y[i][0]] = count_list[y[i][0]] + x[i][-1]  # in this test version, x[-1] stores length

        id_label_dict[list_tuple] = count_list
    # calculate accuracy
    right_predicted = 0
    total_count = 0
    for k in id_label_dict:
        count_list = id_label_dict[k]
        right_predicted = right_predicted + max(count_list)
        total_count += sum(count_list)
    eq_accuracy = right_predicted / total_count
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_with_def_use_length(individual, x, y):
    id_label_dict = defaultdict(list)
    # individual[-1] = np.uint64(
    #     0xFFFFFFFFFFFFFFFF)  # the last element of individual is mask of length, which needs to remain out
    npmask = np.array(individual, dtype=np.uint64)
    masked_x = x & npmask

    for i in range(masked_x.shape[0]):
        # id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
        count_list = [0, 0, 0, 0, 0, 0]
        list_tuple = tuple(masked_x[i].tolist())
        if list_tuple in id_label_dict:
            count_list = id_label_dict[list_tuple]

        count_list[y[i][0]] = count_list[y[i][0]] + x[i][-1]  # in this test version, x[-1] stores length

        id_label_dict[list_tuple] = count_list
    # calculate accuracy
    right_predicted = 0
    total_count = 0
    for k in id_label_dict:
        count_list = id_label_dict[k]
        right_predicted = right_predicted + max(count_list)
        total_count += sum(count_list)
    eq_accuracy = right_predicted / total_count
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_single_proj(individual, x, y):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    masked_x = x & npmask
    for i in range(masked_x.shape[0]):
        # id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
        id_label_dict[tuple(masked_x[i].tolist())].append(y[i][0])
    # calculate accuracy
    right_predicted = 0
    for k in id_label_dict:
        label_list = id_label_dict[k]
        result = Counter(label_list)
        right_predicted = right_predicted + max(list(result.values()))
    eq_accuracy = right_predicted / x.shape[0]
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_single_proj_exclusive(individual, x, y):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    masked_x = x & npmask
    for i in range(masked_x.shape[0]):
        # id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
        id_label_dict[tuple(masked_x[i].tolist())].append(y[i][0])
    # calculate accuracy
    right_predicted = 0
    for k in id_label_dict:
        label_list = id_label_dict[k]
        result = Counter(label_list)
        counts = list(result.values())
        if max(counts) > sum(counts) * Config.exclusive_base:
            right_predicted = right_predicted + max(counts)
    eq_accuracy = right_predicted / x.shape[0]
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_single_proj_mse(individual, x, y):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    masked_x = x & npmask
    for i in range(masked_x.shape[0]):
        # id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
        id_label_dict[tuple(masked_x[i].tolist())].append(y[i][0])
    # calculate accuracy
    right_predicted = 0
    eq_accuracy_list = []
    sum_square_list = []
    for k in id_label_dict:
        label_list = id_label_dict[k]
        result = Counter(label_list)
        counts = list(result.values())
        max_counts = max(counts)
        sum_counts = sum(counts)
        eq_accuracy_list.append((2 * max_counts - sum_counts) * (2 * max_counts - sum_counts))
        sum_square_list.append(sum_counts * sum_counts)
    eq_accuracy = np.sum(eq_accuracy_list) / np.sum(sum_square_list)
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_single_proj_sdc_only(individual, x, y):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    masked_x = x & npmask
    y_1d = np.reshape(y, -1)
    sdc_and_correct_index = (y_1d == 3) | (y_1d == 4)
    masked_x_sdc_and_correct = masked_x[sdc_and_correct_index]
    y_sdc_and_correct = y[sdc_and_correct_index]

    for i in range(masked_x_sdc_and_correct.shape[0]):
        id_label_dict[tuple(masked_x_sdc_and_correct[i].tolist())].append(y_sdc_and_correct[i][0])
    # calculate accuracy
    right_predicted = 0
    for k in id_label_dict:
        label_list = id_label_dict[k]
        result = Counter(label_list)
        right_predicted = right_predicted + max(list(result.values()))
    eq_accuracy = right_predicted / masked_x_sdc_and_correct.shape[0]
    num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_single_proj_force_merge_prop_his(individual, x, y, blockid_index, prop_his_index):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    npmask[blockid_index] = np.uint64(0xFFFFFFFFFFFFFFFF)  # force
    npmask[prop_his_index] = np.uint64(0xFFFFFFFFFFFFFFFF)  # force
    prop_trace_dict = defaultdict(OrderedSet)
    masked_x = x & npmask

    temp = np.delete(masked_x, blockid_index, 1)
    masked_x_delete_blockid_prophis = np.delete(temp, prop_his_index, 1)

    for i in range(masked_x.shape[0]):
        prop_trace_dict[tuple([masked_x[i][blockid_index], masked_x[i][prop_his_index]])].add(
            tuple(masked_x_delete_blockid_prophis[i].tolist()))

    for i in range(masked_x_delete_blockid_prophis.shape[0]):
        id_label_dict[tuple(masked_x_delete_blockid_prophis[i].tolist())].append(y[i][0])

    for k in prop_trace_dict:
        prop_trace = prop_trace_dict[k]
        new_key = tuple(list(k) + list(prop_trace))
        for t in prop_trace:
            id_label_dict[new_key] += id_label_dict[t]
            id_label_dict[t] = []

    # calculate accuracy
    right_predicted = 0
    for k in id_label_dict:
        label_list = id_label_dict[k]
        if len(label_list) > 0:
            result = Counter(label_list)
            right_predicted = right_predicted + max(list(result.values()))
    eq_accuracy = right_predicted / x.shape[0]
    num_eq_class = 0
    for k in id_label_dict:
        if len(id_label_dict[k]) > 0:
            num_eq_class += 1
    # num_eq_class = len(id_label_dict)
    return num_eq_class, eq_accuracy


def projvec_evaluate_multi_proj(pop_dict, index, blockid_index, x, y):
    blockid_unique = np.unique(x[:, blockid_index])

    right_predicted = 0
    num_eq_class = 0

    x_remain = x
    y_remain = y
    for block_id in blockid_unique:
        sub_index = x_remain[:, blockid_index] == block_id
        x_sub = x_remain[sub_index]
        y_sub = y_remain[sub_index]
        if (len(x_sub) < Config.min_sub_block_size) or (block_id not in pop_dict):
            continue

        not_sub_index = (1 - sub_index).astype(np.bool)
        x_remain = x_remain[not_sub_index]
        y_remain = y_remain[not_sub_index]

        individual = pop_dict[block_id][index]

        npmask = np.array(individual, dtype=np.uint64)
        masked_x_sub = x_sub & npmask

        id_label_dict = defaultdict(list)
        for i in range(masked_x_sub.shape[0]):
            id_label_dict[tuple(masked_x_sub[i].tolist())].append(tuple((y_sub[i]).tolist()))

        for k in id_label_dict:
            label_list = list(id_label_dict[k])
            result = Counter(label_list)
            right_predicted = right_predicted + max(list(result.values()))
        num_eq_class += len(id_label_dict)

    # remain
    x_sub = x_remain
    y_sub = y_remain
    individual = pop_dict[-1][index]

    npmask = np.array(individual, dtype=np.uint64)
    masked_x_sub = x_sub & npmask

    id_label_dict = defaultdict(list)
    for i in range(masked_x_sub.shape[0]):
        id_label_dict[tuple(masked_x_sub[i].tolist())].append(tuple((y_sub[i]).tolist()))

    for k in id_label_dict:
        label_list = list(id_label_dict[k])
        result = Counter(label_list)
        right_predicted = right_predicted + max(list(result.values()))
    num_eq_class += len(id_label_dict)

    eq_accuracy = right_predicted / x.shape[0]
    return num_eq_class, eq_accuracy

# def projvec_evaluate_multi_proj(individual, x, y, blockid_index):
#     id_label_dict = defaultdict(list)
#     npmask = np.array(individual, dtype=np.uint64)
#     masked_x = x & npmask
#     for i in range(masked_x.shape[0]):
#         id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y[i]).tolist()))
#     # calculate accuracy
#     right_predicted = 0
#     for k in id_label_dict:
#         label_list = list(id_label_dict[k])
#         result = Counter(label_list)
#         right_predicted = right_predicted + max(list(result.values()))
#     eq_accuracy = right_predicted / x.shape[0]
#     num_eq_class = len(id_label_dict)
#     return num_eq_class, eq_accuracy


# def projvec_evaluate(individual, x_train, y_train):
#     id_label_dict = defaultdict(list)
#     npmask = np.array(individual, dtype=np.uint64)
#     masked_x = x_train & npmask
#     for i in range(masked_x.shape[0]):
#         id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y_train[i]).tolist()))
#     # calculate accuracy
#     right_predicted = 0
#     for k in id_label_dict:
#         label_list = list(id_label_dict[k])
#         result = Counter(label_list)
#         right_predicted = right_predicted + max(list(result.values()))
#     eq_accuracy = right_predicted / x_train.shape[0]
#     num_eq_class = len(id_label_dict)
#     return num_eq_class, eq_accuracy


# def projvec_evaluate_test(individual, x_test, y_test):
#     id_label_dict = defaultdict(list)
#     npmask = np.array(individual, dtype=np.uint64)
#     masked_x = x_test & npmask
#     for i in range(masked_x.shape[0]):
#         id_label_dict[tuple(masked_x[i].tolist())].append(tuple((y_test[i]).tolist()))
#     # calculate accuracy
#     right_predicted = 0
#     for k in id_label_dict:
#         label_list = list(id_label_dict[k])
#         result = Counter(label_list)
#         right_predicted = right_predicted + max(list(result.values()))
#     eq_accuracy = right_predicted / x_test.shape[0]
#     num_eq_class = len(id_label_dict)
#     return num_eq_class, eq_accuracy
