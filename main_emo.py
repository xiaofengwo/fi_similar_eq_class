import random
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import numpy as np
import fi_similar_eq_class.data_utils as data

from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd

from fi_similar_eq_class.config import Config
import fi_similar_eq_class.emo_mutate as my_mutate
import fi_similar_eq_class.emo_init as my_init
import fi_similar_eq_class.emo_evaluation as my_evaluation

import dataflow_analysis.Get_block_static_dynamic_hashmap as bsd_map

# program = Config.current_program
#
# print(program.progname)

# Problem definition
# VALID_BITS = 64
# BOUND_LOW, BOUND_UP = 0, 2 ** (VALID_BITS - 10)
MASK_BITS_BOUNDS_LIST = []
# X_train_rows_count, NDIM = x_train.shape

x_train = None
x_test = None
y_train = None
y_test = None
X_train_rows_count = None
NDIM = None

# dataflow similarity division
df_equal = None
static_dynamic_dict = None

blockid_index = None
prop_his_index = None


def my_multibitflip(individual, indpb):
    if Config.mutate_type == "ALLBITS":
        new_ind = my_mutate.mxkmultibitflip_allbits(individual, indpb, MASK_BITS_BOUNDS_LIST)
    elif Config.mutate_type == "8BITS":
        new_ind = my_mutate.mxkmultibitflip_8bits(individual, indpb, MASK_BITS_BOUNDS_LIST)
    elif Config.mutate_type == "BITWISE":
        new_ind = my_mutate.mxkmultibitflip_bitwise(individual, indpb, MASK_BITS_BOUNDS_LIST)
    elif Config.mutate_type == 'RANDOM_NUMBER':
        new_ind = my_mutate.mxkmultibitflip_by_a_random_number(individual, indpb, MASK_BITS_BOUNDS_LIST)
    return new_ind


def my_initfunc():
    if Config.init_type == "RANDOM":
        initresult = my_init.random_init(NDIM, MASK_BITS_BOUNDS_LIST)
    elif Config.init_type == "ALLZERO":
        initresult = my_init.allzeroinit(NDIM)
    elif Config.init_type == "ALLONE":
        initresult = my_init.alloneinit(NDIM)
    return initresult


def get_mask_bounds(x_train):
    row_count, col_count = x_train.shape
    BIT_BOUND_HIGH = 64
    BIT_BOUND_LOW = 1
    appropriate_mask_bit_count_list = []
    for col_index in range(0, col_count):
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
        one = np.uint64(1)
        appropriate_mask_bit_count = 0
        while appropriate_mask_bit_count <= BIT_BOUND_HIGH:
            x_train_temp = x_train[:, col_index] & mask
            x_train_temp_unique = np.unique(x_train_temp)
            if len(x_train_temp_unique) == 1:
                break
            mask = mask << one
            appropriate_mask_bit_count = appropriate_mask_bit_count + 1
        appropriate_mask_bit_count_list.append(appropriate_mask_bit_count)
    return appropriate_mask_bit_count_list


def evaluate_train(individual):  # eval_mse eval_exclusive eval_force_merge_prop_his eval_sdc_only
    if Config.evaluation_method == "eval_force_merge_prop_his":
        result = my_evaluation.projvec_evaluate_single_proj_force_merge_prop_his(individual, x_train, y_train,
                                                                                 blockid_index, prop_his_index)
    elif Config.evaluation_method == "eval_sdc_only":
        result = my_evaluation.projvec_evaluate_single_proj_sdc_only(individual, x_train, y_train)
    elif Config.evaluation_method == "eval_exclusive":
        result = my_evaluation.projvec_evaluate_single_proj_exclusive(individual, x_train, y_train)
    elif Config.evaluation_method == "eval_mse":
        result = my_evaluation.projvec_evaluate_single_proj_mse(individual, x_train, y_train)
    elif Config.evaluation_method == 'def_use_length':
        result = my_evaluation.projvec_evaluate_with_def_use_length(individual, x_train, y_train)
    elif Config.evaluation_method == 'length_same_reg':
        result = my_evaluation.projvec_evaluate_with_def_use_length_only_merge_same_register(individual, x_train,
                                                                                             y_train)
    elif Config.evaluation_method == 'with_dataflow':
        result = my_evaluation.evaluate_with_dataflow(individual, x_train, y_train, df_equal, static_dynamic_dict)
    elif Config.evaluation_method == 'normal':
        result = my_evaluation.projvec_evaluate_single_proj(individual, x_train, y_train)
    else:
        raise (BaseException('No evaluation method is designated'))
    return result


def evaluate_test(individual):
    if Config.evaluation_method == "eval_force_merge_prop_his":
        result = my_evaluation.projvec_evaluate_single_proj_force_merge_prop_his(individual, x_test, y_test,
                                                                                 blockid_index, prop_his_index)
    elif Config.evaluation_method == "eval_sdc_only":
        result = my_evaluation.projvec_evaluate_single_proj_sdc_only(individual, x_test, y_test)
    elif Config.evaluation_method == "eval_exclusive":
        result = my_evaluation.projvec_evaluate_single_proj_exclusive(individual, x_test, y_test)
    elif Config.evaluation_method == "eval_mse":
        result = my_evaluation.projvec_evaluate_single_proj_mse(individual, x_test, y_test)
    elif Config.evaluation_method == 'def_use_length':
        result = my_evaluation.projvec_evaluate_with_def_use_length(individual, x_test, y_test)
    elif Config.evaluation_method == 'length_same_reg':
        result = my_evaluation.projvec_evaluate_with_def_use_length_only_merge_same_register(individual, x_test,
                                                                                             y_test)
    elif Config.evaluation_method == 'with_dataflow':
        result = my_evaluation.evaluate_with_dataflow(individual, x_test, y_test, df_equal, static_dynamic_dict)
    elif Config.evaluation_method == 'normal':
        result = my_evaluation.projvec_evaluate_single_proj(individual, x_test, y_test)
    else:
        raise (BaseException('No evaluation method is designated'))
    return result


creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
# creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
# creator.create("Individual", list, fitness=creator.FitnessMin)
# creator.create("Individual", array.array, typecode=np.int64, fitness=creator.FitnessMin)
creator.create("Individual", np.ndarray, typecode=np.uint64, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("mxk_init", my_initfunc)
# toolbox.register("attr_bool", random.randint, BOUND_LOW, BOUND_UP)
# toolbox.register("attr_int", allzeroinit, VALID_BITS)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.mxk_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_train)
# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mate", tools.cxOnePoint)
# flip each attribute/gene of 0.05
toolbox.register("mutate", my_multibitflip, indpb=Config.indpb)
if Config.selection_algorithm == "SPEA2":
    toolbox.register("select", tools.selSPEA2)
else:
    toolbox.register("select", tools.selNSGA2)


def plot_front(pop, epoch, do_plot_train_parato_front=True, do_plot_test_parato_front=False,
               target_program=Config.current_program):
    for ind in pop:
        print(ind.fitness.values)
        print(','.join(map(bin, ind)))

    if do_plot_train_parato_front:
        # plot train_parato_front
        train_pareto_front = np.array([ind.fitness.values for ind in pop])

        fig, ax = plt.subplots()

        ax.set_title(target_program.progname)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('#Fault-Similarity Classes')
        ax.set_xscale('log')
        if Config.fake_scale:
            ax.scatter(pow(10, train_pareto_front[:, 0] / 200), train_pareto_front[:, 1], marker='o', c='',
                       edgecolors='b',
                       label='train')
        else:
            ax.scatter(train_pareto_front[:, 0], train_pareto_front[:, 1], marker='o', c='',
                       edgecolors='b',
                       label='train')
        ax.legend(loc="lower right")

    if do_plot_test_parato_front:
        # plot test
        test_fitnesses = test(pop)
        test_pareto_front = np.array(test_fitnesses)

        # sorted_test_pareto_front = np.sort(test_pareto_front, axis=0)
        # plt.scatter(sorted_test_pareto_front[:, 0], sorted_test_pareto_front[:, 1], c="r")
        if Config.fake_scale:
            ax.scatter(pow(10, test_pareto_front[:, 0] / 200), test_pareto_front[:, 1], marker='o', c='r', label='test')
        else:
            ax.scatter(test_pareto_front[:, 0], test_pareto_front[:, 1], marker='o', c='r', label='test')
        ax.legend(loc="lower right")

    # plot diff
    # diff_parato_front = test_pareto_front - train_pareto_front

    # plt.scatter(diff_parato_front[:, 0], diff_parato_front[:, 1], c="g")
    if do_plot_test_parato_front:
        # plt.axis([0, Config.max_size * Config.test_size, 0.75, 1.0])
        # plt.axis('tight')
        plt.axis()
    else:
        # plt.axis([0, Config.max_size * Config.test_size, 0.75, 1.0])
        plt.axis()
        # plt.axis('tight')

    config_info_str = str(target_program.progname) + '_'
    # config_info_str += 'max_size=' + str(Config.max_size) + '_'
    # config_info_str += 'run_mode=' + str(Config.run_mode) + '_'
    # config_info_str += 'with_prop_his=' + str(Config.using_prop_his) + '_'
    # config_info_str += 'mutate_type=' + str(Config.mutate_type) + '_'
    # config_info_str += 'init_type=' + str(Config.init_type) + '_'
    # config_info_str += 'model=' + str(Config.model) + '_'
    # config_info_str += 'res_model=' + str(Config.restore_model) + '_'
    # config_info_str += 'test_size=' + str(Config.test_size) + '_'
    # config_info_str += 'min_blk=' + str(Config.min_sub_block_size) + '_'
    # config_info_str += 'evaluation=' + str(Config.evaluation_method) + '_'
    # config_info_str += 'indpb=' + str(Config.indpb) + '_'
    # config_info_str += 'NGEN=' + str(Config.NGEN) + '_'
    # config_info_str += 'MU=' + str(Config.MU) + '_'
    # config_info_str += 'CXPB=' + str(Config.CXPB) + '_'
    config_info_str += 'epoch_size=' + str(Config.epoch_size)

    saveimage(plt, "save/plt/plt_" + str(epoch + 1) + config_info_str + ".png")
    saveimage(plt, 'save/plt/plt_0_' + config_info_str + '.png')
    plt.show()


def plot_front_multi_proj(pop_dict, do_plot_train_parato_front=True, do_plot_test_parato_front=False):
    if do_plot_train_parato_front:
        # plot train_parato_front
        train_pareto_front = np.array(test_multi_proj(pop_dict, blockid_index, x_train, y_train))
        plt.scatter(train_pareto_front[:, 0], train_pareto_front[:, 1], c="b")

    if do_plot_test_parato_front:
        # plot test_parato_front
        test_fitnesses = test_multi_proj(pop_dict, blockid_index, x_test, y_test)
        test_pareto_front = np.array(test_fitnesses)
        plt.scatter(test_pareto_front[:, 0], test_pareto_front[:, 1], c="r")
        sorted_test_pareto_front = np.sort(test_pareto_front, axis=0)
        # print(sorted_test_pareto_front)

    # plot diff
    # diff_parato_front = test_pareto_front - train_pareto_front

    # plt.scatter(diff_parato_front[:, 0], diff_parato_front[:, 1], c="g")
    if do_plot_test_parato_front:
        plt.axis([0, Config.max_size * Config.test_size, 0.75, 1.0])
    else:
        plt.axis('tight')

    config_info_str = str(Config.program_name) + '_'
    config_info_str += 'max_size=' + str(Config.max_size) + '_'
    config_info_str += 'run_mode=' + str(Config.run_mode) + '_'
    config_info_str += 'with_prop_his=' + str(Config.using_prop_his) + '_'
    config_info_str += 'mutate_type=' + str(Config.mutate_type) + '_'
    config_info_str += 'init_type=' + str(Config.init_type) + '_'
    config_info_str += 'model=' + str(Config.model) + '_'
    config_info_str += 'restore_model=' + str(Config.restore_model) + '_'
    config_info_str += 'test_size=' + str(Config.test_size) + '_'
    config_info_str += 'min_sub_block_size=' + str(Config.min_sub_block_size) + '_'
    config_info_str += 'sdc_only=' + str(Config.sdc_only) + '_'
    config_info_str += 'indpb=' + str(Config.indpb) + '_'
    config_info_str += 'NGEN=' + str(Config.NGEN) + '_'
    config_info_str += 'MU=' + str(Config.MU) + '_'
    config_info_str += 'CXPB=' + str(Config.CXPB) + '_'
    config_info_str += 'epoch_size=' + str(Config.epoch_size)

    saveimage(plt, 'save/plt/plt_0_' + config_info_str + '.png')
    plt.show()


def train(seed=None, target_program=Config.current_program):
    random.seed(seed)
    global MASK_BITS_BOUNDS_LIST
    MASK_BITS_BOUNDS_LIST = get_mask_bounds(x_train)

    # set MASK_BITS_BOUNDS_LIST in a fixed range
    for i in range(len(MASK_BITS_BOUNDS_LIST)):
        if MASK_BITS_BOUNDS_LIST[i] > 16:
            MASK_BITS_BOUNDS_LIST[i] = 16

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=Config.MU)

    if Config.restore_model:
        try:
            pop = loadobj(Config.model_save_path)
            print("model loaded")
        except FileNotFoundError as fne:
            print("Model file not found, will train new model...")
        except Exception as err:
            print("Model file load error, may be the model file is not exist, will train new model...")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fitnesses = toolbox.map(toolbox.evaluate,invalid_ind, len(invalid_ind) * [x_train], len(invalid_ind) * [y_train])

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, Config.NGEN):

        if Config.selection_algorithm == "SPEA2":
            offspring = toolbox.select(pop, len(pop))
        else:
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= Config.CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop[:] + offspring, Config.MU)
        # front = numpy.array([ind.fitness.values for ind in pop])
        # print(front)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # print pareto front every 100 round
        if gen % Config.epoch_size == 0:
            epoch = gen / Config.epoch_size
            saveobj(pop, "save/model/model_" + str(epoch + 1) + ".ckpt")
            saveobj(pop, "save/model/model.ckpt")
            plot_front(pop, epoch, do_plot_train_parato_front=True, do_plot_test_parato_front=False,
                       target_program=target_program)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook


def test(pop):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop]
    fitnesses_map = map(evaluate_test, invalid_ind)
    fitnesses_list = list(fitnesses_map)
    return fitnesses_list


def test_multi_proj(pop_dict, blockid_index, x_test, y_test):
    fitnesses_list = []
    for i in range(Config.MU):
        fitnesses_list.append(my_evaluation.projvec_evaluate_multi_proj(pop_dict, i, blockid_index, x_test, y_test))
    return fitnesses_list


def saveimage(plt, filepath):
    path, filename = os.path.split(filepath)
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(filepath)


def saveobj(obj, filepath):
    path, filename = os.path.split(filepath)
    if not os.path.isdir(path):
        os.makedirs(path)
    joblib.dump(obj, filepath)


def loadobj(filepath):
    obj = joblib.load(filepath)
    return obj


def main(seed=None, program=Config.current_program):
    if Config.run_mode == 'TRAIN' or Config.run_mode == 'TRAIN_TEST':
        # train
        global x_train, y_train
        pop_dict = {}
        if Config.model == 'MULTI_PROJ':
            x_train_backup = np.copy(x_train)
            y_train_backup = np.copy(y_train)
            blockid_unique = np.unique(x_train_backup[:, blockid_index])
            # print('==================all x_train=======================')
            # x_train = x_train_backup
            # y_train = y_train_backup
            # pop, logbook = train()
            #
            # pop.sort(key=lambda x: x.fitness.values)
            # pop_dict[-1] = pop

            x_remain = x_train_backup
            y_remain = y_train_backup

            for block_id in blockid_unique:
                sub_train_index = x_remain[:, blockid_index] == block_id

                x_train = x_remain[sub_train_index]
                if len(x_train) < Config.min_sub_block_size:
                    continue
                x_train = x_remain[sub_train_index]
                y_train = y_remain[sub_train_index]
                not_sub_index = (1 - sub_train_index).astype(np.bool)
                x_remain = x_remain[not_sub_index]
                y_remain = y_remain[not_sub_index]

                print('==================x_train with block_id ' + str(block_id) + '=======================')
                print('==================sub_x_train len = ' + str(x_train.shape) + '=======================')
                pop, logbook = train()

                pop.sort(key=lambda x: x.fitness.values)
                pop_dict[block_id] = pop

            print('==================remain x_train=======================')
            x_train = x_remain
            y_train = y_remain
            pop, logbook = train()

            pop.sort(key=lambda x: x.fitness.values)
            pop_dict[-1] = pop

            saveobj(pop_dict, "save/model/multi_model")
            x_train = x_train_backup
            y_train = y_train_backup

        elif Config.model == 'SINGLE_PROJ':
            # train
            pop, logbook = train()
            pop.sort(key=lambda x: x.fitness.values)
            saveobj(pop, "save/model/single_model")

    if Config.run_mode == 'TEST' or Config.run_mode == 'TRAIN_TEST':
        if Config.model == 'MULTI_PROJ':
            pop_dict = loadobj('save/model/multi_model')
            x_train_remain = x_train
            x_test_remain = x_test
            for k in pop_dict:
                if k != -1:
                    sub_train_index = x_train_remain[:, blockid_index] == k
                    sub_test_index = x_test_remain[:, blockid_index] == k
                    sub_x_train = x_train_remain[sub_train_index]
                    sub_x_test = x_test_remain[sub_test_index]

                    not_sub_train_index = (1 - sub_train_index).astype(np.bool)
                    not_sub_test_index = (1 - sub_test_index).astype(np.bool)
                    x_train_remain = x_train_remain[not_sub_train_index]
                    x_test_remain = x_test_remain[not_sub_test_index]

                    print('pop_dict[' + str(k) + '] ===> ' + 'len(sub_x_train) with block_id == k : ' + str(
                        len(sub_x_train)))
                    print('pop_dict[' + str(k) + '] ===> ' + 'len(sub_x_test) with block_id == k : ' + str(
                        len(sub_x_test)))
            print('pop_dict[' + str(-1) + '] ===> ' + str(len(x_train_remain)))
            print('pop_dict[' + str(-1) + '] ===> ' + str(len(x_test_remain)))

            plot_front_multi_proj(pop_dict, do_plot_train_parato_front=True, do_plot_test_parato_front=True)
        elif Config.model == 'SINGLE_PROJ':
            pop = loadobj('save/model/single_model')
            plot_front(pop, 0, do_plot_train_parato_front=True, do_plot_test_parato_front=True, target_program=program)

    return


def main2():
    global x_train, x_test, y_train, y_test, X_train_rows_count, NDIM, df_equal, static_dynamic_dict
    for program in Config.Benchmarks:
        x_train, x_test, y_train, y_test, X_train_rows_count, NDIM = data.prepare_data(program)


        # dataflow similarity division
        df_dataflow_candidate_pair = pd.read_csv(Config.current_program.dataflow_similarity_result)
        df_equal = df_dataflow_candidate_pair[df_dataflow_candidate_pair['is_equal'] == 1]
        static_dynamic_dict, block_static_dict, df_dynmic_ins_indexed, df_block_id_ins_indexed = bsd_map.process(
            program)

        # eval_mse, eval_exclusive, eval_force_merge_prop_his, eval_sdc_only, normal, def_use_length, length_same_reg
        # evaluation_methods = ['eval_mse', 'eval_exclusive', 'normal',
        #                       'def_use_length', 'length_same_reg']
        #  'eval_sdc_only', 'normal', 'eval_exclusive', 'eval_mse'
        evaluation_methods = ['with_dataflow', 'normal']
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'k']
        markers = ['o', '^', 's', 'd']
        pop_dict = {}
        for em in evaluation_methods:
            Config.evaluation_method = em
            if Config.run_mode == 'TRAIN' or Config.run_mode == 'TRAIN_TEST':
                # train
                if Config.model == 'SINGLE_PROJ':
                    # train
                    pop, logbook = train(target_program=program)
                    pop.sort(key=lambda x: x.fitness.values)
                    saveobj(pop, "save/model/single_model_" + program.progname)

            if Config.run_mode == 'TEST' or Config.run_mode == 'TRAIN_TEST':
                if Config.model == 'SINGLE_PROJ':
                    pop = loadobj("save/model/single_model_" + program.progname)
                    plot_front(pop, 0, do_plot_train_parato_front=True, do_plot_test_parato_front=True,
                               target_program=program)

            pop_dict[em] = pop

        # plot all pops
        fig, ax = plt.subplots()
        ax.set_title(program.progname)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('#Fault-Similarity Classes')
        ax.set_xscale('log')
        for i in range(len(evaluation_methods)):
            em = evaluation_methods[i]
            Config.evaluation_method = em
            pop = pop_dict[em]

            if Config.run_mode == 'TRAIN' or Config.run_mode == 'TRAIN_TEST':
                train_pareto_front = np.array([ind.fitness.values for ind in pop])
                if Config.fake_scale:
                    ax.scatter(pow(10, train_pareto_front[:, 0] / 200), train_pareto_front[:, 1], marker=markers[i],
                               c='',
                               edgecolors=colors[i],
                               label=em + '_train')
                else:
                    ax.scatter(train_pareto_front[:, 0], train_pareto_front[:, 1], marker=markers[i], c='',
                               edgecolors=colors[i],
                               label=em + '_train')

            if Config.run_mode == 'TEST' or Config.run_mode == 'TRAIN_TEST':

                test_fitnesses = test(pop)
                test_pareto_front = np.array(test_fitnesses)
                if Config.fake_scale:
                    ax.scatter(pow(10, test_pareto_front[:, 0] / 200), test_pareto_front[:, 1], marker=markers[i],
                               c=colors[i],
                               label=em + '_test')
                else:
                    ax.scatter(test_pareto_front[:, 0], test_pareto_front[:, 1], marker=markers[i], c=colors[i],
                               label=em + '_test')
        ax.legend(loc="lower right")
        saveimage(plt, 'save/plt/' + program.progname + '_' + Config.run_mode + '.png')
        plt.show()

    return


if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    main2()

    # pop.sort(key=lambda x: x.fitness.values)
    #
    # # print(stats)
    # # print("Convergence: ", convergence(pop, optimal_front))
    # # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    #
    # front = np.array([ind.fitness.values for ind in pop])
    #
    # # optimal_front = numpy.array(optimal_front)
    # # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis([0, Config.max_size * Config.test_size, 0.75, 1.0])
    # plt.show()
