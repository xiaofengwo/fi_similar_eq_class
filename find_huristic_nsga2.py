#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json
import sys

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import numpy as np
import data
from configuration import Config
from collections import defaultdict
from collections import Counter


# prepare data
df_results_with_machine_states = data.inner_join_result_and_machine_states(Config.results_path, Config.machine_states_path, Config.results_with_machine_states_path)
x_train, x_test, y_train, y_test = data.load_data_from_csv(Config.results_with_machine_states_path)

# Problem definition
VALID_BITS = 64
BOUND_LOW, BOUND_UP = 0, 2**(VALID_BITS - 10)
MASK_BITS_BOUNDS_LIST = []
X_train_rows_count, NDIM = x_train.shape

def onemax(individual):
    f1 = individual[0]
    f2 = sum(individual)
    return f1, f2

# def mxkmultibitflip(individual, indpb, validbits):
#     for i in range(len(individual)):
#         for i in range(0, validbits-1):
#             if np.random.random() < indpb:
#                 mask = (mask | 1) << 1
#             else:
#                 mask = mask << 1
#         if np.random.random() < indpb:
#             mask = mask | 1
#         individual[i] = individual[i] ^ mask
#     return individual,

# def mxkmultibitflip(individual, indpb):
#     for i in range(len(individual)):
#         mask = np.uint64(0)
#         one = np.uint64(1)
#         for j in range(0, VALID_BITS - 1):
#             if np.random.random() < indpb:
#                 mask = (mask | one) << one
#             else:
#                 mask = mask << one
#         if np.random.random() < indpb:
#             mask = mask | one
#         individual[i] = individual[i] ^ mask
#     return individual,

# def random_mask(bit_bound):
#     for i in range(NDIM):
#         mask = np.uint64(0)
#         one = np.uint64(1)
#         cur_bit = 0
#         while cur_bit < MASK_BITS_BOUNDS_LIST[i]:
#             if np.random.randint(0, 2) == 1:
#                 mask = mask | one
#             cur_bit = cur_bit + 1
#             if cur_bit < MASK_BITS_BOUNDS_LIST[i]:
#                 mask = mask << one
#         individual[i] = individual[i] ^ mask


# def mxkmultibitflip(individual, indpb):
#     for i in range(len(individual)):
#         # mask = random_mask(bit_bound)
#         mask = np.uint64(0)
#         one = np.uint64(1)
#         cur_bit = 0
#         while cur_bit < MASK_BITS_BOUNDS_LIST[i]:
#             if np.random.random() < indpb:
#                 mask = mask | one
#             cur_bit = cur_bit + 1
#             if cur_bit < MASK_BITS_BOUNDS_LIST[i]:
#                 mask = mask << one
#         individual[i] = individual[i] ^ mask
#     return individual,

def mxkmultibitflip(individual, indpb):
    for i in range(len(individual)):
        # mask = random_mask(bit_bound)
        mask = np.uint64(0)
        one = np.uint64(1)
        cur_bit = 0
        if np.random.random() < indpb:
            while cur_bit < MASK_BITS_BOUNDS_LIST[i]:
                mask = mask | one
                cur_bit = cur_bit + 1
            if cur_bit < MASK_BITS_BOUNDS_LIST[i]:
                mask = mask << one
        individual[i] = individual[i] ^ mask
    return individual,

def alloneinit(validbits):
    return 2**(validbits) - 1

def allzeroinit(validbits):
    return np.uint64(0)

def alloneinit(validbits):
    return 2 ** (validbits) - 1

def random_init():
    init_np_ndarray = np.zeros(NDIM, dtype=np.uint64)
    for i in range(NDIM):
        mask = np.uint64(0)
        # one = np.uint64(1)
        # cur_bit = 0
        # while cur_bit < MASK_BITS_BOUNDS_LIST[i]:
        #     if np.random.randint(0, 2) == 1:
        #         mask = mask | one
        #     cur_bit = cur_bit + 1
        #     if cur_bit < MASK_BITS_BOUNDS_LIST[i]:
        #         mask = mask << one
        init_np_ndarray[i] = init_np_ndarray[i] ^ mask
    return init_np_ndarray

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


def accuracy(individual):
    id_label_dict = defaultdict(list)
    npmask = np.array(individual, dtype=np.uint64)
    masked_x_train = x_train & npmask
    for i in range(masked_x_train.shape[0]):
        id_label_dict[tuple(masked_x_train[i].tolist())].append(tuple((y_train[i]).tolist()))

    # calculate accuracy
    right_predicted = 0
    for k in id_label_dict:
        label_list = list(id_label_dict[k])
        result = Counter(label_list)
        right_predicted = right_predicted + max(list(result.values()))
    eq_accuracy = right_predicted / x_train.shape[0]

    num_eq_class = len(id_label_dict)

    return num_eq_class, eq_accuracy

creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
# creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
# creator.create("Individual", list, fitness=creator.FitnessMin)
# creator.create("Individual", array.array, typecode=np.int64, fitness=creator.FitnessMin)
creator.create("Individual", np.ndarray, typecode=np.uint64, fitness=creator.FitnessMin)
toolbox = base.Toolbox()



toolbox.register("random_init", random_init)
# toolbox.register("attr_bool", random.randint, BOUND_LOW, BOUND_UP)
# toolbox.register("attr_int", allzeroinit, VALID_BITS)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", accuracy)
# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)
# flip each attribute/gene of 0.05
toolbox.register("mutate", mxkmultibitflip, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
# toolbox.register("select", tools.selSPEA2)


def main(seed=None):
    random.seed(seed)

    global MASK_BITS_BOUNDS_LIST
    MASK_BITS_BOUNDS_LIST = get_mask_bounds(x_train)

    NGEN = 100
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
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
        pop = toolbox.select(pop + offspring, MU)

        front = numpy.array([ind.fitness.values for ind in pop])
        # print(front)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook


if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = main()
    pop.sort(key=lambda x: x.fitness.values)

    print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    import matplotlib.pyplot as plt
    import numpy

    front = numpy.array([ind.fitness.values for ind in pop])

    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()

