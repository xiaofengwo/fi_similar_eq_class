import numpy as np
import random
import math


#  flip every bit with a equal probality
def mxkmultibitflip_eq(individual, indpb, validbits):
    for i in range(len(individual)):
        mask = np.uint64(0)
        one = np.uint64(1)
        for j in range(0, validbits - 1):
            if np.random.random() < indpb:
                mask = (mask | one) << one
            else:
                mask = mask << one
        if np.random.random() < indpb:
            mask = mask | one
        individual[i] = individual[i] ^ mask
    return individual,


# flip by a random mask number
def mxkmultibitflip_by_a_random_number(individual, indpb, mask_bits_bounds_list):
    if np.random.random() < indpb:
        for i in range(len(individual)):
            max_maxk = math.pow(2, mask_bits_bounds_list[i]) - 1
            mask = random.randint(0, max_maxk)
            individual[i] = individual[i] ^ np.uint64(mask)
    return individual,


# only flip the effective bits given by mask_bits_bounds_list and with BITWISE FLIP!!!
def mxkmultibitflip_bitwise(individual, indpb, mask_bits_bounds_list):
    for i in range(len(individual)):
        # mask = random_mask(bit_bound)
        mask = np.uint64(0)
        one = np.uint64(1)
        cur_bit = 0
        while cur_bit < mask_bits_bounds_list[i]:
            if np.random.random() < indpb:
                mask = mask | one
            cur_bit = cur_bit + 1
            if cur_bit < mask_bits_bounds_list[i]:
                mask = mask << one
        individual[i] = individual[i] ^ mask
    return individual,


# only flip the effective bits given by mask_bits_bounds_list AND ALL BITS FLIP AT A TIME
def mxkmultibitflip_allbits(individual, indpb, mask_bits_bounds_list):
    for i in range(len(individual)):
        # mask = random_mask(bit_bound)
        mask = np.uint64(0)
        one = np.uint64(1)
        cur_bit = 0
        if np.random.random() < indpb:
            while cur_bit < mask_bits_bounds_list[i]:
                # while cur_bit < 64:
                mask = mask | one
                cur_bit = cur_bit + 1
                if cur_bit < mask_bits_bounds_list[i]:
                    # if cur_bit < 64:
                    mask = mask << one
            individual[i] = individual[i] ^ mask
    return individual,


def mxkmultibitflip_8bits(individual, indpb, mask_bits_bounds_list):
    for i in range(len(individual)):
        # mask = random_mask(bit_bound)
        mask = np.uint64(0)
        one = np.uint64(1)
        cur_bit = 0
        cur_byte_index = 0
        one_byte_flip = np.uint64(0xFF)
        bytes_count = mask_bits_bounds_list[i] // 8
        for j in range(bytes_count):
            one_byte_flip = one_byte_flip << np.uint64(j * 8)
            if np.random.random() < indpb:
                mask = mask | one_byte_flip
        individual[i] = individual[i] ^ mask
    return individual,
