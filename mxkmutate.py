import numpy as np


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
                mask = mask | one
                cur_bit = cur_bit + 1
                if cur_bit < mask_bits_bounds_list[i]:
                    mask = mask << one
            individual[i] = individual[i] ^ mask
    return individual,