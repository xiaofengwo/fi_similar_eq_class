import numpy as np


def allzeroinit(NDIM):
    return np.zeros(NDIM, dtype=np.uint64)


def alloneinit(NDIM):
    init_np_ndarray = np.zeros(NDIM, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    for i in range(NDIM):
        init_np_ndarray[i] = init_np_ndarray[i] ^ mask
    return init_np_ndarray


def random_init(NDIM, MASK_BITS_BOUNDS_LIST):
    init_np_ndarray = np.zeros(NDIM, dtype=np.uint64)
    for i in range(NDIM):
        mask = np.uint64(0)
        one = np.uint64(1)
        cur_bit = 0
        while cur_bit < MASK_BITS_BOUNDS_LIST[i]:
            if np.random.randint(0, 2) == 1:
                mask = mask | one
            cur_bit = cur_bit + 1
            if cur_bit < MASK_BITS_BOUNDS_LIST[i]:
                mask = mask << one
        init_np_ndarray[i] = init_np_ndarray[i] ^ mask
    return init_np_ndarray