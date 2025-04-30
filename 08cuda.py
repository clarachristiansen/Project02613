from os.path import join
import matplotlib.pyplot as plt
import sys
from multiprocessing.pool import Pool
import numpy as np
from numba import cuda
from time import time
from simulate import load_floorplans, summary_stats

@cuda.jit
def iteration_kernel(u0, interior_mask, u1):
    i, j = cuda.grid(2)
    if (i > 0 and i < u0.shape[0] - 1) and (j > 0 and j < u0.shape[1] - 1):
        if interior_mask[i - 1, j - 1]:
            u1[i, j] = 0.25 * (u0[i-1, j] + u0[i+1, j] + u0[i, j-1] + u0[i, j+1])

@cuda.jit
def copy_kernel(u0, u1):
    i, j = cuda.grid(2)
    if (i < u0.shape[0]) and (j < u0.shape[1]):
        u0[i, j] = u1[i, j]


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u0 = np.copy(u)

    d_u0 = cuda.to_device(u0) # to GPU
    d_u1 = cuda.to_device(u0) # to GPU
    d_interior_mask = cuda.to_device(interior_mask)
    #d_delta = cuda.to_device(np.zeros(u0.shape))
    tpb = 16, 16
    bpg = u0.shape[1]//16, u0.shape[0]//16
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        iteration_kernel[bpg, tpb](d_u0, d_interior_mask, d_u1)
        copy_kernel[bpg, tpb](d_u0, d_u1)
    u0 = d_u0.copy_to_host()
    return u0

if __name__ == '__main__':
    N = 1
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    building_ids, all_u0, all_interior_mask = load_floorplans()
    # Constant arguments (standard)
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    _ = jacobi(all_u0[0], all_interior_mask[0], 10, ABS_TOL)
    # The wrapper function has to feed each tuple to the jacobi function
    t0 = time()
    u_list = [jacobi(u0, interior_mask, MAX_ITER, ABS_TOL) for u0, interior_mask in zip(all_u0, all_interior_mask)]
    t = time() - t0
    print(t)
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, u_list, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
