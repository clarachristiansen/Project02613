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
def delta_kernel(u0, u1, delta):
    i, j = cuda.grid(2)
    if (i < u0.shape[0]) and (j < u0.shape[1]):
        delta[i, j] = abs(u0[i, j] - u1[i, j])

@cuda.jit
def copy_kernel(u0, u1):
    i, j = cuda.grid(2)
    if (i < u0.shape[0]) and (j < u0.shape[1]):
        u0[i, j] = u1[i, j]

@cuda.jit
def block_max_kernel(data, out, n):
    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y # Index in block
    i, j = cuda.grid(2) # Global index
    s = 1
    sdata = cuda.shared.array((16, 16), data.dtype)
    sdata[tidx, tidy] = data[i, j] if (i < n[0] and j < n[1]) else 0.0
    cuda.syncthreads()
    while s < max(cuda.blockDim.x, cuda.blockDim.y):
        i2x = 2*s*tidx
        i2y = 2*s*tidy
        if i2x < cuda.blockDim.x and i2y < cuda.blockDim.y:
            sdata[i2x, i2y] = max(sdata[i2x, i2y], sdata[i2x + s, i2y], sdata[i2x, i2y + s], sdata[i2x + s, i2y + s])
        s *= 2
        cuda.syncthreads() # Sync. block
    if tidx == 0 and tidy == 0: # First thread in block
        out[cuda.blockIdx.x, cuda.blockIdx.y] = sdata[0, 0]

def max_reduce(d_x, tpb):
    shape = d_x.shape
    bpg = shape[0]//tpb[0] + 1, shape[1]//tpb[1] + 1
    out = cuda.device_array(bpg, dtype=d_x.dtype)
    while bpg[0] > 1 and bpg[1] > 1:
        block_max_kernel[bpg, tpb](d_x, out, shape)
        shape = bpg
        bpg = shape[0]//tpb[0] + 1, shape[1]//tpb[1] + 1
        d_x[:shape[0], :shape[1]] = out[:shape[0], :shape[1]]
    block_max_kernel[bpg, tpb](d_x, out, shape)


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u0 = np.copy(u)

    d_u0 = cuda.to_device(u0) # to GPU
    d_u1 = cuda.to_device(u0) # to GPU
    d_interior_mask = cuda.to_device(interior_mask)
    d_delta = cuda.to_device(np.zeros(u0.shape))
    tpb = 16, 16
    bpg = u0.shape[1]//16, u0.shape[0]//16
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        iteration_kernel[bpg, tpb](d_u0, d_interior_mask, d_u1)
        
        if i % 1000 == 999:
            delta_kernel[bpg, tpb](d_u0, d_u1, d_delta)
            max_reduce(d_delta, tpb)
            delta = d_delta[0, 0]
            if delta < atol:
                #print(i)
                break
        
        copy_kernel[bpg, tpb](d_u0, d_u1)
    #print(i)
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
    """
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, u_list, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    """
    num_over_18 = 0
    num_under_15 = 0
    mean_temp_list = []
    sum_mean_temp = 0
    sum_std_temp = 0
    for bid, u, interior_mask in zip(building_ids, u_list, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        mean_temp_list.append(stats[stat_keys[0]])
        sum_mean_temp += stats[stat_keys[0]]
        sum_std_temp += stats[stat_keys[1]]
        if stats[stat_keys[2]] >= 50:
            num_over_18 += 1
        if stats[stat_keys[3]] >= 50:
            num_under_15 += 1
    mean_temp_array = np.array(mean_temp_list)
    np.save("AllMeans.npy", mean_temp_array)
    print("Average mean temperature:\n" + str(sum_mean_temp/N))
    print("Average standard deviation of temperature:\n" + str(sum_std_temp/N))   
    print("Number of buildings with at least 50%% area above 18C:\n" + str(num_over_18))
    print("Number of buildings with at least 50%% area below 15C:\n" + str(num_under_15))
    