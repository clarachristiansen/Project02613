from numba import jit
import sys
import numpy as np
import time
from simulate import load_floorplans, summary_stats

@jit(nopython=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    
    u_new_interior = np.copy(u)
    
    for k in range(max_iter):
        delta = 0
        for i in range(1,u.shape[0]-1):
            for j in range(1,u.shape[1]-1):
                if interior_mask[i-1, j-1]:
                    u_new_interior[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    delta = max(delta, abs(u[i, j] - u_new_interior[i, j]))
        u = u_new_interior.copy()
        if delta < atol:
            break
    return u

if __name__ == '__main__':
    # Load data
    building_ids, all_u0, all_interior_mask = load_floorplans()

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)

    #run jacobi once to compile the function
    u0 = all_u0[0]
    interior_mask = all_interior_mask[0]
    u_test = jacobi_numba(u0, interior_mask, 10, ABS_TOL)

    
    #time the jacobi iterations
    s = time.time()

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
    
    e = time.time()
    print(f"Jacobi iterations took {e-s:.2f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
