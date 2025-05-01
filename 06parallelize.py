from os.path import join
import sys
from multiprocessing.pool import Pool
import numpy as np
from simulate import load_data, load_floorplans, jacobi, summary_stats

def jacobi_wrapper(args):
    """ Has to make a wrapper function for the jacobi when using Pool """
    u0, interior_mask, max_iter, abs_tol = args
    return jacobi(u0, interior_mask, max_iter, abs_tol)

if __name__ == '__main__':
    building_ids, all_u0, all_interior_mask = load_floorplans()
    
    # Constant arguments (standard)
    CORES = int(sys.argv[2])
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    # The wrapper function has to feed each tuple to the jacobi function
    args_list = [(u0, interior_mask, MAX_ITER, ABS_TOL) for u0, interior_mask in zip(all_u0, all_interior_mask)] 
    # To make the program static schedueling

    with Pool(CORES) as pool:
        all_u = np.array(pool.map(jacobi_wrapper, args_list))

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))