import matplotlib.pyplot as plt
from simulate import load_floorplans, jacobi
from Ex07numbaJacobi import jacobi_numba
import numpy as np

# run script 
# python 03visualize.py 4

building_ids, all_u0, all_interior_mask = load_floorplans()

# Run jacobi iterations for each floor plan
MAX_ITER = 20_000
ABS_TOL = 1e-4

all_u = np.empty_like(all_u0)
for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    #u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
    all_u[i] = u

#visualize
fig, axs = plt.subplots(1, 4, figsize=(20, 6))
for i in range(4):
    axs[i].imshow(all_u[i], cmap='magma')
    axs[i].set_title(f"Building {building_ids[i]}")
    axs[i].axis('off')

plt.tight_layout()

#save 
plt.savefig('Figures/03visualizefloorplans.png')
#plt.savefig('Figures/03visualizefloorplansCuda.png')





