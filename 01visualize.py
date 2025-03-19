import matplotlib.pyplot as plt
from simulate import load_floorplans

# run script 
# python 01visualize.py 6

building_ids, all_u0, all_interior_mask = load_floorplans()

#visualize
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(all_u0[i], cmap='magma')
    ax.set_title(f"Building {building_ids[i]}")
    ax.axis('off')
plt.tight_layout()

#save 
plt.savefig('Figures/01visualizefloorplans.png')





