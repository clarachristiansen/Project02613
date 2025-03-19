import matplotlib.pyplot as plt
from simulate import load_floorplans

# run script 
# python 01visualize.py 4

building_ids, all_u0, all_interior_mask = load_floorplans()

#visualize
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for i in range(4):
    axs[0,i].imshow(all_u0[i], cmap='magma')
    axs[0,i].set_title(f"Building {building_ids[i]}")
    axs[0,i].axis('off')

    axs[1,i].imshow(all_interior_mask[i], cmap='gray')
    axs[1,i].axis('off')

plt.tight_layout()

#save 
plt.savefig('Figures/01visualizefloorplans.png')





