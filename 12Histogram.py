import numpy as np
import matplotlib.pyplot as plt
means = np.load("AllMeans.npy")
hist, bin_edges = np.histogram(means, bins=20)
plt.stairs(hist, bin_edges, fill=True)
plt.xlabel("Mean temperature in C")
plt.ylabel("Number of buildings")
plt.savefig("Figures/12histogram.png")