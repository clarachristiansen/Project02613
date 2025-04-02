import matplotlib.pyplot as plt
import numpy as np

def plot_speedup(cores, speedup):
    plt.plot(cores, speedup)
    plt.xlabel('Number of processors')
    plt.ylabel('Speedup')
    plt.title('Speed-up')
    plt.grid()
    plt.tight_layout()
    plt.savefig('Figures/05_speedup.png')

def estimate_parrallel_fraction(cores, speedup):
    # Rewritten from Amdahl's law
    # S(N) = 1/((1-F) * F/N)
    # S(N) * ((1-F) * F/N) = 1
    # S(N) - S(N) * F + S(N) * F/N = 1
    # S(N) - F * (S(N) - S(N)/N) = 1
    # F = (S(N) - 1) / (S(N) - S(N)/N)
    # F = (S(N) - 1) / (S(N) (1 - 1/N)
    return (speedup - 1) / (speedup * (1 - 1 / cores))

if __name__ == '__main__':
    cores = np.array([1, 2, 4, 8, 16])
    time_str = ['8m54.016s', '4m48.439s', '3m38.149s', '2m22.484s', '2m2.209s']
    time = [int(x.split('m')[0])*60 + int(x.split('m')[1].split('.')[0]) for x in time_str]
    print(time)
    baseline = time[0]  # Baseline (1 core time)
    speedup = np.array([baseline / t for t in time])

    plot_speedup(cores, speedup)

    F = estimate_parrallel_fraction(cores, speedup)
    print(F)

    # Theoretical max
    amdahls_max = 1/(1 - F[-1])
    print(f"Theoretical max (Amdahl's law): {amdahls_max}")
    print(f"Proportion of max achieved ({cores[-1]} cores): {speedup[-1] / amdahls_max}")
    total_estimated_time = 4571 / 50 * time[-1]
    print(f"Estimated time to process all buildings: {total_estimated_time / 60 / 60} h")