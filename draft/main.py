import numpy as np
N = 10000
n_floors = 10
nums = np.random.randint(0, n_floors, size=(N, 2))

floor_freqs = np.zeros(n_floors)

for n in nums:
    if n[0] == n[1]: continue
    floor_freqs[min(n): max(n) + 1] += 1

import matplotlib.pyplot as plt
plt.plot(floor_freqs)
plt.show()

