"""
Estimate pi in one line using sampling
"""

import numpy as np

# Declare amount of monte carlo samples
N = 10000000

# One line estimation of PI using monte carlo
pi = 4. * np.count_nonzero(np.sqrt(np.square(np.random.rand(N, 2).astype(np.float32)).sum(axis=1)) < 1.0) / N

# Print to console
print(pi)

