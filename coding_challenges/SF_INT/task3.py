import numpy as np
import itertools

def eval(factors):
    factors_sorted = sorted(factors, reverse=True)
    if np.product(factors_sorted) != 36 or factors_sorted[0] == factors_sorted[1]:
        return False, factors_sorted, np.sum(factors_sorted)
    return True, factors_sorted, np.sum(factors_sorted)

factors = [1, 2, 3, 4, 6, 9, 12, 18, 36]

for comb in itertools.combinations(factors, 3):
    res, c, sum = eval(comb)
    if res:
        print(c, sum)

