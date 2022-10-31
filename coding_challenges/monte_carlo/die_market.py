import numpy as np
from collections import OrderedDict

n_trials = 10000

def run_trial():
    sum = 0
    vals = []
    while True:
        rnd_val = np.random.randint(1,7)
        sum += rnd_val
        if sum > 10:
            return vals[-1]
        vals.append(sum)

vals = {}
for trial in range(n_trials):
    last_val = run_trial()
    if last_val not in vals.keys():
        vals[last_val] = 1
    else:
        vals[last_val] += 1

vals = OrderedDict(sorted(vals.items()))

val_n_sum = 0
for k, v in vals.items():
    val_n_sum += vals[k]

for k, v in vals.items():
    vals[k] /= val_n_sum

print(vals.items())
print([i / sum(range(1, 7)) for i in range(1, 7)])