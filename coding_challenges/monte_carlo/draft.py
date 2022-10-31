import random
import numpy as np

N = 50
arr = list(range(N))
missing_val = random.randint(0,49)
del arr[missing_val]
print(f"Missing val: {missing_val}")

# Estimate missing val using for loop
sorted_arr = sorted(arr)

for i in range(N):
    if sorted_arr[i] != i:
        print(f"Missing val found at: {i}")
        break

