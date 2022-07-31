import os

import numpy as np

# Make data dir if doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Path relative to current directory
file_name = "data/npy_file.npy"

# Robust way of getting filename
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, file_name)

# Make array
arr = np.arange(10)

# Write array to file
np.save(filename, arr)

# Read contents and print
arr_loaded = np.load(filename)
print(arr_loaded)