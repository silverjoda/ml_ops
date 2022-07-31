import csv
import os

import numpy as np

# Make data dir if doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Path relative to current directory
file_name = "data/data.csv"

# Robust way of getting filename
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, file_name)

# Make array
data = np.arange(10)

# Write array to file
with open(filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(data)

# Read contents and print
with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
