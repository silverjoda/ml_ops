import os

# Make data dir if doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Path relative to current directory
file_name = "data/text_file.txt"

# Robust way of getting filename
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, file_name)

with open(file_name, 'w') as f:
    f.write("First line \n")

print("Written to file")

# Append to file
with open(file_name, '+a') as f:
    f.write("Second line \n")

# Append to file with multiple lines
with open(file_name, '+a') as f:
    f.writelines(["Third line", "\n" ,"Fourth line"])

# Read contents of file
with open(file_name, 'r') as f:
    content = f.readlines()

print("Content from file:")
print(content)
