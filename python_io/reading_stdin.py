import sys

def read_input():
    k_lists = []

    # Read space separated values.
    K_str, M_str = sys.stdin.readline().split(" ")

    # Read values are strings so we need to convert to int
    K, M = int(K_str), int(M_str)

    # Read out the lists
    for k in range(K):
        # Read whole line, split by spaces
        line_list = sys.stdin.readline().split(" ")

        # Read values are strings so we need to convert to int
        list_int = [int(li) for li in line_list[1:]]

        k_lists.append(list_int)

    return K, M, k_lists

if __name__ == '__main__':
    K, M, k_lists = read_input()
    print(K, M)
    print(k_lists)