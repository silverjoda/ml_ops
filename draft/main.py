# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import itertools

def read_input():
    k_lists = []

    # Read first two numbers
    K_str, M_str = sys.stdin.readline().split(" ")
    K, M = int(K_str), int(M_str)

    # Read out the lists
    for k in range(K):
        line_list = sys.stdin.readline().split(" ")
        list_int = [int(li) for li in line_list[1:]]
        k_lists.append(list_int)

    return K, M, k_lists

def calc_maximum(k_lists, M):
    best_res = 0
    for prod in itertools.product(*k_lists):
        mod_res = sum([p ** 2 for p in prod]) % M
        if mod_res > best_res:
            best_res = mod_res
    return best_res

def calc_maximum_loop(k_lists, M):
    def increment_counter(counter, counts):
        for i in range(len(counts)):
            if counter[i] < (counts[i] - 1):
                counter[i] += 1
                break
            counter[i] = 0

    best_res = 0
    k_counts = [len(k) for k in k_lists]
    k_counter = [0] * len(k_lists)

    while True:
        # Increment counter
        increment_counter(k_counter, k_counts)

        # Map counter to solution
        mod_res = sum([k_lists[i][k] ** 2 for i, k in enumerate(k_counter)]) % M
        if mod_res > best_res:
            best_res = mod_res

        # Check if we should break
        if all([k_counter[i] == (k_counts[i] - 1) for i in range(len(k_counts))]):
            break

    return best_res

def calc_maximum_rec(k_lists, cum_sum, M):
    K = len(k_lists)

    if K == 0:
        return cum_sum % M

    current_list = k_lists[0]

    best_res = 0
    for k in range(len(current_list)):
        res = calc_maximum_rec(k_lists[1:], cum_sum + current_list[k] ** 2, M)
        if res > best_res:
            best_res = res

    return best_res

if __name__ == '__main__':
    K, M, k_lists = read_input()
    s_max = calc_maximum_loop(k_lists, M)
    print(s_max)


