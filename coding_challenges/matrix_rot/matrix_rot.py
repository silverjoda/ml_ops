"""
Matrix rotation challenge from hackerrank (solution times out on 3/10 tests)
"""

from copy import deepcopy

def rotate(matrix_A, matrix_B):
    M, N = len(matrix_A), len(matrix_A[0])

    m_midpoint = M / 2
    n_midpoint = N / 2

    for m in range(M):
        for n in range(N):
            if m >= m_midpoint:
                if n >= n_midpoint:
                    # right, up
                    if n < (N - 1 - (M - 1 - m)):
                        # Move right
                        matrix_B[m][n + 1] = matrix_A[m][n]
                    else:
                        # Move up
                        matrix_B[m - 1][n] = matrix_A[m][n]
                else:
                    # right, down
                    if m < (M - 1 - (n)):
                        # Move down
                        matrix_B[m + 1][n] = matrix_A[m][n]
                    else:
                        # Move right
                        matrix_B[m][n + 1] = matrix_A[m][n]
            else:
                if n >= n_midpoint:
                    # left, up
                    if m > (N - 1 - n):
                        # Move up
                        matrix_B[m - 1][n] = matrix_A[m][n]
                    else:
                        # Move left
                        matrix_B[m][n - 1] = matrix_A[m][n]
                else:
                    # left, down
                    if n > (0 + m):
                        # Move left
                        matrix_B[m][n - 1] = matrix_A[m][n]
                    else:
                        # Move down
                        matrix_B[m + 1][n] = matrix_A[m][n]


def matrixRotation(matrix, r):
    matrix_A = deepcopy(matrix)
    matrix_B = deepcopy(matrix)
    for i in range(r):
        if i % 2 == 0:
            rotate(matrix_A, matrix_B)
        else:
            rotate(matrix_B, matrix_A)

    printmat = matrix_B
    if r % 2 == 0:
        printmat = matrix_A

    mat_str = ""
    for m in printmat:
        mat_str += (" ".join(str(x) for x in m) + '\n')
    print(mat_str)

if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    m = int(first_multiple_input[0])

    n = int(first_multiple_input[1])

    r = int(first_multiple_input[2])

    matrix = []

    for _ in range(m):
        matrix.append(list(map(int, input().rstrip().split())))

    matrixRotation(matrix, r)
