"""
Challenge from leetcode. Solution is not complete, needs one more fix to update all components.
"""

import numpy as np


class Solution:
    def comps_in_4hood(self, m, n, M, N):
        n_hood = []
        if m > 0:
            n_hood.append([m - 1, n])
        if n > 0:
            n_hood.append([m, n - 1])
        if m < M - 1:
            n_hood.append([m + 1, n])
        if n < N - 1:
            n_hood.append([m, n + 1])

        return n_hood

    def largestIsland(self, grid) -> int:
        grid = np.array(grid)

        if not np.any(grid == 0):
            return grid.shape[0] * grid.shape[1]

        comp_grid = np.ones_like(grid) * -1
        M, N = grid.shape

        largest_island = 0

        # Go over grid and find components
        cur_comp_idx = 0
        comp_dict = {}
        for m in range(M):
            for n in range(N):
                # Island
                if grid[m, n] == 1:
                    # Check if there are comps in 4hood
                    has_adjacent_comp = False
                    for nh in self.comps_in_4hood(m, n, M, N):
                        # If has adjacent components:
                        adj_comp_idx = comp_grid[nh[0], nh[1]]
                        if adj_comp_idx >= 0:
                            comp_grid[m, n] = adj_comp_idx
                            comp_dict[adj_comp_idx] += 1

                            if comp_dict[adj_comp_idx] > largest_island:
                                largest_island = comp_dict[adj_comp_idx]

                            has_adjacent_comp = True
                            break

                    if not has_adjacent_comp:
                        comp_grid[m, n] = cur_comp_idx
                        comp_dict[cur_comp_idx] = 1
                        cur_comp_idx += 1

        # Second pass, for each zero in grid try to join all neighboring components
        for m in range(M):
            for n in range(N):
                if grid[m, n] == 0:
                    # Find out size of all neighboring components
                    size_sum = 1
                    added_comps_list = []
                    for nh in self.comps_in_4hood(m, n, M, N):
                        adj_comp_idx = comp_grid[nh[0], nh[1]]
                        if adj_comp_idx >= 0:
                            if adj_comp_idx not in added_comps_list:
                                size_sum += comp_dict[adj_comp_idx]
                                added_comps_list.append(adj_comp_idx)

                    if size_sum > largest_island:
                        largest_island = size_sum

        return largest_island

if __name__=="__main__":
    grid = [[1,1],
            [1,0]]

    s = Solution()
    sol = s.largestIsland(grid)
    print(sol)

