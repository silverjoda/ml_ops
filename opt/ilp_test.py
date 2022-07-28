# Import OR-Tools wrapper for linear programming
from ortools.linear_solver import pywraplp

UNITS = [
    '🗡️Swordsmen',
    '🛡️Men-at-arms',
    '🏹Bowmen',
    '❌Crossbowmen',
    '🔫Handcannoneers',
    '🐎Horsemen',
    '♞Knights',
    '🐏Battering rams',
    '🎯Springalds',
    '🪨Mangonels',
]

DATA = [
    [60, 20, 0, 6, 70],
    [100, 0, 20, 12, 155],
    [30, 50, 0, 5, 70],
    [80, 0, 40, 12, 80],
    [120, 0, 120, 35, 150],
    [100, 20, 0, 9, 125],
    [140, 0, 100, 24, 230],
    [0, 300, 0, 200, 700],
    [0, 250, 250, 30, 200],
    [0, 400, 200, 12*3, 240]
]

RESOURCES = [183000, 90512, 80150]


def solve_army(UNITS, DATA, RESOURCES):
  # Create the linear solver using the CBC backend
  solver = pywraplp.Solver('Maximize army power', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

  # 1. Create the variables we want to optimize
  units = [solver.IntVar(0, solver.infinity(), unit) for unit in UNITS]

  # 2. Add constraints for each resource
  for r, _ in enumerate(RESOURCES):
    solver.Add(sum(DATA[u][r] * units[u] for u, _ in enumerate(units)) <= RESOURCES[r])

  # 3. Maximize the new objective function
  solver.Maximize(sum((10*DATA[u][-2] + DATA[u][-1]) * units[u] for u, _ in enumerate(units)))

  # Solve problem
  status = solver.Solve()

  # If an optimal solution has been found, print results
  if status == pywraplp.Solver.OPTIMAL:
    print('================= Solution =================')
    print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
    print()
    print(f'Optimal value = {solver.Objective().Value()} 💪power')
    print('Army:')
    for u, _ in enumerate(units):
      print(f' - {units[u].name()} = {units[u].solution_value()}')
  else:
    print('The solver could not find an optimal solution.')

solve_army(UNITS, DATA, RESOURCES)