from visual_sudoku_solver import solver

import numpy as np

# load digits
digits = np.loadtxt('data/digits.txt')

print(digits)

valid, sol = solver.solve(digits)

print('Found solution:', valid)
print(sol)
