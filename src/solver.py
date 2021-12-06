import numpy as np


def solve(digits):
    for p, d in np.ndenumerate(digits):
        if d == 0:
            for i in range(1, 9):
                digits[p] = i
                valid, digits_next = solve(digits)
                if valid:
                    print(f'found valid digit {i} for {p}')
                    digits = digits_next
                    break
    return check_sudoku(digits), digits


def check_sudoku_1to9(digits):
    non_1to9 = (digits < 0) & (digits > 9)
    return np.sum(non_1to9) == 0


def check_sudoku_unique(array):
    # count non unique entries
    u, c = np.unique(array, return_counts=True)
    dup = u[(c > 1) & (u > 0)]
    return len(dup) == 0


def check_sudoku(digits):
    # check that only digits 1 to 9 are present
    if not check_sudoku_1to9(digits):
        return False

    # check rows
    for row in digits:
        if not check_sudoku_unique(row):
            return False

    # check columns
    for col in digits.T:
        if not check_sudoku_unique(col):
            return False

    # check boxes
    for br in range(3):
        for bc in range(3):
            if not check_sudoku_unique(digits[3*br:3*(br+1), 3*bc:3*(bc+1)]):
                return False

    return True
