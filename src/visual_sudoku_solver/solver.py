import numpy as np


def to_digits(digits):
    """Convert any digits array to a usable np array of integer datatype"""
    if digits is np.array:
        if digits.dtype == int:
            # if already correctly configured just return
            return digits
        # if already np.array convert to integer
        return digits.astype(int)
    return np.array(digits, dtype=int)


def solve(digits):
    digits = to_digits(digits)

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


def check_sudoku_0to9(digits):
    digits = to_digits(digits)

    non_1to9 = (digits < 0) | (digits > 9)
    return np.sum(non_1to9) == 0


def check_sudoku_unique_0to9(digits):
    digits = to_digits(digits)

    # count non unique entries
    u, c = np.unique(digits, return_counts=True)
    dup = u[(c > 1) & (u > 0)]
    return len(dup) == 0


def check_sudoku(digits):
    digits = to_digits(digits)

    # check that only digits 1 to 9 are present
    if not check_sudoku_0to9(digits):
        return False

    # check rows
    for row in digits:
        if not check_sudoku_unique_0to9(row):
            return False

    # check columns
    for col in digits.T:
        if not check_sudoku_unique_0to9(col):
            return False

    # check boxes
    for br in range(3):
        for bc in range(3):
            if not check_sudoku_unique_0to9(digits[3*br:3*(br+1), 3*bc:3*(bc+1)]):
                return False

    return True
