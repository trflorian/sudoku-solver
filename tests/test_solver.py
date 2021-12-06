from visual_sudoku_solver import solver

import pytest
import numpy as np


@pytest.mark.parametrize("start,end,expected", [
    (2, 7, True),
    (1, 8, True),
    (0, 9, True),
    (1, 10, False),
    (0, 10, False),
    (-5, 5, False),
    (-8, -3, False),
    (12, 16, False),
])
def test_1to9_range(start, end, expected):
    # end is inclusive
    range_array = np.array(range(start, end+1))
    assert solver.check_sudoku_0to9(range_array) == expected
