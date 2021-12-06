from utils import find_sudoku_contour, crop_to_contour, extract_digits

import cv2


def sudoku_image(img, size, dig=True, sudoku_grid_size=400):
    # we work with grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extract sudoku contour
    contour = find_sudoku_contour(gray)

    # crop to found contour
    cropped = crop_to_contour(gray, contour, sudoku_grid_size)

    # digits
    digits = None
    if dig:
        # find digits
        digits = extract_digits(cropped)

    return digits, cv2.resize(cropped, (size, size))
