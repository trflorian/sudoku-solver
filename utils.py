import cv2
import numpy as np
import pytesseract

# for digit extraction
SUDOKU_GIRD_SIZE = 400


def find_sudoku_contour(img):
    # apply blur filtering and 01 thresholding
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # find contours and their areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourAreas = [cv2.contourArea(c) for c in contours]
    contourMax = contours[np.argmax(contourAreas)]

    return contourMax


def crop_to_contour(img, contour, dst_size):
    # mask max contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    cv2.drawContours(mask, [contour], 0, 0, 2)

    # apply mask
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    # apply mask to grayscale image
    crop_start = np.min(np.where(mask == 255), axis=1)[:2]
    crop_end = np.max(np.where(mask == 255), axis=1)[:2]
    out_cropped = out[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

    # find polygon for contour
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approximations = cv2.approxPolyDP(contour, epsilon, True)
    # subtract crop region
    approximations -= np.flip(crop_start)
    poly = cv2.polylines(out_cropped, [approximations], True, color=(255, 0, 0), thickness=1)
    # cv2.imshow("cropped", poly)

    # find homography
    dst = np.array([(0, 0), (0, dst_size), (dst_size, dst_size), (dst_size, 0)])
    H, _ = cv2.findHomography(approximations, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    # un warp with homography
    return cv2.warpPerspective(out_cropped, H, (dst_size, dst_size), flags=cv2.INTER_LINEAR)


def extract_digits(img):
    # apply binary threshold
    _, sudoku = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # find contours for 3x3 and 1x1 sudoku boxes
    box_s = int(img.shape[0] / 9)

    sudoku_rgb = cv2.cvtColor(sudoku, cv2.COLOR_BGR2RGB)

    numbers = np.zeros([9, 9])
    for row in range(9):
        for col in range(9):
            box = sudoku_rgb[row * box_s + 8:(row + 1) * box_s - 2,
                  col * box_s + 10:(col + 1) * box_s - 5]
            text = pytesseract.image_to_string(box, config='--psm 10')
            if len(text) > 0 and text[0].isdigit():
                numbers[row, col] = int(text[0])
                print('found number :', text[0])
            else:
                print(f'no number found ({text})')

    return numbers


def sudoku_image(img, size, dig=True):
    # we work with grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extract sudoku contour
    contour = find_sudoku_contour(gray)

    # crop to found contour
    cropped = crop_to_contour(gray, contour, SUDOKU_GIRD_SIZE)

    # digits
    digits = None
    if dig:
        # find digits
        digits = extract_digits(cropped)

    return digits, cv2.resize(cropped, (size, size))
