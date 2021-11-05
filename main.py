import cv2
import numpy as np
import pytesseract

# load picture and apply basic filtering and 01 thresholding
image = cv2.imread('img/sudoku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# find contours and their areas
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contourAreas = [cv2.contourArea(c) for c in contours]
contourMax = contours[np.argmax(contourAreas)]

# mask max contour
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [contourMax], 0, 255, -1)
cv2.drawContours(mask, [contourMax], 0, 0, 2)

# apply mask
out = np.zeros_like(gray)
out[mask == 255] = gray[mask == 255]

# apply mask to grayscale image
crop_start = np.min(np.where(mask == 255), axis=1)
crop_end = np.max(np.where(mask == 255), axis=1)
out_cropped = out[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

# find polygon for contour
epsilon = 0.01 * cv2.arcLength(contourMax, True)
approximations = cv2.approxPolyDP(contourMax, epsilon, True)
# subtract crop region
approximations -= np.flip(crop_start)
poly = cv2.polylines(out_cropped, [approximations], True, color=(255, 0, 0), thickness=1)
# cv2.imshow("cropped", poly)

# find homography
DST_SIZE = 400
dst = np.array([(0, 0), (0, DST_SIZE), (DST_SIZE, DST_SIZE), (DST_SIZE, 0)])
H, _ = cv2.findHomography(approximations, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)

# un warp with homography
sudoku = cv2.warpPerspective(out_cropped, H, (DST_SIZE, DST_SIZE), flags=cv2.INTER_LINEAR)
# cv2.imshow("sudoku", sudoku)

# apply more blur / thresh
_, sudoku = cv2.threshold(sudoku, 100, 255, cv2.THRESH_BINARY)
# sudoku = cv2.GaussianBlur(sudoku, (11, 11), 0)
# sudoku = cv2.adaptiveThreshold(sudoku, 255, 1, 1, 11, 2)

# find contours for 3x3 and 1x1 sudoku boxes
box_size_3x3 = int(DST_SIZE / 3)
box_size_1x1 = int(DST_SIZE / 9)

sudoku_rgb = cv2.cvtColor(sudoku, cv2.COLOR_BGR2RGB)

numbers = np.zeros([9, 9])
for row in range(9):
    for col in range(9):
        box = sudoku_rgb[row * box_size_1x1 + 8:(row + 1) * box_size_1x1 - 2,
              col * box_size_1x1 + 10:(col + 1) * box_size_1x1 - 5]
        text = pytesseract.image_to_string(box, config='--psm 10')
        if len(text) > 0 and text[0].isdigit():
            numbers[row, col] = int(text[0])
            print('found number :', text[0])
        else:
            print(f'no number found ({text})')

        # cv2.imshow('box', box)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print(numbers)

cv2.waitKey(0)
cv2.destroyAllWindows()
