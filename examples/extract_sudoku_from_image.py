import numpy as np

from src.visual_sudoku_solver import extractor
import cv2.cv2 as cv2

# load sudoku image
img = cv2.imread('data/image.jpg')

# crop and save
crop = extractor.sudoku_image(img, 400)
cv2.imwrite('data/cropped.jpg', crop)

# read digits and save
digits = extractor.digits_from_sudoku_image(crop)
np.savetxt('data/digits.txt', digits.astype(int), fmt='%s')

# show cropped sudoku
cv2.imshow('Sudoku', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

