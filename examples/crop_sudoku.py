from visual_sudoku_solver import extractor
import cv2.cv2 as cv2

# load sudoku picture
img = cv2.imread('img/sudoku.jpg')
crop = extractor.sudoku_image(img, 400)
cv2.imwrite('img/sudoku_cropped.jpg', crop)
