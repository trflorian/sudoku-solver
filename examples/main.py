from visual_sudoku_solver import solver, extractor
import cv2.cv2 as cv2

# load sudoku picture
img = cv2.imread('img/sudoku.jpg')

# extract sudoku image and cropped output
crop = extractor.sudoku_image(img, 400)
digits = extractor.extract_digits(crop)

# show digits
print(digits)

print(solver.solve(digits))

# show cropped sudoku
cv2.imshow('Sudoku', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
