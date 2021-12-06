from src.utils import *
from src.solver import *

# load sudoku picture
img = cv2.imread('img/sudoku.jpg')

# extract sudoku image and cropped output
digits, crop = sudoku_image(img, 400, dig=True)

# show digits
print(digits)

print(solve(digits))

# show cropped sudoku
cv2.imshow('Sudoku', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
