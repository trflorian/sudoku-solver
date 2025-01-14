from pathlib import Path
import cv2
import numpy as np

from PIL import Image

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

from model import DigitClassifier


def run_sudoku_detection(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_annotated = img.copy()

    # apply hough transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sorty by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # approximate contours, find first, biggest rectangle
    approx = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            break

    # draw approximated rectangle
    cv2.drawContours(img_annotated, [approx], 0, (0, 0, 255), 2)

    # get corners of rectangle, unwrap perspective to get top-down view and crop

    # Extract the coordinates of the contour
    pts = approx.reshape(4, 2)

    # order them from top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Compute the width and height of the new image
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the top-down view
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation matrix
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # apply clahe
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    warped = clahe.apply(gray_warped)

    cell_size = 80
    border = 10
    sudoku_grid = cv2.resize(
        warped, (9 * (cell_size + 2 * border), 9 * (cell_size + 2 * border))
    )

    cells = [np.vsplit(row, 9) for row in np.hsplit(sudoku_grid, 9)]

    # remove 10% of the cell size from each side
    cells = [[cell[border:-border, border:-border] for cell in row] for row in cells]

    # threshold the cells
    # cells = [[cv2.threshold(cell, 80, 255, cv2.THRESH_BINARY_INV)[1] for cell in row] for row in cells]

    # use adaptive thresholding
    cells = [
        [
            # cv2.adaptiveThreshold(
            #     cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=21, C=20
            # )
            cv2.threshold(cell, 80, 255, cv2.THRESH_BINARY_INV)[1]
            for cell in row
        ]
        for row in cells
    ]

    # apply morphology opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cells = [
        [cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel) for cell in row]
        for row in cells
    ]
    cells = [
        [cv2.morphologyEx(cell, cv2.MORPH_DILATE, kernel) for cell in row]
        for row in cells
    ]

    # find empty cells
    is_cell_empty = [[np.mean(cell) < 10 for cell in row] for row in cells]

    # show the cells
    composite = np.hstack([np.vstack(row) for row in cells])

    # convert to BGR
    composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)

    model = DigitClassifier.load_from_checkpoint("model.ckpt", map_location="cuda")
    model.eval()

    preproces_transform = T.Compose(
        [
            T.Resize((28, 28)), #, interpolation=InterpolationMode.NEAREST),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.1307,), (0.3081,)),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    # fill empty cells with red
    for i, row in enumerate(is_cell_empty):
        for j, is_empty in enumerate(row):
            if is_empty:
                cv2.rectangle(
                    composite,
                    (i * cell_size, j * cell_size),
                    ((i + 1) * cell_size, (j + 1) * cell_size),
                    (0, 0, 255),
                    -1,
                )
            else:
                # detect digit
                cell_img = Image.fromarray(cells[i][j])
                cell_img = preproces_transform(cell_img)
                cell_img = cell_img.unsqueeze(0).cuda()
                out = model(cell_img)

                cell_img= cells[i][j].copy()
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
                cell_img = cv2.resize(cell_img, (512, 512), interpolation=cv2.INTER_NEAREST)

                score = torch.nn.functional.softmax(out, dim=1)
                cv2.putText(
                    cell_img,
                    f"Predictions: {[round(float(s), 2) for s in score[0]]}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    cell_img,
                    f"Predicted: {torch.argmax(out, dim=1).item()}",
                    (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("Cell", cell_img)
                cv2.waitKey(0)


                digit = torch.argmax(out, dim=1).item()
                digit = f"{digit}"
                cv2.putText(
                    composite,
                    digit,
                    (i * cell_size + 5, j * cell_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

    return img_annotated, warped, composite


def main():
    for img_path in sorted(Path("images").glob("*.jpg"))[:1]:
        img = cv2.imread(str(img_path))

        img_annotated, warped, composite = run_sudoku_detection(img)

        cv2.imshow("Warped", warped)
        cv2.imshow("Image", img_annotated)
        cv2.imshow("Cells", composite)

        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
