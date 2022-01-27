import sys
import numpy as np
import cv2 as cv

sys.path.append("./src/detector")

from board_detector import BoardDetector

TEST_DATA = "./tests/test_data"

img = cv.imread(f"{TEST_DATA}/full.JPG")
img = cv.resize(img, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)

detector = BoardDetector(img, debug=True)
img = detector.get_board()

cv.imshow('dst',img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()