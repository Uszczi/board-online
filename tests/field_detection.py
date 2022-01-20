import sys
import numpy as np
import cv2 as cv

sys.path.append("./src/detector") # TODO: How to Fix this? xdd

from board_detector import BoardDetector

TEST_DATA = "./tests/test_data"

img = cv.imread(f"{TEST_DATA}/full.JPG")
img = cv.resize(img, None, fx=0.4, fy=0.4, interpolation = cv.INTER_CUBIC)

detector = BoardDetector()
corners = detector.detect_fields(img)

for row in range(len(corners)):
    for point in range(len(corners[row])):
        color = (0, 0, 255) if row == point else (0,255,0)
        cv.circle(img, (int(corners[row][point][0]), int(corners[row][point][1])), 7, color, 2)
        cv.putText(img,  f"{(point, row)}", (int(corners[row][point][0]), int(corners[row][point][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255))

cv.imshow('dst',img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()