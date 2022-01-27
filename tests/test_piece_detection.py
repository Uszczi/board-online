import unittest
import sys, os
import glob
import json

import numpy as np
import cv2 as cv

sys.path.append("./src/detector")

from board_detector import BoardDetector
from field_detector import FieldDetector, draw_corners
from piece_detector import PieceDetector

TEST_DATA = "./tests/test_data"
DEBUG = True

def testPieceDetection():
    image_files = glob.glob(f"{TEST_DATA}/*.JPG")

    for image_name in image_files:
        cv.destroyAllWindows()

        img = cv.imread(image_name)
        img = cv.resize(img, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)
        original = img.copy()

        detector = BoardDetector(img, crop=True, rotate=False, debug=False)
        img = detector.get_board()

        if len(img) == 0:
            continue

        detector = FieldDetector(img, debug=False)
        fields = detector.detect_fields()

        try:
            piece_detector = PieceDetector(img, debug=DEBUG)
            pieces = piece_detector.detect_pieces(fields)

            # if DEBUG:
            #     draw_corners(img, fields)
            print("Pieces: ", pieces.plain())
        except Exception:
            print("No pieces detected!")

        cv.imshow('original', original)
        cv.imshow('img', img)
        cv.waitKey(0)

if __name__ == "__main__":
    testPieceDetection()
    cv.destroyAllWindows()
