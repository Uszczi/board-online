import unittest
import sys, os
import glob
import json

from pprint import pprint
import io

import numpy as np
import cv2 as cv

sys.path.append("./src/detector") # TODO: How to Fix this? xdd

from board_detector import BoardDetector, draw_corners
from piece_detector import PieceDetector

class PieceDetectionTest(unittest.TestCase):
    TEST_DATA = "./tests/test_data"

    def testPieceDetection(self):
        image_files = glob.glob(f"{self.TEST_DATA}/*.JPG")
        # image_files = [f"{self.TEST_DATA}/full.JPG" ]

        for image_name in image_files:
            cv.destroyAllWindows()

            print(image_name)

            img = cv.imread(image_name)
            img = cv.resize(img, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)

            detector = BoardDetector(img, rotate=False, debug=False)
            img = detector.get_board()

            if len(img) == 0:
                continue

            detector.update_image(img)
            fields = detector.detect_fields()

            piece_detector = PieceDetector(img, debug=True)
            pieces = piece_detector.detect_pieces(fields)

            json_file = os.path.splitext(image_name)[0] + ".json"
            with open(json_file) as expected_data:
                expected = json.load(expected_data)

            draw_corners(img, fields)

            cv.imshow('img', img)
            cv.waitKey(0)

            # self.assertTrue(self.__compare(expected, pieces.plain()))

    def __compare(self, data1, data2):
        return self.__ordered(data1) == self.__ordered(data2)

    def __ordered(self, obj):
        if isinstance(obj, dict):
            return sorted((k, self.__ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(self.__ordered(x) for x in obj)
        else:
            return obj

if __name__ == "__main__":
    unittest.main()
    cv.destroyAllWindows()
