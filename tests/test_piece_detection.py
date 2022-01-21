import unittest
import sys, os
import glob
import json

import numpy as np
import cv2 as cv

sys.path.append("./src/detector") # TODO: How to Fix this? xdd

from board_detector import BoardDetector
from piece_detector import PieceDetector

class PieceDetectionTest(unittest.TestCase):
    TEST_DATA = "./tests/test_data"

    def testPieceDetection(self):
        # image_files = glob.glob(f"{self.TEST_DATA}/*.JPG")
        image_files = [f"{self.TEST_DATA}/full.JPG" ]

        for image_name in image_files:
            img = cv.imread(image_name)
            img = cv.resize(img, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)

            detector = BoardDetector()
            fields = detector.detect_fields(img)

            piece_detector = PieceDetector()
            pieces = piece_detector.detect_pieces(img, fields)

            json_file = os.path.splitext(image_name)[0] + ".json"
            with open(json_file) as expected_data:
                expected = json.load(expected_data)

            self.assertTrue(self.__compare(expected, pieces.plain()))

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