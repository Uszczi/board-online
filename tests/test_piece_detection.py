import unittest
import sys, os
import glob
import json

from pprint import pprint
import io

import numpy as np
import cv2 as cv

sys.path.append("./src/detector") # TODO: How to Fix this? xdd

from board_detector import BoardDetector
from piece_detector import PieceDetector

class PieceDetectionTest(unittest.TestCase):
    TEST_DATA = "./tests/test_data"

    def testPieceDetection(self):
        image_files = glob.glob(f"{self.TEST_DATA}/*.JPG")
        #image_files = [f"{self.TEST_DATA}/full.JPG" ]

        for image_name in image_files:
            img = cv.imread(image_name)
            img = cv.resize(img, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)

            detector = BoardDetector(img)
            img = detector.get_board()

            detector.update_image(img)
            fields = detector.detect_fields()

            piece_detector = PieceDetector(img, debug=True)
            pieces = piece_detector.detect_pieces(fields)

            json_file = os.path.splitext(image_name)[0] + ".json"
            with open(json_file) as expected_data:
                expected = json.load(expected_data)

            corners = fields
            for row in range(len(corners)):
                for point in range(len(corners[row])):
                    color = (0, 0, 255) if row == point else (0,255,0)
                    cv.circle(img, (int(corners[row][point][0]), int(corners[row][point][1])), 7, color, 2)
                    cv.putText(img,  f"{(row, point)}", (int(corners[row][point][0]), int(corners[row][point][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255))

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