import string
import json

import numpy as np
import cv2 as cv

from checkers_board import CheckersBoard

class PieceDetector:
    def __init__(self, image, debug=False):
        self.lower_white = np.array([0,0,20])
        self.upper_white = np.array([180,30,255])

        # lower boundary RED color range values; Hue (0 - 10)
        self.lower1 = np.array([0, 70, 50])
        self.upper1 = np.array([20, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        self.lower2 = np.array([150, 70, 50])
        self.upper2 = np.array([180, 255, 255])

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv = cv.GaussianBlur(hsv, (5,5), 1)

        self.image = hsv
        self.debug = debug
        self.debug_img = image

    def detect_pieces(self, fields):
        board = CheckersBoard()
        fields = np.int0(fields)

        for x in range(0, len(fields) - 1):
            y0 = 1 if x % 2 == 0 else 0
            for y in range(y0, len(fields[x]) - 1, 2):
                top_left, bottom_right = fields[x][y], fields[x + 1][y + 1]

                field_image = self.__get_field_image(top_left, bottom_right)
                field_name = self.from_xy(x, y)

                if self.__is_white_piece(field_image):
                    board.add_white(field_name)
                    self.__drawDebug(top_left, bottom_right, (69, 255, 69))
                elif self.__is_black_piece(field_image):
                    board.add_black(field_name)
                    self.__drawDebug(top_left, bottom_right, (69, 69, 255))

        return board

    def __drawDebug(self, pt1, pt2, color):
        if self.debug:
            cv.rectangle(self.debug_img, pt1, pt2, color, thickness=2)

    def __get_field_image(self, top_left, bottom_right):
        return self.image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    def __is_white_piece(self, field_image, threshold = 60):
        mask = cv.inRange(field_image, self.lower_white, self.upper_white)
        return np.average(mask) > threshold

    def __is_black_piece(self, field_image, threshold = 80):
        lower_mask = cv.inRange(field_image, self.lower1, self.upper1)
        upper_mask = cv.inRange(field_image, self.lower2, self.upper2)
        full_mask = lower_mask | upper_mask

        # cv.imshow("full_mask", full_mask)
        # cv.waitKey(0)

        return np.average(full_mask) > threshold

    def from_xy(self, x, y):
        return f"{string.ascii_uppercase[x]}{8 - y}"

