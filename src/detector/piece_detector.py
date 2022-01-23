import string
import json

import numpy as np
import cv2 as cv

from checkers_board import CheckersBoard

class PieceDetector:
    def __init__(self, image, debug=False):
        self.image = image
        self.debug = debug

    def detect_pieces(self, fields):
        board = CheckersBoard()
        avg_value = np.average(self.image) - 20
        fields = np.int0(fields)

        for x in range(0, len(fields) - 1):
            y0 = 1 if x % 2 == 0 else 0
            for y in range(y0, len(fields[x]) - 1, 2):
                top_left, bottom_right = fields[x][y], fields[x + 1][y + 1]

                field_image = self.__get_field_image(top_left, bottom_right)
                field_name = self.from_xy(x, y)

                if self.__is_white_piece(field_image, avg_value):
                    board.add_white(field_name)
                    self.__drawDebug(top_left, bottom_right, (69, 255, 69))
                elif self.__is_black_piece(field_image, avg_value):
                    board.add_black(field_name)
                    self.__drawDebug(top_left, bottom_right, (69, 69, 255))

        return board

    def __drawDebug(self, pt1, pt2, color):
        if self.debug:
            cv.rectangle(self.image, pt1, pt2, color, thickness=2)

    def __get_field_image(self, top_left, bottom_right):
        return self.image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    def __is_white_piece(self, field_image, threshold = 100):
        return np.average(field_image) > threshold

    def __is_black_piece(self, field_image, threshold_red = 100):
        return np.average(field_image[:, :, 2]) > threshold_red

    def from_xy(self, x, y):
        return f"{string.ascii_uppercase[x]}{8 - y}"

