import string
import json

import numpy as np
import cv2 as cv

from checkers_board import CheckersBoard

class PieceDetector:
    def detect_pieces(self, image, fields):
        avg_value = np.average(image) - 20

        board = CheckersBoard()
        for x in range(0, len(fields) - 1):
            y0 = 1 if x % 2 == 0 else 0
            for y in range(y0, len(fields[x]) - 1, 2):
                field_image = self.__get_field_image(image, fields[x][y], fields[x + 1][y + 1])
                field_name = self.from_xy(x, y)
                if self.__is_white_piece(field_image, avg_value):
                    board.add_white(field_name)
                elif self.__is_black_piece(field_image, avg_value):
                    board.add_black(field_name)

        return board

    def __get_field_image(self, image, top_left, bottom_right):
        return image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    def __is_white_piece(self, field_image, threshold = 100):
        return np.average(field_image) > threshold

    def __is_black_piece(self, field_image, threshold_red = 100):
        return np.average(field_image[:, :, 2]) > threshold_red

    def from_xy(self, x, y):
        return f"{string.ascii_uppercase[x]}{8 - y}"

