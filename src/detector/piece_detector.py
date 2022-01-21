import string
import json

import numpy as np
import cv2 as cv

from checkers_board import CheckersBoard

class PieceDetector:
    def detect_pieces(self, image, fields):
        board = CheckersBoard()
        for x in range(0, len(fields) - 1):
            y0 = 1 if x % 2 == 0 else 0
            for y in range(y0, len(fields[x]) - 1, 2):
                field_image = self.__get_field_image(image, fields[x][y], fields[x + 1][y + 1])
                if self.__is_piece_on_field(field_image):
                    field_name = self.from_xy(x, y)
                    if self.__is_white_piece(field_image):
                        board.add_white(field_name)
                    else:
                        board.add_black(field_name)
        return board

    def __get_field_image(self, image, top_left, bottom_right):
        return image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    def __is_piece_on_field(self, field_image):
        avg = np.average(field_image)
        return avg > 50

    def __is_white_piece(self, field_image):
        return np.average(field_image[:,:,1]) > 80

    def from_xy(self, x, y):
        return f"{string.ascii_uppercase[x]}{8 - y}"

