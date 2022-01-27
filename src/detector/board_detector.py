import numpy as np
import cv2 as cv

class BoardDetector:
    def __init__(self, image, crop=True, rotate=True, debug=False):
        self.crop = crop
        self.rotate = rotate
        self.debug = debug

        self.update_image(image)

    def __debug_show(self, image):
        if self.debug:
            cv.imshow(f"{image}", image)
            cv.waitKey(0)

    def update_image(self, image):
        self.image = image
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        self.__debug_show(self.gray)

        # Remove details
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,8))
        opening = cv.morphologyEx(self.gray, cv.MORPH_OPEN, kernel)

        # Blur image - remove more details
        self.filtered = cv.GaussianBlur(opening, (3,3), 1)
        self.__debug_show(self.filtered)

    def get_board(self):
        rect = self.detect_board()
        box = np.int0(cv.boxPoints(rect))

        if self.debug:
            cv.rectangle(self.image, box[0], box[2], (0, 0, 255), 2)
            cv.imshow('image', self.image)
            cv.waitKey(0)

        if rect is not None:
            return self.crop_minAreaRect(self.image, rect)
        return self.image

    def detect_board(self):
        _, thresh = cv.threshold(self.filtered, 50, 255, cv.THRESH_BINARY_INV)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,8))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

        self.__debug_show(opening)

        contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_rect = None
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_rect = cv.minAreaRect(cnt)

        return max_rect

    def crop_minAreaRect(self, img, rect):
        angle = np.clip(rect[2], 0, 90)
        rows,cols = img.shape[0], img.shape[1]
        box = cv.boxPoints(rect)

        if self.rotate:
            M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img_rot = cv.warpAffine(img, M, (cols,rows))
            pts = np.int0(cv.transform(np.array([box]), M))[0]
        else:
            pts = np.int0(box)
            img_rot = img

        pts[pts < 0] = 0

        if self.crop:
            return img_rot[pts[1][1]:pts[2][1], pts[0][0]:pts[2][0]]
        return img_rot

