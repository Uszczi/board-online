import itertools

import numpy as np
import cv2 as cv

CORNER_COUNT = 9

def draw_corners(img, corners):
    for row in range(len(corners)):
        for point in range(len(corners[row])):
            color = (0, 0, 255) if row == point else (0,255,0)
            cv.circle(img, (int(corners[row][point][0]), int(corners[row][point][1])), 7, color, 2)
            cv.putText(img,  f"{(row, point)}", (int(corners[row][point][0]), int(corners[row][point][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255))

class BoardDetector:
    def __init__(self, image, debug=False):
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

        return self.crop_minAreaRect(self.image, rect)

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
        angle = rect[2]
        rows,cols = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rot = cv.warpAffine(img,M,(cols,rows))

        # rotate bounding box
        box = cv.boxPoints(rect)
        pts = np.int0(cv.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                        pts[1][0]:pts[2][0]]

        return img_crop

    def detect_fields(self):
        corners = self.__detect_corners()
        locations = self.__get_corner_locations(corners)
        filtered = self.__sort_and_cleanup_corners(locations)

        if self.debug:
            img = np.copy(self.image)
            draw_corners(img, filtered)
            cv.imshow('image', img)
            cv.waitKey(0)

        try:
            return self.__redetect_corners(filtered)
        except IndexError:
            print("WARN: Cannot match board fields")
            return []

    def __redetect_corners(self, corners):
        transposed = [[item for item in row if item is not None] for row in itertools.zip_longest(*corners)]
        y_values = []
        x_values = []

        # Re-detect points by virtual line intersection
        for row in range(len(corners)):
            y_values.append(np.median(corners[row], 0)[1])
            x_values.append(np.median(transposed[row], 0)[0])

        new_corners = np.full((len(x_values), len(y_values), 2), 0)
        for x in range(len(x_values)):
            for y in range(len(y_values)):
                new_corners[x, y] = (x_values[x], y_values[y])

        return new_corners

    def __detect_corners(self):
        # Detect corners
        filtered = np.float32(self.filtered)
        corners = cv.cornerHarris(filtered, 4, 5, 0.04)

        # Remove noise around corners
        filtered_corner_data = cv.morphologyEx(corners, cv.MORPH_CLOSE, None)
        return filtered_corner_data

    def __get_corner_locations(self, corner_data):
        # Get local maximas
        ret, dst = cv.threshold(corner_data, 0.1*corner_data.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Get corner centers (subpixel accuracy)
        corners = cv.cornerSubPix(self.gray, np.float32(centroids), (5,5), (-1,-1), criteria)
        return corners

    def __sort_and_cleanup_corners(self, corners, field_size=60):
        rows = [[]]

        # Sort by Y-axis
        corners = corners[corners[:, 1].argsort()]

        # Classify row of each point
        prev_y = corners[0, 1]
        for point in corners:
            if abs(point[1] - prev_y) > field_size/4:
                rows.append([]) # If distance is greater then classify to next row

            prev_y = point[1]
            rows[-1].append(point)

        output = []
        current_index = 0
        prev_y = 0
        for row in rows:
            # Sort each row by X-values
            row.sort(key=lambda k: k[0])

            # Filter entire rows
            if current_index > 0 and abs(row[0][1] - prev_y) < field_size/3 or len(row) < 3:
                current_index += 1
                continue

            current_index += 1
            prev_y = row[0][1]

            # Median distance on X-axis in a row
            avg = np.average(np.abs([j[0] - i[0] for i, j in zip(row[:-1], row[1:])]))

            # Filter points on X-axis
            indexes_to_remove = []
            prev_index = 0
            for point in row[1:]:
                if abs(point[0] - row[prev_index][0]) < avg/2:
                    indexes_to_remove.append(prev_index + 1)
                prev_index += 1

            for index in reversed(indexes_to_remove):
                del row[index]

            output.append(row)

        return output
