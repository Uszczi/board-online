import numpy as np
import cv2 as cv

CORNER_COUNT = 9

class BoardDetector:
    def detect_fields(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        corners = self.__detect_corners(gray)
        locations = self.__get_corner_locations(gray, corners)
        filtered = self.__sort_and_cleanup(locations)

        return filtered

    def __detect_corners(self, image):
        # Remove details
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,8))
        opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

        # Blur image - remove more details
        blurred = cv.GaussianBlur(opening, (3,3), 0)

        # Detect corners
        blurred = np.float32(blurred)
        corners = cv.cornerHarris(blurred, 4, 5, 0.04)

        # Remove noise around corners
        filtered_corner_data = cv.morphologyEx(corners, cv.MORPH_CLOSE, None)
        return filtered_corner_data

    def __get_corner_locations(self, gray_image, corner_data):
        # Get local maximas
        ret, dst = cv.threshold(corner_data, 0.1*corner_data.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Get corner centers (subpixel accuracy)
        corners = cv.cornerSubPix(gray_image, np.float32(centroids), (5,5), (-1,-1), criteria)
        return corners

    def __sort_and_cleanup(self, corners):
        MAX_DELTA = 3
        rows = [[]]

        # Sort by Y-axis
        corners = corners[corners[:, 1].argsort()]

        # Classify row of each point
        prev_y = corners[0, 1]
        for point in corners:
            if abs(point[1] - prev_y) > MAX_DELTA:
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
            if current_index > 0 and abs(row[0][1] - prev_y) < 15 or len(row) < 8:
                current_index += 1
                continue

            current_index += 1
            prev_y = row[0][1]

            # Averange distance on X-axis in a row
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
