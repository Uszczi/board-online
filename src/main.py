import sys
sys.path.append("./src/detector")

import numpy as np
import cv2 as cv

from board_detector import BoardDetector
from field_detector import FieldDetector, draw_corners
from piece_detector import PieceDetector

camera = cv.VideoCapture("./src/IMG_0668.MOV")

if camera is None:
    print("Cannot detect camera")
    sys.exit(1)

while(True):
    ret, frame = camera.read()

    if ret is False:
        print("No frame from camera")
        break

    img = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
    detector = BoardDetector(img, crop=False, rotate=False)

    img = detector.get_board()

    if len(img) == 0:
        print("No pieces detected!")
        continue

    detector = FieldDetector(img)
    fields = detector.detect_fields()

    try:
        piece_detector = PieceDetector(img, debug=True)
        pieces = piece_detector.detect_pieces(fields)
        print("Pieces: ", pieces.plain())
    except Exception:
        print("No pieces detected!")

    cv.imshow("camera", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()


