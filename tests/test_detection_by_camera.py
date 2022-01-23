import sys
import numpy as np
import cv2 as cv

sys.path.append("./src/detector")

from board_detector import BoardDetector, draw_corners
from piece_detector import PieceDetector

camera = cv.VideoCapture(0)

if camera is None:
    print("Cannot detect camera")
    sys.exit(1)

while(True):
    ret, frame = camera.read()

    if ret is False:
        print("No frame from camera")
        continue

    img = cv.resize(frame, None, fx=0.8, fy=0.8, interpolation = cv.INTER_CUBIC)

    detector = BoardDetector(img, rotate=False, debug=False)
    img = detector.get_board()

    if len(img) == 0:
        print("No board detected!")
        continue

    try:
        detector.update_image(img)
        fields = detector.detect_fields()

        piece_detector = PieceDetector(img, debug=True)
        pieces = piece_detector.detect_pieces(fields)

        print(pieces.plain())
        cv.imshow('img', img)
    except Exception as e:
        print(str(e))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
