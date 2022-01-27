import sys
import numpy as np
import cv2 as cv

from detector.board_detector import BoardDetector
from detector.field_detector import FieldDetector, draw_corners
from detector.piece_detector import PieceDetector

camera = cv.VideoCapture(0)

if camera is None:
    print("Cannot detect camera")
    sys.exit(1)

while(True):
    ret, frame = camera.read()

    if ret is False:
        print("No frame from camera")
        break

    img = cv.resize(frame, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)
    detector = BoardDetector(img, rotate=False)

    img = detector.get_board()

    if len(img) == 0:
        print("No pieces detected!")
        continue

    detector = FieldDetector(img)
    fields = detector.detect_fields()

    try:
        piece_detector = PieceDetector(img)
        pieces = piece_detector.detect_pieces(fields)
        print("Pieces: ", pieces.plain())
    except Exception:
        print("No pieces detected!")

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()


