import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8) #調大小改x
markerImage = cv.aruco.drawMarker(dictionary, 0, 50, markerImage, 1);
#cv.aruco.drawMarker(dictionary, y, x, markerImage, 1) 調大小改x 換id改y

cv.imwrite("6x6marker501.png", markerImage);

