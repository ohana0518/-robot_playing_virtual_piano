"""白鍵預處理+找到白鍵座標"""
import cv2
import numpy as np
distance = 10
def preprocessing_white(img):
    srcImg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, srcImg_gray) = cv2.threshold(srcImg_gray, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow("whitegray", srcImg_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
    opening = cv2.morphologyEx(srcImg_gray, cv2.MORPH_OPEN, kernel)
    return opening

def findwhitekey(binary_pic,tone_list):
    contours, hierarchy = cv2.findContours(binary_pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    xy_old_coordinate = np.empty([len(tone_list), 2], dtype="int32")
    iter = 0
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        xy_old_coordinate[iter][0], xy_old_coordinate[iter][1], iter = int((x*2 + w)/2), int((y*2 + h)/2), iter + 1
    xy_old_coordinate = np.asarray(sorted(xy_old_coordinate, key=lambda s: s[0]), dtype="int32")
    return xy_old_coordinate





