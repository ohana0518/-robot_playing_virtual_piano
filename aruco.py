import cv2
import numpy as np
import time
from math import *

import cv2.aruco as aruco

dist = np.array(([[0.02996607, 0.05339151, 0.00265015, 0.0009626, -0.1547677]]))
newcameramtx = ([[690.11325726, 0, 344.43035488], [0., 689.63654094, 222.98096563], [0., 0., 1.]])
mtx = np.array([[690.11325726, 0, 344.43035488], [0., 689.63654094, 222.98096563], [0., 0., 1.]])

matrix = [0 for i in range(50)]
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
j = 0

while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    frame = dst1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        # 顯示圖片
    cv2.imshow('live', frame)
    # 按下 q 鍵離開迴圈

    if ids is not None:
        rvec, tevc, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        (rvec - tevc).any()

        for i in range(rvec.shape[0]):
            if j >= 5:
                break


            else:
                aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tevc[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners)
                rotate_matrix = cv2.Rodrigues(rvec)[0]
                RM = np.array(rotate_matrix)
                theta_x = np.arctan2(RM[1, 0], RM[0, 0]) / np.pi * 180
                theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2])) / np.pi * 180
                theta_z = np.arctan2(RM[2, 1], RM[2, 2]) / np.pi * 180

                matrix[j] = int(theta_x)
                print(f"Euler angles:\ntheta_x: {matrix[j]}")
                key = cv2.waitKey(2000)
                matrix[j] = matrix[j] * (-1)
                j = j + 1
                print(i)
                print(j)
                cv2.putText(frame, "id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('frame' + '.png', frame)
        print('儲存:', 'frame' + '.png')
        print(matrix[4])
        break

    else:
        cv2.putText(frame, "No ids ", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

cap.release()
cv2.destroyAllWindows()


def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置參數爲角度參數負值表示順時針旋轉; 1.0位置參數scale是調整尺寸比例（圖像縮放參數），建議0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此處爲白色，可自定義
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默認是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


img = cv2.imread('frame.png')
cv2.imshow("frame", frame)
frameRotation = rotate_bound_white_bg(frame, matrix[4])
# frame = cv2.resize(frame, (256, 192))
# frameRotation=cv2.resize(frame, (256, 192))
cv2.imshow("frame", frame)
cv2.imshow("frameRotation", frameRotation)
cv2.waitKey(0)


