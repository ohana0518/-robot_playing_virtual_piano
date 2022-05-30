import cv2 as cv
import argparse
import sys
import os.path
import numpy as np
import math
import play_song

screenWidth_mm =  41.4  # 螢幕寬度實際尺寸
screenHeight_mm = 31.55  # 螢幕高度實際尺寸


# 將僅鍵盤範圍的琴鍵角點座標轉成原始影像琴鍵座標，並計算琴鍵中心位置
def transformKeyBoxPoints(keys, M_INV, offset):
    boxPoints = np.arange(len(keys) * 4 * 2).reshape(len(keys), 4, 2)
    keyCenters = []
    for i in range(len(keys)):
        (x1, y1, x2, y2) = keys[i]
        points = [(x1 + offset[0], y1 + offset[1]), (x2 + offset[0], y1 + offset[1]), (x2 + offset[0], y2 + offset[1]),
                  (x1 + offset[0], y2 + offset[1])]
        for j in range(len(points)):
            pt = points[j]
            point = np.array([pt[0], pt[1], 1])
            pointRoated = M_INV.dot(point)
            boxPoints[i][j] = pointRoated
        rect = cv.minAreaRect(boxPoints[i])  # 包含琴鍵的最小矩形(rotatedrect)
        keyCenters.append((int(rect[0][0]), int(rect[0][1])))  # 取得琴鍵中心位置
    return keyCenters, boxPoints


# 將所有琴鍵的四邊形畫出，參數boxes shape為(琴鍵個數，4，2)
def drawRoatedBoxes(dst, boxes, centers):
    for i in range(boxes.shape[0]):
        cv.circle(dst, centers[i], 2, (255, 255, 0))
        for j in range(boxes.shape[1] - 1):
            pt1 = boxes[i][j]
            pt2 = boxes[i][j + 1]
            cv.line(dst, pt1, pt2, (255, 0, 255), 2)
        cv.line(dst, boxes[i][3], boxes[i][0], (255, 0, 255), 2)


# 畫出轉正後鍵盤範圍內偵測到的琴鍵矩形
def showKeyBox(dst, keys, offset):
    for i in range(len(keys)):
        (x1, y1, x2, y2) = keys[i]
        points = [(x1 + offset[0], y1 + offset[1]), (x2 + offset[0], y1 + offset[1]), (x2 + offset[0], y2 + offset[1]),
                  (x1 + offset[0], y2 + offset[1])]
        for j in range(len(points) - 1):
            cv.line(dst, points[j], points[j + 1], (255, 0, 255), 2)
        cv.line(dst, points[3], points[0], (255, 0, 255), 2)
        (test_width, text_height), baseline = cv.getTextSize(str(i), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.putText(dst, str(i), ((x1 + x2 - test_width) // 2, (y1 + y2) // 2 + offset[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv.LINE_AA)


# 使用直方圖垂直投影來偵測轉正後不包含黑鍵範圍的鍵盤區域的所有白鍵
def findWhiteKeys(src):
    thresh = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
    (height, width) = src.shape[:2]
    xsum = np.sum(thresh, axis=0) / 255  # vertical projection
    vprojection = np.zeros(src.shape, dtype=np.uint8)
   # cv.imshow('findWhiteKeys adaptiveThreshold', thresh)
    cv.waitKey(1)
    for i in range(width):
        for j in range(int(xsum[i])):
            vprojection[j, i] = 255
    #cv.imshow('findWhiteKeys vProject', vprojection)
    cv.waitKey(1)

    vProjectThreshold = 50  # 根據不包含黑鍵之琴鍵區域垂直投影特徵定此閥值
    vProjectWidth = 10  # 白鍵邊界垂直投影連續寬度
    xroi = xsum > vProjectThreshold
    startx = -1
    xseg = []
    for i in range(width):
        if xroi[i]:
            if startx == -1:
                startx = i
        else:
            endx = i
            if startx != -1:
                if (endx - startx) >= vProjectWidth:  # 白鍵邊界寬度
                    print("startx: {}, endx: {}\n".format(startx, endx))
                    xseg.append((startx, endx))
                startx = -1
    whiteKeys = []
    starty = 0
    endy = height - 1
    for i in range(len(xseg)):
        startx, endx = xseg[i]
        whiteKeys.append((startx, starty, endx, endy))
    return whiteKeys


# 使用直方圖水平與垂直投影來偵測轉正後鍵盤區域的所有黑鍵
def findBlackKeys(gray):
    (_, thresh) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # morphology processing: defining a rectangular structure
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
   # cv.imshow('closed', closed)
    cv.waitKey(1)
    (height, width) = closed.shape[:2]
    xsum = np.sum(closed, axis=0) / 255  # vertical projection
    vprojection = np.zeros(closed.shape, dtype=np.uint8)
    for i in range(width):
        for j in range(int(xsum[i])):
            vprojection[j, i] = 255
  #  cv.imshow('findBlackKeys vProject', vprojection)
    ysum = np.sum(closed, axis=1) / 255  # horizontal projection
    hprojection = np.zeros(closed.shape, dtype=np.uint8)
    for j in range(height):
        for i in range(int(ysum[j])):
            hprojection[j, i] = 255
  #  cv.imshow('findBlackKeys hProject', hprojection)
    cv.waitKey(1)

    vProjectThreshold = height * 0.5  # 黑鍵垂直投影閥值
    vProjectWidth = 10  # 黑鍵垂直投影連續寬度
    xroi = xsum > vProjectThreshold
    startx = -1
    xseg = []
    for i in range(width):
        if xroi[i]:
            if startx == -1:
                startx = i
        else:
            endx = i
            if startx != -1:
                if (endx - startx) > vProjectWidth :
                    print("startx: {}, endx: {}\n".format(startx, endx))
                    xseg.append((startx, endx))
                startx = -1

    min_hProjectThreshold = width * 0.4  # 黑鍵水平投影最小閥值
    max_hProjectThreshold = width * 0.7  # 黑鍵水平投影最大閥值
    hProjectWidth = height * 0.3  # 黑鍵水平投影連續寬度
    yroi = (ysum > min_hProjectThreshold) & (ysum < max_hProjectThreshold)
    starty = -1
    yseg = []
    for i in range(height):
        if yroi[i]:
            if starty == -1:
                starty = i
        else:
            endy = i
            if starty != -1:
                if (endy - starty) > hProjectWidth:
                    print("starty: {}, endy: {}\n".format(starty, endy))
                    yseg.append((starty, endy))
                starty = -1
    blackKeys = []
    starty, endy = yseg[0]
    for i in range(len(xseg)):
        startx, endx = xseg[i]
        blackKeys.append((startx, starty, endx, endy))
    blackKeys.sort(key=lambda s: s[0])
    return blackKeys


# Webcam input
cap = cv.VideoCapture(1, cv.CAP_DSHOW)
# cap = cv.VideoCapture("D:/Git/CV_Course_110/piano/images/ms_pad%02d.jpg")
# cap = cv.VideoCapture("C:/Users/cwt/Documents/GitHub/CV_Course_110/piano/images/ms_pad%02d.jpg")
# cap = cv.VideoCapture("video.jpg")

winName = "Virtual Piano"

while (cap.isOpened()):
    try:
        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            break
        cv.imshow("Original", frame)
        cv.waitKey(1)
        # Load the dictionary that was used to generate the markers.
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_1000)

        # Initialize the detector parameters using default values
        parameters = cv.aruco.DetectorParameters_create()

        # Detect the markers in the image to decide the area of Piano keys
        markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        if len(markerIds) != 5:
            print("ID not all detected")
            continue
        # marker ID0 角點0為固定參考座標點
        index = np.squeeze(np.where(markerIds == 0))
        refPoint = np.squeeze(markerCorners[index[0]])[0]

        padCorners = [];
        # marker ID1 角點1為螢幕左上角位置
        index = np.squeeze(np.where(markerIds == 1))
        leftTop = np.squeeze(markerCorners[index[0]])[1]
        padCorners.append(leftTop)

        # marker ID2 角點0為螢幕右上角位置
        index = np.squeeze(np.where(markerIds == 2))
        rightTop = np.squeeze(markerCorners[index[0]])[0]
        padCorners.append(rightTop)

        # marker ID3 角點3為螢幕右下角位置
        index = np.squeeze(np.where(markerIds == 3))
        rightBottom = np.squeeze(markerCorners[index[0]])[3]
        padCorners.append(rightBottom)

        # marker ID4 角點2為螢幕左下角位置
        index = np.squeeze(np.where(markerIds == 4))
        leftBottom = np.squeeze(markerCorners[index[0]])[2]
        padCorners.append(leftBottom)

        cv.aruco.drawDetectedMarkers(frame, markerCorners)

        # 根據螢幕左上角及右上角位置來判斷平板擺放之傾斜角度
        x1 = leftTop[0]
        y1 = leftTop[1]
        x2 = rightTop[0]
        y2 = rightTop[1]
        if x2 - x1 == 0:
            print("vertical line")
            angle = 90
        elif y2 - y1 == 0:
            print("horizontal line")
            angle = 0
        else:
            k = -(y2 - y1) / (x2 - x1)
            angle = np.arctan(k) * 57.29577
            print("angle={}".format(angle))

        # 根據傾斜角度將擷取影像轉正，以利後續使用水平及垂直投影法偵測黑白鍵
        (h, w) = frame.shape[:2]
        (cX, cY) = (refPoint[0], refPoint[1])
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)  # 轉正影像所需轉換矩陣
        M_INV = cv.getRotationMatrix2D((cX, cY), angle, 1.0)  # 逆轉換座標所需轉換矩陣
        alignedImg = cv.warpAffine(frame, M, (w, h))
        print(cX,cY)# 獲得轉正影像

        padCornersRotated = []  # 用來儲存轉正後之螢幕角點位置
        for corner in padCorners:
            point = np.array([corner[0], corner[1], 1])
            pointRoated = M.dot(point)
            print(pointRoated.astype(int))
            padCornersRotated.append(pointRoated.astype(int))

        if padCornersRotated[0][0] > padCornersRotated[3][0]:
            padScreenLeft = padCornersRotated[0][0] + 20   #平板內縮 + 20
        else:
            padScreenLeft = padCornersRotated[3][0]
            padScreenLeft = padCornersRotated[0][0] + 20   #平板內縮 + 20

        if padCornersRotated[1][0] > padCornersRotated[2][0]:
            padScreenRight = padCornersRotated[2][0] - 5  #平板內縮 - 10
        else:
            padScreenRight = padCornersRotated[1][0] -5   #平板內縮 - 10

        if padCornersRotated[0][1] > padCornersRotated[1][1]:
            padScreenTop = padCornersRotated[0][1]
        else:
            padScreenTop = padCornersRotated[1][1]

        if padCornersRotated[2][1] > padCornersRotated[3][1]:
            padScreenBottom = padCornersRotated[3][1]
        else:
            padScreenBottom = padCornersRotated[2][1]
        pixelWidth_mm = screenWidth_mm / (padScreenRight - padScreenLeft)
        pixelHeight_mm = screenHeight_mm / (padScreenBottom - padScreenTop)
        print("pixelWidth_mm=%s"  % pixelWidth_mm)
        print("pixelHeight_mm=%s" % pixelHeight_mm)
        # 擷取轉正後僅包含螢幕範圍之影像
        alignedScreenImg = alignedImg[padScreenTop:padScreenBottom, padScreenLeft + 5:padScreenRight - 5]
        cv.imshow("Aligned Screen", alignedScreenImg)

        dst = alignedScreenImg.copy()
        gray = cv.cvtColor(alignedScreenImg, cv.COLOR_BGR2GRAY)
       # cv.imshow('gray', gray)

        # 從轉正後的螢幕範圍灰階影像偵測黑鍵位置
        blackKeys = findBlackKeys(gray)
        if len(blackKeys) == 0:
            continue
        offset = (0, 0)  # 包含黑鍵範圍的黑鍵範圍位移值
        showKeyBox(dst, blackKeys, offset)
        offset = (padScreenLeft + 5, padScreenTop)  # 螢幕黑鍵範圍位移值
        blackKeysCenters, rotatedBlackBoxes = transformKeyBoxPoints(blackKeys, M_INV, offset)  # 計算逆轉換後黑鍵於原圖之座標
        drawRoatedBoxes(frame, rotatedBlackBoxes, blackKeysCenters)
        print("共找到%s個黑鍵"%len(blackKeysCenters))
        print("黑鍵座標1%s" % blackKeysCenters)
        B = np.array([0,1])*20
        blackKeysCenters = blackKeysCenters+B#黑鍵會點不到，故加此行
        print("黑鍵座標2%s" % blackKeysCenters)


        # 找出轉正後的螢幕範圍內黑鍵y軸的最下面位置
        blackKeyBottom_Y = 0;
        for i in range(len(blackKeys)):
            (x1, y1, x2, y2) = blackKeys[i]
            if y2 > blackKeyBottom_Y:
                blackKeyBottom_Y = y2
        print("blackKeyBottom_Y{}".format(blackKeyBottom_Y))

        # whiteKeyAreaImg不包含黑鍵範圍的鍵盤區域
        whiteKeyAreaImg = gray[blackKeyBottom_Y + 5: alignedScreenImg.shape[0], 0:alignedScreenImg.shape[1]]
       # cv.imshow('whiteKeyArea', whiteKeyAreaImg)

        # 從不包含黑鍵範圍的鍵盤區域搜尋白鍵
        whiteKeys = findWhiteKeys(whiteKeyAreaImg)
        offset = (0, blackKeyBottom_Y + 5)  # 包含黑鍵範圍的白鍵範圍位移值
        showKeyBox(dst, whiteKeys, offset)
        offset = (padScreenLeft + 5, padScreenTop + blackKeyBottom_Y + 5)  # 螢幕白鍵範圍位移值
        whiteKeysCenters, rotatedWhiteBoxes = transformKeyBoxPoints(whiteKeys, M_INV, offset)  # 計算逆轉換後白鍵於原圖之座標
        drawRoatedBoxes(frame, rotatedWhiteBoxes, whiteKeysCenters)
        print("共找到%s個白鍵" % len(whiteKeysCenters))
       # cv.imshow('Piano Result', dst)

        cv.imshow(winName, frame)

        if cv.waitKey(100) == ord('s'):
             break

    except Exception as inst:
        print(inst)


choose_song = int(input("請輸入要彈奏的歌曲:\n(1)小星星\n(2)小蜜蜂\n(3)聖誕歌\n(4)布穀鳥\n(5)甜蜜蜜\n(6)童話\n"))
play_song.go_press(whiteKeysCenters,blackKeysCenters,choose_song,(cX,cY))


cap.release()
cv.destroyAllWindows()
