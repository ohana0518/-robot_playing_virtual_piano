"""影像座標轉換為機械手臂座標"""
"""
實際拍攝到影像的長 : 41.4(cm)
實際拍攝到影像的寬 : 31.55(cm)

拍攝影像像素點(640 * 480)(px)
"""
screenWidth_cm = 41.4  # 螢幕寬度實際尺寸
screenHeight_cm = 31.55  # 螢幕高度實際尺寸
import numpy as np

def machine_coordinate(whiteKeysCenters,blackKeysCenters,cX_Y):
    print("參考點座標:")
    print(cX_Y)
    print("白鍵座標:%s:" % whiteKeysCenters)
    print("黑鍵座標:%s:" % blackKeysCenters)
    cX_Y_np = np.array(cX_Y)
    whiteKeysCenters_np = np.array(whiteKeysCenters)
    blackKeysCenters_np = np.array(blackKeysCenters)
    inv_data = np.array([1, -1])
    pixel_data = np.array([screenWidth_cm / 640, screenHeight_cm / 480])
    distance_arm_to_aruco0 = np.array([15, 8])  # 15 10  #機械手臂與參考點(aruco0)距離


    Relative_coordinates_aruco0_white = ((whiteKeysCenters_np - cX_Y) * inv_data * pixel_data)
    Relative_coordinates_aruco0_black = ((blackKeysCenters_np - cX_Y) * inv_data * pixel_data)
    print(" 相對參考點實際距離(cm):")

    print(Relative_coordinates_aruco0_white)
    print(Relative_coordinates_aruco0_black)
    print(" 相對機械手臂實際距離(cm):")
    Relative_coordinates_aruco0_white = ((whiteKeysCenters_np - cX_Y) * inv_data * pixel_data) + distance_arm_to_aruco0
    Relative_coordinates_aruco0_black = ((blackKeysCenters_np - cX_Y) * inv_data * pixel_data) + distance_arm_to_aruco0
    print(Relative_coordinates_aruco0_white)
    print(Relative_coordinates_aruco0_black)
    print(" 機械手臂座標:")
    world_data = np.array([1, -1])
    print(Relative_coordinates_aruco0_white)
    print(Relative_coordinates_aruco0_black)
    print(" 世界座標轉換:")
    Relative_coordinates_aruco0_white[:, [0, -1]] = Relative_coordinates_aruco0_white[:, [-1, 0]] * inv_data
    Relative_coordinates_aruco0_black[:, [0, -1]] = Relative_coordinates_aruco0_black[:, [-1, 0]] * inv_data
    print(Relative_coordinates_aruco0_white)
    print(Relative_coordinates_aruco0_black)

    return Relative_coordinates_aruco0_white,Relative_coordinates_aruco0_black





# a = np.array([[1,2],[5,6],[8,9],[10,11]])
# print(a)
#
#
#
#
# a[:,[0, -1]] = a[:,[-1, 0]]
# print(a)


