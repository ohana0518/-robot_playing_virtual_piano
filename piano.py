"""此為主程式"""
import cv2
# import  interpolation_demo as press
# import  presess_white_img as pre_img
# import  phx
import black_key
import white_key
import transfercoordinate as trans_coor
import  drawtext
# phx.turn_on()
# phx.rest_position()
tone_list = ['La^', 'Si^', 'Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#']
black_list = ['D#', 'F#', 'G#', 'A#', 'C', 'D', 'F', 'G', 'A','123']
# ori = cv2.imread("D:\\piano_pic\\test_910.jpg", cv2.IMREAD_COLOR)
ori = cv2.imread("output4.jpg", cv2.IMREAD_COLOR)
# current camera
#----------------------------------------------------------------------------
# cap = cv2.VideoCapture(1)
# while(True):
#     # 擷取影像
#     ret, ori = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#         # 顯示圖片
#     cv2.imshow('live', ori)
#     # 按下 q 鍵離開迴圈
#     if cv2.waitKey(1) == ord('s'):
#         break
# # 釋放該攝影機裝置
# cap.release()
# cv2.destroyAllWindows()
#---------------------------------------------------------------------
ori = cv2.resize(ori, (256, 192), interpolation=cv2.INTER_AREA)
# xy_white_old_coordinate = pre_img.findwhitekey(pre_img.preprocessing_white(ori),  tone_list)
xy_white_old_coordinate = white_key.return_white_coordinate(ori, tone_list)
xy_black_old_coordinate = black_key.return_black_coordinate(ori, black_list)
# xy_white_new_coordinate = trans_coor.machine_coordinate(tone_list,xy_white_old_coordinate)
xy_black_new_coordinate = trans_coor.machine_coordinate(black_list,xy_black_old_coordinate)

result1 = drawtext.draw_tone(tone_list, ori, xy_white_old_coordinate) #"""result1為白鍵"""
result2 = drawtext.draw_tone(black_list, ori, xy_black_old_coordinate)#"""result2為黑鍵"""
cv2.namedWindow("result1", cv2.WINDOW_NORMAL)
cv2.namedWindow("result2", cv2.WINDOW_NORMAL)
cv2.imshow("result1",result1)
cv2.imshow("result2",result2)
# print("xy_new_coordinate =", xy_new_coordinate)
# print("black_new_coordinate =", black_new_coordinate)
"""小蜜蜂"""
# play_song = ['Re#', 'Si', 'Si', 'Do#', 'La', 'La', 'So','La','Si','Do#','Re#','Re#','Re#']
"""白+黑鍵全彈一次"""
# play_song = ['La^', 'Si^', 'Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#','D#', 'F#', 'G#', 'A#', 'C', 'D', 'F', 'G', 'A']
"""只彈所有白鍵"""
play_song = ['La^', 'Si^', 'Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#']
"""待測"""
# play_song = ['Do', 'Do', 'So', 'So', 'La', 'La', 'So','Fa','Fa','Mi','Mi','Re','Re','Do','Si','Si','Do#','Si','La','So','Mi','Re','Re','Mi','La','A#','So']
# play_song = ['D' ,'Do#', 'La','So','Fa','G#','Re',]
# play_song =['Re','So','So','La','So','A#','Mi','Mi','Mi','La','La','Si','La','So','A#','Re','Re']#We wish
# play_song=['La^']
# play_song = ['La^', 'Do']
# for i in range(len(play_song)):
#     for j in range(len(tone_list)):
#         if play_song[i] == tone_list[j]:
#             press.line_demo(6,[xy_white_new_coordinate[j][0] ,xy_white_new_coordinate[j][1]])
#             phx.wait_for_completion()
#             print("彈奏"+tone_list[j])
#             break
#     for k in range(len(black_list)):
#         if play_song[i] == black_list[k]:
#             press.line_demo(6,[xy_black_new_coordinate[k][0] ,xy_black_new_coordinate[k][1]])
#             phx.wait_for_completion()
#             print("彈奏"+black_list[k])
#             break
cv2.waitKey(0)
