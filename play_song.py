import interpolation_demo as press
import phx
import transfercoordinate as trans_coor


def go_press(whiteKeysCenters, blackKeysCenters, choose_song, cX_Y):
    white_list = ['Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#', 'La#', 'Si#']
    black_list = ['C#', 'D#', 'F#', 'G#', 'A#', 'C#2', 'D#2', 'F#2', 'G#2', 'A#2']
    Relative_coordinates_aruco0_white, Relative_coordinates_aruco0_black = trans_coor.machine_coordinate(
        whiteKeysCenters, blackKeysCenters, cX_Y)

    if choose_song == 1:
        """小星星"""
        play_song = ['Do', 'Do', 'So', 'So', 'La', 'La', 'So', 'Fa', 'Fa', 'Mi', 'Mi', 'Re', 'Re', 'Do']
    elif choose_song == 2:
        """小蜜蜂"""
        play_song = ['Re#', 'Si', 'Si', 'Do#', 'La', 'La', 'So', 'La', 'Si', 'Do#', 'Re#', 'Re#', 'Re#']
    elif choose_song == 3:
        """彈聖誕歌曲"""
        play_song = ['Re', 'So', 'So', 'La', 'So', 'F#', 'Mi', 'Mi', 'Mi', 'La', 'La', 'Si', 'La', 'So', 'F#', 'Re',
                     'Re']  # We wish
    elif choose_song == 4:
        """彈布穀鳥歌曲"""
        play_song = ['Do#', 'La', 'Do#', 'La', 'So', 'Fa', 'So', 'Fa', 'So', 'So', 'La', 'A#', 'So', 'La', 'La', 'A#',
                     'Do#', 'La', 'Do#', 'La', 'Do#', 'La', 'A#', 'La', 'Do#', 'Fa']
    elif choose_song == 5:
        """彈甜蜜蜜"""
        play_song = ['Mi#', 'So#', 'Mi#', 'Do#', 'Re#', 'Do#', 'Re#', 'Mi#', 'Mi#', 'Re#', 'Re#', 'Re#', 'Re#', 'Re#',
                     'Do#', 'La', 'So', 'Do#']
    elif choose_song == 6:
        """童話"""
        play_song = ['So', 'Do#', 'Si', 'Do#', 'So', 'So', 'Do#', 'Si', 'Do#', 'So', 'So', 'Do#', 'Si', 'Do#', 'Do#',
                     'Do#', 'La', 'La', 'So']
    elif choose_song == 7:
        """白+黑鍵全彈一次"""
        play_song = ['Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#', 'La#', 'Si#',
                     'C#', 'D#', 'F#', 'G#', 'A#', 'C#2', 'D#2', 'F#2', 'G#2', 'A#2']
    elif choose_song == 8:
        """只彈所有白鍵"""
        play_song = ['Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si', 'Do#', 'Re#', 'Mi#', 'Fa#', 'So#', 'La#', 'Si#']
    elif choose_song == 9:
        """只彈所有黑鍵"""
        play_song = ['C#', 'D#', 'F#', 'G#', 'A#', 'C#2', 'D#2', 'F#2', 'G#2', 'A#2']
    elif choose_song == 10:
        """所有"""
        play_song = ['Do','C#', 'Re', 'D#', 'Mi', 'Fa','F#' 'So','G#' 'La','A#', 'Si','C#2', 'Do#','D#2', 'Re#','F#2', 'Mi#','G#2' 'Fa#', 'A#2' 'So#', 'La#', 'Si#']

    for i in range(len(play_song)):
        for j in range(len(white_list)):
            if play_song[i] == white_list[j]:
                press.line_demo(5, [Relative_coordinates_aruco0_white[j][0], Relative_coordinates_aruco0_white[j][1]])
                phx.wait_for_completion()
                print("彈奏" + white_list[j])
                break
        for k in range(len(black_list)):
            if play_song[i] == black_list[k]:
                press.line_demo(5, [Relative_coordinates_aruco0_black[k][0], Relative_coordinates_aruco0_black[k][1]])
                phx.wait_for_completion()
                print("彈奏" + black_list[k])
                break
    return
