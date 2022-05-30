import kinematics as kin
import numpy as np
import phx
import time

phx.turn_on()
phx.rest_position()
# calculations
def go_to_pos(pickup_pos, theta0_4):
    joint_angles = kin.ik3(pickup_pos)
    theta4 = kin.calculate_theta_4(joint_angles, theta0_4)
    phx.set_wse(joint_angles)
    phx.set_wrist(theta4)
    phx.wait_for_completion()


def go_to_a():
    point_a = [-20, -10, 7]
    theta0_4 = 0
    go_to_pos(point_a, theta0_4)
    phx.wait_for_completion()


def go_to_b():
    point_b = [10, 0,  20]
    theta0_4 = 0
    go_to_pos(point_b, theta0_4)
    phx.wait_for_completion()


def interpolate_line(p_start, p_end, inter_size):
    xyz_matrix = np.zeros([inter_size, 3])  # to store results of interpolated data
    #print("one xyz_matrix = %s one##" % xyz_matrix)
    a_vector = p_end - p_start
    t = np.linspace(0, 1, inter_size)
    for index in range(0, inter_size):
        xyz_matrix[index] = p_start + np.multiply(t[index], a_vector)
    #print("two xyz_matrix = %s two##"  % xyz_matrix)
    return xyz_matrix


def create_joint_matrix(xyz_matrix):
    ik_matrix = np.zeros([xyz_matrix.shape[0], 5])  # to store results of ik3
    for row_num in range(0, xyz_matrix.shape[0]):
        ik3_matrix = kin.ik5(xyz_matrix[row_num])
        print(ik3_matrix)
        ik_matrix[row_num] = [ik3_matrix[0], ik3_matrix[1], ik3_matrix[2], ik3_matrix[3], ik3_matrix[4]]

    return ik_matrix


def line_demo(inter_size, xy_new_coordinate):
    p_start = np.array([xy_new_coordinate[0], xy_new_coordinate[1],  4.8])
    p_end = np.array([xy_new_coordinate[0], xy_new_coordinate[1],  1.3])
    #p_start = np.array([20, 0,  0])
    #p_end = np.array([10, 0,  0])
    r_matrix = interpolate_line(p_start, p_end, inter_size)  
    print(r_matrix)

    ik_matrix = create_joint_matrix(r_matrix)
    ik_matrix = np.rint(ik_matrix)

    phx.set_wsew(ik_matrix[0])
    phx.wait_for_completion()
    for positions in range(0, inter_size):
        phx.set_wsew(ik_matrix[positions])

    p_start = np.array([xy_new_coordinate[0], xy_new_coordinate[1], 4.5])
    p_end = np.array([xy_new_coordinate[0], xy_new_coordinate[1], 4.5])
    # p_start = np.array([20, 0,  0])
    # p_end = np.array([10, 0,  0])
    r_matrix = interpolate_line(p_start, p_end, inter_size)
    print(r_matrix)

    ik_matrix = create_joint_matrix(r_matrix)
    ik_matrix = np.rint(ik_matrix)
    phx.set_wsew(ik_matrix[0])
    phx.wait_for_completion()

    return
