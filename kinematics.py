import numpy as np
import phx
from rbx_toolkit import rbx_toolkit as rbx


# Link Lengths
a1 = 5
a2 = 7
a3 = 7
a4 = 7
a5 = 9
link_lengths = np.array([a1, a2, a3, a4,a5])

def DH(a, alpha, d, cta):
    ans = np.array([
        [np.cos(cta), -np.sin(cta)*np.cos(alpha), np.sin(cta)*np.sin(alpha), a*np.cos(cta)],
        [np.sin(cta), np.cos(cta)*np.cos(alpha), -np.cos(cta)*np.sin(alpha), a*np.sin(cta)],
        [      0    ,         np.sin(alpha)    ,      np.cos(alpha)     ,          d      ],
        [      0    ,                 0        ,            0           ,          1      ]  ] ,dtype=object)

    return ans

def htm3(theta):
    """Input: 1x3 angle array  Returns: 4x4 htm matrix"""
    # h0_1
    r0_1 = np.dot(rbx.rot_x(90), rbx.rot_y(theta[0]))
    #print("r0_1=%s#" % r0_1)
    d0_1 = rbx.transl(0, 0, a1)
    #print("d0_1=%s#" %d0_1)
    h0_1 = rbx.htm(r0_1, d0_1)
    #print("h0_1=%s#" % h0_1)
    # h1_2
    r1_2 = rbx.rot_z(theta[1])
    x1_2 = a2*np.cos(np.radians(theta[1]))
    y1_2 = a2*np.sin(np.radians(theta[1]))
    z1_2 = 0
    d1_2 = rbx.transl(x1_2, y1_2, z1_2)
    h1_2 = rbx.htm(r1_2, d1_2)
    #print("d1_2=%s#" % d1_2)
    #print("h1_2=%s#" % h1_2)
    # h2_3
    r2_3 = rbx.rot_z(theta[2])
    x2_3 = a3*np.cos(np.radians(theta[2]))
    y2_3 = a3*np.sin(np.radians(theta[2]))
    z2_3 = 0
    d2_3 = rbx.transl(x2_3, y2_3, z2_3)
    h2_3 = rbx.htm(r2_3, d2_3)
    #print("d2_3=%s#" % d2_3)
    #print("h2_3=%s#" % h2_3)
    # h3_4
    r3_4 = rbx.rot_z(theta[3])
    x3_4 = a4 * np.cos(np.radians(theta[3]))
    y3_4 = a4 * np.sin(np.radians(theta[3]))
    z3_4 = 0
    d3_4 = rbx.transl(x3_4, y3_4, z3_4)
    h3_4 = rbx.htm(r3_4, d3_4)
    #print("d3_4=%s#" % d3_4)
    #print("h3_4=%s#" % h3_4)

    # h0_4
    h0_2 = np.dot(h0_1, h1_2)
    h0_3 = np.dot(h0_2, h2_3)
    h0_4 = np.dot(h0_3, h3_4)

    #print("h0_4=%s#" % h0_4)
    return h0_4


def htm4(theta):
    """ Input: 1x5 angle array  Returns: 4x4 htm matrix"""
    h0_4 = htm3(theta)

    # h4_5
    r4_5 = rbx.rot_z(theta[4])
    x4_5 = a5 * np.cos(np.radians(theta[4]))
    y4_5 = a5 * np.sin(np.radians(theta[4]))
    z4_5 = 0
    d4_5 = rbx.transl(x4_5, y4_5, z4_5)
    h4_5 = rbx.htm(r4_5, d4_5)
    h0_5 = np.dot(h0_4, h4_5)
    #print("h0_5=%s#" % h0_5)
    return h0_5



def fk4(theta):
    """Input: 1x4 angle array  Returns: 1x3 position array"""
    h0_5 = htm4(theta)
    x0_5 = h0_5[0, 3]
    #print("x0_5=%s" % x0_5)
    y0_5 = h0_5[1, 3]
    #print("y0_5=%s" % y0_5)
    z0_5 = h0_5[2, 3]
    #print("z0_5=%s" % z0_5)
    d0_5 = [x0_5, y0_5, z0_5]
    return d0_5


def ik5(xyz_array):
    #xyz_array = np.array([[0], [15], [0]])  # 目標點

    print()
    """ 第一顆另外算，幾何法  """
    theta_1 = np.arctan2(xyz_array[1], xyz_array[0])
    """ 定義參數 關節數+DH參數"""
    JOINT_SIZE = 4 + 1
    a = np.transpose([[7, 7, 7, 7.5]])
    alpha = np.transpose([[0, 0, 0, 0]]) * np.pi / 180
    d = np.transpose([[0, 0, 0, 0]])
    cta = np.transpose([[0, 0, 0, 0]]) * np.pi / 180
    """ 計算正運動學"""
    T = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * 5, dtype=object)
    P = np.array([[0, 0, 0]] * 5, dtype=object)
    R = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 5, dtype=object)

    for k in range(1, JOINT_SIZE):
        T[k] = np.dot(T[k - 1], np.asarray(DH(a[k - 1], alpha[k - 1], d[k - 1], cta[k - 1]), dtype=float))
        P[k][0] = T[k][0][3]
        P[k][1] = T[k][1][3]
        P[k][2] = T[k][2][3]
        R[k] = T[k][0:3, 0:3]
    """ 迭代法求逆解"""
    #b = []
    #for i in range(len(xyz_array)):
    #    b.append(float(sum(xyz_array[i])))
    #xyz_array = np.asarray(b, dtype=float)
    target = np.array([[np.hypot(xyz_array[0], xyz_array[1])], [xyz_array[2] - 5], [-77 * np.pi / 180.0]])  # 目標點
    #print("x = %sxxx" % np.hypot(xyz_array[0], xyz_array[1]))
    #print("y = %syyy" % [xyz_array[2] - 5])
    PATH_SIZE = 20
    save_cta = np.zeros((4, PATH_SIZE))
    cta_temp = np.array([0, 0, 0, 0])
    for i in range(0, PATH_SIZE):
        save_cta[0, i] = cta[0]
        save_cta[1, i] = cta[1]
        save_cta[2, i] = cta[2]
        save_cta[3, i] = cta[3]

        # 誤差
        error = np.array([target[0] - P[JOINT_SIZE - 1][0], target[1] - P[JOINT_SIZE - 1][1],
                          target[2] - (cta[0] + cta[1] + cta[2] + cta[3])])

        Jacob0 = [[[-a[3] * np.sin(cta[0] + cta[1] + cta[2] + cta[3]) - a[1] * np.sin(cta[0] + cta[1]) - a[0] * np.sin(
            cta[0]) - a[2] * np.sin(cta[0] + cta[1] + cta[2])], [
                       -a[3] * np.sin(cta[0] + cta[1] + cta[2] + cta[3]) - a[1] * np.sin(cta[0] + cta[1]) - a[
                           2] * np.sin(cta[0] + cta[1] + cta[2])],
                   [-a[3] * np.sin(cta[0] + cta[1] + cta[2] + cta[3]) - a[2] * np.sin(cta[0] + cta[1] + cta[2])],
                   [-a[3] * np.sin(cta[0] + cta[1] + cta[2] + cta[3])]],
                  [[a[3] * np.cos(cta[0] + cta[1] + cta[2] + cta[3]) + a[1] * np.cos(cta[0] + cta[1]) + a[0] * np.cos(
                      cta[0]) + a[2] * np.cos(cta[0] + cta[1] + cta[2])], [
                       a[3] * np.cos(cta[0] + cta[1] + cta[2] + cta[3]) + a[1] * np.cos(cta[0] + cta[1]) + a[
                           2] * np.sin(cta[0] + cta[1] + cta[2])],
                   [a[3] * np.cos(cta[0] + cta[1] + cta[2] + cta[3]) + a[2] * np.cos(cta[0] + cta[1] + cta[2])],
                   [a[3] * np.cos(cta[0] + cta[1] + cta[2] + cta[3])]],
                  [[1], [1], [1], [1]]
                  ]


        Jacob0 = np.asfarray([[*map(sum, e)] for e in list(Jacob0)], dtype=float)

        # cta = cta + np.dot(np.linalg.pinv(Jacob0),error)
        cta_temp = np.dot(np.linalg.pinv(Jacob0), error)
        #cta_temp = np.dot(np.linalg.pinv(Jacob0), error)
        cta[0] = cta[0] + cta_temp[0]
        cta[1] = cta[1] + cta_temp[1]
        cta[2] = cta[2] + cta_temp[2]
        cta[3] = cta[3] + cta_temp[3]
        if cta[0] < -92 * np.pi / 180.0:
            cta[0] = -92 * np.pi / 180.0
        if cta[0] > 143 * np.pi / 180.0:
            cta[0] = 143 * np.pi / 180.0
        if cta[1] < -92 * np.pi / 180.0:
            cta[1] = -92 * np.pi / 180.0
        if cta[1] >85 * np.pi / 180.0:
            cta[1] = 85 * np.pi / 180.0
        if cta[2] < -92 * np.pi / 180.0:
            cta[2] = -92 * np.pi / 180.0
        if cta[2] > 85 * np.pi / 180.0:
            cta[2] = 85 * np.pi / 180.0
        if cta[3] < -92 * np.pi / 180.0:
            cta[3] = -92 * np.pi / 180.0
        if cta[3] > 85 * np.pi / 180.0:
            cta[3] = 85 * np.pi / 180.0
        for k in range(1, JOINT_SIZE):
            T[k] = np.dot(T[k - 1], np.asarray(DH(a[k - 1], alpha[k - 1], d[k - 1], cta[k - 1]), dtype=float))

            P[k][0] = T[k][0][3]
            P[k][1] = T[k][1][3]
            P[k][2] = T[k][2][3]
            R[k] = T[k][0:3, 0:3]



    b = []
    for i in range(len(cta)):
        b.append(cta[i])

    cta = np.asarray(b, dtype=float)
    #print("角度1 = %s" % (theta_1 * 180 / np.pi))
    #print("角度2 = %s" % (cta[0] * 180 / np.pi))
    #print("角度3 = %s" % (cta[1] * 180 / np.pi))
    #print("角度4 = %s" % (cta[2] * 180 / np.pi))
    #print("角度5 = %s" % (cta[3] * 180 / np.pi))
    theta0_5 = np.array([(theta_1 * 180 / np.pi), (cta[0] * 180 / np.pi), (cta[1] * 180 / np.pi), (cta[2] * 180 / np.pi), (cta[3] * 180 / np.pi)],dtype=float)
    return theta0_5


def calculate_theta_4(joint_angles, theta0_4):
    # R0_3
    theta_1 = joint_angles[0]
    theta_2 = joint_angles[1]
    theta_3 = joint_angles[2]
    # R0_4
    R0_4a = np.dot(rbx.rot_z(theta_1), rbx.rot_x(90))
    R0_4 = np.dot(R0_4a, rbx.rot_z(theta0_4))
    R0_1 = np.dot(rbx.rot_x(90), rbx.rot_y(theta_1))
    R1_2 = rbx.rot_z(theta_2)
    R2_3 = rbx.rot_z(theta_3)
    R0_2 = np.dot(R0_1, R1_2)
    R0_3 = np.dot(R0_2, R2_3)
    # R3_4
    R3_4 = np.dot(np.transpose(R0_3), R0_4)
    # theta_4
    theta_4 = np.degrees(np.arcsin(R3_4[1, 0]))
    return theta_4


def interpolate_line(p_start, p_end, inter_size):
    xyz_matrix = np.zeros([inter_size, 3])  # to store results of interpolated data
    a_vector = p_end - p_start
    t = np.linspace(0, 1, inter_size)
    for index in range(0, inter_size):
        xyz_matrix[index] = p_start + np.multiply(t[index], a_vector)
    return xyz_matrix


def create_joint_matrix(xyz_matrix):
    ik_matrix = np.zeros([xyz_matrix.shape[0], 4])  # to store results of ik3
    for row_num in range(0, xyz_matrix.shape[0]):
        ik_matrix[row_num] = calculate_theta_4(xyz_matrix[row_num], 0)
    return ik_matrix


def line_demo():
    p_start = np.array([-15, -15,  5])
    p_end = np.array([20, 0,  15])

    inter_size = 60
    r_matrix = interpolate_line(p_start, p_end, inter_size)
    29(r_matrix)

    ik_matrix = create_joint_matrix(r_matrix)
    ik_matrix = np.rint(ik_matrix)
    print(ik_matrix)
    phx.set_wsew(ik_matrix[0])
    phx.wait_for_completion()

    for positions in range(0, inter_size):
        phx.set_wsew(ik_matrix[positions])