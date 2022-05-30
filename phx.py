from dxl_control.Ax12 import Ax12
import time

# motor objects
all_motors = Ax12(254)
waist = Ax12(1)
shoulder = Ax12(2)
elbow = Ax12(3)
wrist = Ax12(4)
gripper = Ax12(5)

# default_speed = 75
# max_speed = 1023
# half_speed = 512
# default_speed =200
# max_speed = 1200
# half_speed = 600
default_speed = int(200* 1.4)
max_speed     = int(1200*1.4)
half_speed    = int(600* 1.4)
def map_val(x_in, x_min, x_max, y_min, y_max):
    """Linearly maps x to y; returns corresponding y value"""
    m = ((y_max - y_min) / (x_max - x_min))
    y_out = m * (x_in - x_min) + y_min
    #print("map_val y_out = %s#"%y_out)
    return y_out


def deg_to_pos(obj, deg):
    pos = map_val(deg, obj.min_map_deg, obj.max_map_deg,
                  obj.MIN_POS_VAL, obj.MAX_POS_VAL)
    return pos


def set_map_range(motor_object, min_angle, max_angle):
    motor_object.min_map_deg = min_angle
    motor_object.max_map_deg = max_angle


def set_deg_limit(motor_object, min_limit, max_limit):
    motor_object.min_deg_limit = min_limit
    motor_object.max_deg_limit = max_limit


def check_limit(motor_object, deg):
    if deg > motor_object.max_deg_limit:
        return motor_object.max_deg_limit
    elif deg < motor_object.min_deg_limit:
        return motor_object.min_deg_limit
    else:
        return deg


def config_motor_angles():
    set_map_range(waist, -150, 150)
    set_deg_limit(waist, -150, 150)
    set_map_range(shoulder, -150, 150)
    set_deg_limit(shoulder, -150, 150)
    set_map_range(elbow, -150, 150)
    set_deg_limit(elbow, -150, 150)
    set_map_range(wrist, -150, 150)
    set_deg_limit(wrist, -150, 150)
    set_map_range(gripper, -150, 150)
    set_deg_limit(gripper, -150, 150)
 #m = ((MAX_POS_VAL - MIN_POS_VAL) / (max_map_deg - min_map_deg))  300/
  #  y_out = m * (deg - min_map_deg) + MIN_POS_VAL


def set_waist(deg):
    deg = check_limit(waist, deg)
    pos = int(deg_to_pos(waist, deg))
    #print("waist_deg = %s" % deg)
    #print("waist_pos = %s" % pos)
    waist.set_position(pos)


# check to make sure order motors are correct
def set_shoulder(deg):
    deg = check_limit(shoulder, deg)
    pos = int(deg_to_pos(shoulder, deg))
    #print("shoulder_deg = %s" % deg)
    #print("shoulder_pos = %s" % pos)
    shoulder.set_position(pos)



# check to make sure order motors are correct
def set_elbow(deg):
    deg = check_limit(elbow, deg)
    pos = int(deg_to_pos(elbow, deg))
    #print("elbow_deg = %s" % deg)
    #print("elbow_pos = %s" % pos)
    elbow.set_position(pos)


def set_wrist(deg):
    deg = check_limit(wrist, deg)
    pos = int(deg_to_pos(wrist, deg))
    #print("wrist_deg = %s" % deg)
    #print("wrist_pos = %s" % pos)
    wrist.set_position(pos)


def set_gripper(deg):
    deg = check_limit(gripper, deg)
    pos = int(deg_to_pos(wrist, deg))
    gripper.set_position(pos)
    #print("gripper_pos = %s" % pos)


def close_gripper():
    gripper.set_position(0)


def open_gripper():
    gripper.set_position(204)


def set_angle(self, input_deg):
    """Sets motor to specified input angle."""
    dxl_goal_position = int(map_val(
        input_deg, self.min_angle, self.max_angle, self.MIN_POS_VAL, self.MAX_POS_VAL))
    self.set_position(dxl_goal_position)


def get_angle(self):
    """Returns present angle."""
    dxl_present_position = self.get_position()
    dxl_angle = int(map_val(dxl_present_position, self.MIN_POS_VAL, self.MAX_POS_VAL, self.min_angle, self.max_angle))
    return dxl_angle


def wait_for_completion():
    while waist.is_moving():
            pass
    while shoulder.is_moving():
            pass
    while elbow.is_moving():
            pass
    while wrist.is_moving():
            pass
    while gripper.is_moving():
            pass


def set_wse(joint_angles):
    """Set the first 3 joints: waist, shoulder, elbow."""
    set_waist(joint_angles[0])
    set_shoulder(joint_angles[1])
    set_elbow(joint_angles[2])
    set_wrist(joint_angles[3])



def set_wsew(joint_angles):
    """Sets the 4 joints: waist, shoulder, elbow and wrist"""
    set_wse(joint_angles)
    set_gripper(joint_angles[4])


def rest_position():
    """Rest Position of Phx."""
    joint_angles = [0.0000, 90, 90, 0, 0]


    set_wsew(joint_angles)

    wait_for_completion()


def zero_position():
    """All joints set to zero degrees to match kin diagram."""
    waist_deg = 0  # between -150 to 150
    shoulder_deg = 0  # between 0 and 180
    elbow_deg = 0  # between 0 and -180
    wrist_deg = 0
    gripper_deg = 0
    set_wsew([waist_deg, shoulder_deg, elbow_deg, wrist_deg, gripper_deg])


def turn_on():
    Ax12.open_port()
    Ax12.set_baudrate()
    config_motor_angles()
    print('Connection Successful')
    all_motors.set_moving_speed(default_speed)


def sleep_position():
    waist_deg = 0  # between -150 to 150
    shoulder_deg = 170  # between 0 and 180
    elbow_deg = -160  # between 0 and -180
    wrist_deg = 30
    all_motors.set_moving_speed(20)
    set_wsew([waist_deg, shoulder_deg, elbow_deg, wrist_deg])
    wait_for_completion()


def turn_off():
    all_motors.disable_torque()
    Ax12.close_port()