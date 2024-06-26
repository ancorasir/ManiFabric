import time
import numpy as np
import rospy
import os
import yaml
import threading

from ADFM.real_robot.scripts.robot import Robot
from ADFM.real_robot.scripts.rs_camera import RSListener

class EnvReal:
    def __init__(self):
        rospy.init_node('env_real')

        path = os.path.dirname(os.path.abspath(__file__))

        with open(path + '/../cfg/robots.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # init controller client and move to init pose
        self.robot_left = Robot(config['robot_left'])
        self.robot_right = Robot(config['robot_right'])

        self.rs_listener = RSListener()
        image = self.get_image()
        self.camera_height, self.camera_width, _ = image.shape

        self.dt = 0.01 # actual control sequence time interval

    def gripper_close(self):
        self.robot_left.set_gripper_open(False)
        self.robot_right.set_gripper_open(False)
    def gripper_open(self):
        # self.robot_left.set_gripper_open(True)
        # self.robot_right.set_gripper_open(True)
        signal = True
        record_thread = threading.Thread(target=self.robot_left.set_gripper_open,
                                         args=(signal,))
        record_thread.start()
        self.robot_right.set_gripper_open(True)

    def step(self, actions, dt):
        # input: actions in [steps, action]; dt: dt per step

        traj_left = self.robot_left.prepare_traj(actions[:, 4:7], dt)
        traj_right = self.robot_right.prepare_traj(actions[:, :3], dt)

        self.robot_left.send_traj(traj_left)
        self.robot_right.send_traj(traj_right)
    def move(self, pose, arm='left'):
        if arm == 'left':
            goal = self.robot_left.prepare_move(pose)
            self.robot_left.send_traj(goal)
        else:
            goal = self.robot_right.prepare_move(pose)
            self.robot_right.send_traj(goal)
    def move_dual_arms(self, poses):
        self.move(poses[1], arm='left')
        self.move(poses[0], arm='right')

    def reset(self):
        self.robot_left.move_to_init_pose()
        self.robot_right.move_to_init_pose()
        print('Robot reset')

    def get_vox_pc(self):
        return self.rs_listener.vox_pc

    def get_image_mask(self):
        return self.rs_listener.mask

    def get_picker_position(self):
        picker_left = self.robot_left.get_picker_pose_in_origin()
        picker_right = self.robot_right.get_picker_pose_in_origin()
        return np.array([picker_right, picker_left])

    def get_ee_pose(self):
        ee_left = self.robot_left.get_ee_pose_in_origin()
        ee_right = self.robot_right.get_ee_pose_in_origin()
        return np.array([ee_right, ee_left])

    def get_image(self):
        timeout = 5
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.rs_listener.image is not None:
                return self.rs_listener.image
            time.sleep(0.1)

        raise TimeoutError("Timeout waiting for rs_listener.image to exist")

    def get_intrinsic(self):
        return [self.rs_listener.cx, self.rs_listener.cy, self.rs_listener.fx, self.rs_listener.fy]

    def get_depth_image(self):
        timeout = 5
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.rs_listener.depth_image is not None:
                return self.rs_listener.depth_image
            time.sleep(0.1)

        raise TimeoutError("Timeout waiting for rs_listener.depth_image to exist")
