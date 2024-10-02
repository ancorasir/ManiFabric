# coding=utf-8
from __future__ import print_function
import copy
import math
import numpy as np
import os
import sys

import rospy
from tf import transformations as tfs
import actionlib

import geometry_msgs.msg
import std_msgs.msg
import moveit_commander # NOTE: use it before roslauch moveit_planning_execution.launch & moveit_rviz.launch
from moveit_commander.conversions import pose_to_list
import moveit_msgs.msg

import ur_dashboard_msgs.msg
import ur_dashboard_msgs.srv
import ur_msgs.msg
import ur_msgs.srv  

# import helper

DEF_TOPIC_HW = 'ur_hardware_interface'
dashboard_srv_prefix = DEF_TOPIC_HW + '/dashboard/'


def gen_pose(x, y, z, rx, ry, rz, rw):
    """
    Generate a geometry_msgs.msg.Pose object
    args: x, y, z, rx, ry, rz, rw (scalar-last quaternion)
    """
    pose = geometry_msgs.msg.Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = rx
    pose.orientation.y = ry
    pose.orientation.z = rz
    pose.orientation.w = rw
    return pose


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class MoveGroupPythonInteface(object):
    """MoveGroupPythonIntefaceTutorial"""

    def __init__(self, robot_description="robot_description", ns="", sim=False):
        super(MoveGroupPythonInteface, self).__init__()
        rospy.loginfo('Moveit init start ...')
        cleaned_args = [a for a in sys.argv if not os.path.basename(__file__) in a]
        moveit_commander.roscpp_initialize(cleaned_args)

        self.in_sim = sim
        
        group_name = "manipulator"

        self.robot = moveit_commander.RobotCommander(robot_description,ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns)
        self.group = moveit_commander.MoveGroupCommander(group_name,robot_description,ns)

        # Print infos
        # planning_frame = self.group.get_planning_frame()
        # print("============ Reference frame: %s" % planning_frame)

        # eef_link = self.group.get_end_effector_link()
        # print("============ End effector: %s" % eef_link)

        # group_names = self.robot.get_group_names()
        # print("============ Robot Groups:", group_names)

        # print("============ Printing robot state")
        # print(self.robot.get_current_state())
        # print("")

        self.vel_scale = 0.15
        self.acc_scale = 0.15

        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)

        # Allow replanning to increase the odds of a solution
        self.group.allow_replanning(True)

        # self.group.set_support_surface_name()
        # self.group.pick()
        # self.group.set_pose_reference_frame()
        # self.position_pid = PID()
        # self.position_pid.setWindup(0.05)
        # self.position_pid.out_max = 0.2
        # self.position_pid.out_min = -self.position_pid.out_max
        # self.position_pid.Kp = 5
        #
        # self.z_pid = PID(P=7., I=0., D=0.25)
        # self.z_pid.out_max = 0.4
        # self.z_pid.out_min = -self.z_pid.out_max

        self.display_trajectory_publisher = rospy.Publisher(f'{ns}/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

        self.script_pub = rospy.Publisher(f"{ns}/"+DEF_TOPIC_HW + '/script_command', std_msgs.msg.String, queue_size=10)

        if not self.in_sim:
            self.client_set_mode = actionlib.SimpleActionClient(f"{ns}/"+DEF_TOPIC_HW + '/set_mode', ur_dashboard_msgs.msg.SetModeAction)
            self.client_set_mode.wait_for_server()

        # Ok
        rospy.loginfo('Moveit init finished!')

    def __del__(self):
        print("__del__ MoveGroupPythonIntefaceTutorial")
        moveit_commander.roscpp_shutdown()

    # Read data ------------------------------------------------
    def get_pose_list(self):
        return pose_to_list(self.group.get_current_pose().pose)

    def get_joint_pos_list(self):
        return self.group.get_current_joint_values()

    # def get_tf_baselink2ee(self):
    #     return helper.pose_to_mat44(self.group.get_current_pose().pose)

    def print_cur_state(self):
        cur_pose = self.group.get_current_pose().pose
        euler = tfs.euler_from_quaternion(
            [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w])
        euler = np.degrees(np.array(euler))
        print("{}, {}, {}; {.2f}, {.2f}, {.2f}".format(cur_pose.position.x, cur_pose.position.y, cur_pose.position.z,
                                               euler[0], euler[1], euler[2]))

    def get_robot_current_pose(self):
        cur_pose = self.group.get_current_pose().pose
        tmp_pose = np.zeros(6)
        tmp_pose[0] = cur_pose.position.x
        tmp_pose[1] = cur_pose.position.y
        tmp_pose[2] = cur_pose.position.z

        # TODO: CHECK boarder
        # if tmp_pose[2] < -0.070 + gripper_length:
        #     self.stopl_control(2)
        #     rospy.loginfo("Robot is stopped because small z {}".format(tmp_pose[2]))
        #     sys.exit(-1)

        # print("Current robot pose is {}.".format(tmp_pose))
        return tmp_pose

    # Plan ----------
    def plan_to_pose(self, pose):
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(pose)
        plan = self.group.plan()
        return plan

    def plan_cartesian_path(self, goal_pose):
        waypoints = []
        waypoints.append(copy.deepcopy(goal_pose))

        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0, avoid_collisions=True)
        return plan, fraction

    def display_trajectory(self, plan):
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)

    # Move with Moveit --------------------------------------------------------
    def go_to_joint_state(self, base, shoulder, elbow, whist1, whist2, whist3):
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = base
        joint_goal[1] = shoulder
        joint_goal[2] = elbow
        joint_goal[3] = whist1
        joint_goal[4] = whist2
        joint_goal[5] = whist3

        self.group.go(joint_goal, wait=True)
        self.group.stop()

        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self, pose_goal):
        self.group.set_pose_target(pose_goal)

        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def go_cartesian_path(self, goal_pose, execute=True):
        """
        base_link frame --> eef_link frame
        """
        if type(goal_pose) is geometry_msgs.msg.Pose:
            waypoints = [copy.deepcopy(goal_pose)]
        elif type(goal_pose) is list:
            waypoints = copy.deepcopy(goal_pose)
            goal_pose = goal_pose[-1]
        else:
            rospy.logerr("goal_pose should be geometry_msgs.msg.Pose/list[geometry_msgs.msg.Pose]")
            return None, None

        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0, avoid_collisions=True)
        rospy.loginfo('plan_fraction: {}'.format(fraction))
        traj_out = None
        if fraction > 0.8:
            num_pts = len(plan.joint_trajectory.points)
            rospy.loginfo("Path computed successfully. intermediate waypoints = {}\n"
                          "Moving the arm to ({}, {}, {})".
                          format(num_pts, goal_pose.position.x, goal_pose.position.y, goal_pose.position.z))

            traj_out = self.group.retime_trajectory(self.robot.get_current_state(), plan,
                                                    velocity_scaling_factor=self.vel_scale,
                                                    acceleration_scaling_factor=self.acc_scale,
                                                    algorithm='time_optimal_trajectory_generation')
                                                    # algorithm='iterative_spline_parameterization')
                                                    # algorithm='iterative_time_parameterization')
            if execute:
                self.group.execute(plan)
                rospy.loginfo("Path execution complete.")
        return traj_out, fraction

    # Setup scene -------------------------------------------------------------
    def remove_allobjects(self, timeout=4):
        eef_link = 'ee_link'
        base_link = 'base_link'

        self.scene.remove_attached_object(eef_link)
        self.scene.remove_attached_object('tool0')
        self.scene.remove_attached_object(base_link)
        self.scene.remove_world_object()

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects()
            known_objects = self.scene.get_known_object_names()

            if (len(attached_objects.keys()) == 0) and (len(known_objects) == 0):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()
        return False

    def setup_scene(self):
        ret1 = self.add_basebox()
        rospy.loginfo('add_basebox: ' + str(ret1))
        ret2 = self.attach_basebox()
        rospy.loginfo('attach_basebox: ' + str(ret2))
        return (ret1 and ret2)
        # ret3 = self.add_conveyer()
        # rospy.loginfo('add_conveyer: ' + str(ret3))
        # return (ret1 and ret2 and ret3)

    def wait_for_state_update(self, object_name, object_is_known=False, object_is_attached=False, timeout=4):
        scene = self.scene

        ## Ensuring Collision Updates Are Receieved
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_known_object_names()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([object_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = object_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (object_is_attached == is_attached) and (object_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def add_basebox(self, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"

        box_name = "base_box"

        box_size = (0.43, 0.43, 0.86)
        quat = tfs.quaternion_from_euler(math.radians(0), math.radians(0), math.radians(0))
        box_pose.pose.position.x = 0
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0 - box_size[2] / 2
        box_pose.pose.orientation.x = quat[0]
        box_pose.pose.orientation.y = quat[1]
        box_pose.pose.orientation.z = quat[2]
        box_pose.pose.orientation.w = quat[3]

        self.scene.add_box(box_name, box_pose, size=box_size)

        return self.wait_for_state_update(box_name, object_is_known=True, timeout=timeout)

    def attach_basebox(self, timeout=4):
        base_link = 'base_link'
        box_name = "base_box"
        self.scene.attach_box(base_link, box_name, touch_links=[base_link])

        return self.wait_for_state_update(box_name, object_is_attached=True, object_is_known=False, timeout=timeout)

    # UR Script ---------------------------------------------------------------
    def speedl_control(self, vel, acc, time2move):
        """
        control speed of tool0_controller in base frame

        ========== speedl(xd, a, t, aRot=’a’) ==========
        Tool speed

        Accelerate linearly in Cartesian space and continue with constant tool
        speed. The time t is optional; if provided the function will return after
        time t, regardless of the target speed has been reached. If the time t is
        not provided, the function will return when the target speed is reached.
        Parameters
        xd:     tool speed [m/s, rad/s] (spatial vector)
        a:      tool position acceleration [m/s^2]
        t:      time [s] before function returns (optional)
        aRot:   tool acceleration [rad/s^2] (optional), if not defined
                a, position acceleration, is used
        Example command: speedl([0.5,0.4,0,1.57,0,0], 0.5, 0.5)
            • Example Parameters:
                – xd → Tool speeds of: x=500 mm/s, y=400 mm/s, rx=90 deg/s,
                    ry and rz=0 mm/s
                – a = 0.5 rad/s^2 → acceleration of the leading axis (shoulder is
                    this case)
                – t = 0.5 s → time before the function returns

        """
        move_msg = "speedl([{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}],{:.3f},{:.3f})\n". \
            format(vel[0], vel[1], vel[2],
                   vel[3], vel[4], vel[5],
                   acc, time2move)
        self.script_pub.publish(move_msg)

    def movel_control(self, pose, acc, vel):
        """
        move tool0_controller in base frame

        ========== movej(q, a=1.4, v=1.05, t=0, r=0) ==========
        Move to position (linear in tool-space)

        See movej.
        Parameters
            pose: target pose (pose can also be specified as joint
            positions, then forward kinematics is used to
            calculate the corresponding pose)
            a: tool acceleration [m/s^2]
            v: tool speed [m/s]
            t: time [S]
            r: blend radius [m]
        Example command:
            movel(pose, a=1.2, v=0.25, t=0, r=0)
            • Example Parameters:
                – pose = p[0.2,0.3,0.5,0,0,3.14] → position in base frame of x =
                  200 mm, y = 300 mm, z = 500 mm, rx = 0, ry = 0, rz = 180 deg
                – a = 1.2 → acceleration of 1.2 m/s^2
                – v = 0.25 → velocity of 250 mm/s
                – t = 0 → the time (seconds) to make the move is not specified.
                  If it were specified the command would ignore the a and v
                  values.
                – r = 0 → the blend radius is zero meters.
        """
        move_msg = "movel([{},{},{},{},{},{}],{},{})\n". \
            format(str(pose[0]), str(pose[1]), str(pose[2]),
                   str(pose[3]), str(pose[4]), str(pose[5]),
                   str(acc), str(vel))
        self.script_pub.publish(move_msg)

    def stopl_control(self, acc_trans):
        """
        ========== stopl(a, aRot=’a’) ==========
        Stop (linear in tool space)

        Decelerate tool speed to zero
        Parameters
        a:      tool accleration [m/s^2]
        aRot:   tool acceleration [rad/s^2] (optional), if not defined
                a, position acceleration, is used
        Example command: stopl(20)
            • Example Parameters:
                – a = 20 m/s^2 → rate of deceleration of the tool
                – aRot → tool deceleration [rad/s^2] (optional), if not defined,
                  position acceleration, is used. i.e. it supersedes the "a"
                  deceleration.
        """
        # Stop (linear in tool space) Decelerate tool speed to zero
        move_msg = "stopl({})\n".format(str(acc_trans))
        self.script_pub.publish(move_msg)

    # def velocity_control(self, pose_target, vel_target):
    #     self.pose_now = self.get_robot_current_pose()
    #     self.pose_target = pose_target
    #     self.vel_target = vel_target
    #
    #     pose_loop_vel = self.position_pid.update(self.pose_target, self.pose_now)
    #     # pose_loop_vel = np.zeros(6)
    #     # pose_loop_vel = self.PIDPositionControl(self.pose_now, self.pose_target)
    #
    #     out = self.vel_target + pose_loop_vel
    #
    #     # currently, only x, y
    #     for i in range(2, len(out)):
    #         out[i] = 0
    #
    #     self.vel_out = np.clip(out, self.position_pid.out_min, self.position_pid.out_max)
    #     return True

    # def velocity_control_3d(self, pose_target, vel_target):
    #     self.pose_now = self.get_robot_current_pose()
    #     self.pose_target = pose_target
    #     self.vel_target = vel_target
    #
    #     pose_loop_vel = self.position_pid.update(self.pose_target, self.pose_now)
    #
    #     out = self.vel_target + pose_loop_vel
    #
    #     # currently, only x, y, z
    #     for i in range(3, len(out)):
    #         out[i] = 0
    #
    #     self.vel_out = np.clip(out, self.position_pid.out_min, self.position_pid.out_max)
    #     return True

    # def PIDPositionControl(self, current_pose, target_pose):
    #
    #     pid_err = target_pose - current_pose
    #
    #     for i in range(6):
    #         if self.error_pose_1[i] <= self.pose_delta_2frame[i] and self.error_pose_1[i] >= -self.pose_delta_2frame[i]:
    #             self.error_pose_1[i] = 0
    #         else:
    #             self.error_pose_1[i] = self.error_pose_1[i] - self.pose_delta_2frame[i]
    #
    #     target_pose_pid = self.pid_pose[0] * self.error_pose_1 + \
    #                       self.pid_pose[1] * (self.error_pose_1 + self.error_pose_2 + self.error_pose_3) + \
    #                       self.pid_pose[2] * (self.error_pose_1 - 2 * self.error_pose_2 + self.error_pose_3)
    #
    #     '''
    #     for j in range(6):
    #         if target_pose_pid[j] > self.max_vel_tcp[j]:   # prevent too high speed
    #             target_pose_pid[j] = self.max_vel_tcp[j]
    #         if target_pose_pid[j] < self.min_vel_tcp[j]:   # prevent too low speed
    #             target_pose_pid[j] = self.min_vel_tcp[j]
    #     '''
    #     self.error_pose_3 = self.error_pose_2
    #     self.error_pose_2 = self.error_pose_1
    #
    #     print("Target pose loop vel  is {}".format(target_pose_pid))
    #     print("Error pose between 2f is {}".format(self.error_pose_1))
    #     print("Delata pose           is {}\n\n\n.".format(self.pose_delta_2frame))
    #
    #     return target_pose_pid

    # End UR Script -----------------------------------------------------------
    # -------------------------------------------------------------------------

    # UR APIs -----------------------------------------------------------------
    def play_program(self, wait_result=True):
        if self.in_sim:
            return True
        mode=ur_dashboard_msgs.msg.RobotMode(mode=ur_dashboard_msgs.msg.RobotMode.RUNNING)
        mode_action_goal = ur_dashboard_msgs.msg.SetModeGoal(target_robot_mode=mode, stop_program=True, play_program=True)
        self.client_set_mode.send_goal(mode_action_goal)
        res=None
        if wait_result:
            self.client_set_mode.wait_for_result(rospy.Duration.from_sec(5.0))
            res = self.client_set_mode.get_result()
        return res

    def set_speed_slider(self, fraction, ns, wait_result=True):
        assert 0 < fraction <= 1, "'speed_slider_fraction' must be from (0, 1.0]"
        
        if self.in_sim:
            return True
        
        srv_name = f"{ns}/"+DEF_TOPIC_HW + '/set_speed_slider'
        rospy.wait_for_service(srv_name)
        try:
            srv_fun = rospy.ServiceProxy(srv_name, ur_msgs.srv.SetSpeedSliderFraction)
            resp1 = srv_fun(fraction)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        return None

    def call_service(self, ns, srv='get_loaded_program'):
        srv_name = f"{ns}/"+ dashboard_srv_prefix + srv
        rospy.wait_for_service(srv_name)
        try:
            proxy = rospy.ServiceProxy(srv_name, ur_dashboard_msgs.srv.GetLoadedProgram)
            resp1 = proxy()
            rospy.loginfo(("srv_name: {}\n {} \n" +"-"*10).format(srv_name, resp1))
            return resp1
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def load_program(self, ns, urp_file='ext_control_17.urp'):
        srv_name = f"{ns}/"+ dashboard_srv_prefix + 'load_program'
        rospy.wait_for_service(srv_name)
        try:
            proxy = rospy.ServiceProxy(srv_name, ur_dashboard_msgs.srv.Load)
            resp1 = proxy(urp_file)
            rospy.loginfo(("srv_name: {}\n {} \n" + "-"*10).format(srv_name, resp1))
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    # End UR APIs -------------------------------------------------------------
    # -------------------------------------------------------------------------


if __name__ == '__main__':
    rospy.init_node("my_unique_node_name")
    ur_control = MoveGroupPythonInteface()
    rospy.loginfo('init ok.')
    rospy.sleep(1)

    # get current info.
    rospy.loginfo('current pose[xyzxyzw]: \n{}'.format(np.round(ur_control.get_pose_list(),6)))
    rospy.loginfo('current joint: \n{}'.format(np.round(ur_control.get_joint_pos_list(),6)))
    rospy.loginfo('shut down')

# ret = ur_control.add_basebox()
# rospy.loginfo('add_basebox: ' + str(ret))
# ret = ur_control.attach_basebox()
# rospy.loginfo('attach_basebox: ' + str(ret))
# ret = ur_control.add_board()
# rospy.loginfo('add_board: ' + str(ret))
#
# JS_HOME = np.radians([0, -90, 0, -90, 0, 0]) # UR竖直
# JS_START = np.radians([-82, -57, 103, -118, 86, -5])
# ur_control.go_to_joint_state(ur_control, *JS_HOME)
#
# # 高处 ready
# JS_READY= [-1.4276731649981897, -1.5548022429095667, 1.7654247283935547, -1.7804668585406702, -1.5711053053485315, -0.10726625124086553]
# ur_control.go_to_joint_state(ur_control, *JS_READY)
#
# # ur_control.go_to_joint_state(ur_control, *JS_START)
#
# # ur_control.go_to_pose_goal()
#
#
# # cartesian_plan, fraction=tutorial.plan_cartesian_path(-0.054077938141, -0.555849527928, 0.252996572804, -0.400325729972, -0.406671685541, -0.557778955449, 0.602694024327)
# # tutorial.display_trajectory(cartesian_plan)
# # tutorial.execute_plan(cartesian_plan)
#
#
#
# quat = tfs.quaternion_from_euler(math.radians(-104.059507844), math.radians(90), math.radians(180))
# goal_pose = gen_pose(0.177145663366, -0.464385814842, 0.34975215481, *quat)
# ur_control.go_to_pose_goal(goal_pose)
#
#
# fall_pose = copy.deepcopy(goal_pose)
# fall_pose.position.z = 0.1
# ur_control.go_to_pose_goal(fall_pose)

# import time
# script=ur_control.speedl_control([-0.1, 0, 0, 0, 0, 0], 0.2, 3)
# ur_control.script_pub.publish(script)
# time.sleep(1)
# script=ur_control.speedl_control([0, 0, 0, 0, 0, 0], 0.2, 3)
# ur_control.script_pub.publish(script)
# time.sleep(1)
# script=ur_control.speedl_control([0.1, 0, 0, 0, 0, 0], 0.2, 3)
# ur_control.script_pub.publish(script)
#
# print_cur_state(ur_control)
# a=ur_control.group.get_current_pose().pose
# ur_control.group.get_current_pose().pose
# ur_control.robot.get_current_state()
# start_pose = gen_pose(0.188030545591, -0.595997693534, 0.213596399722, 0.599286810258, -0.536236615971, 0.403848155934, 0.436133325591)
#
# cur_pose = ur_control.group.get_current_pose().pose
# cur_pose.orientation.x
# euler = tfs.euler_from_quaternion([cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w])
# euler= np.array(euler)*180/np.pi
#
# (plan, fraction) = plan_cartesian_path(ur_control, start_pose)
# ur_control.group.execute(plan, wait=False)
# time.sleep(0.2)
# (plan, fraction) = plan_cartesian_path(ur_control, goal_pose)
# ur_control.group.execute(plan)
# time.sleep(1)
# test()
# def test():
#     (plan, fraction) = plan_cartesian_path(ur_control, goal_pose)
#     t_end = plan.joint_trajectory.points[-1].time_from_start
#     import time
#     t_move_start = time.time()
#     ur_control.group.execute(plan)
#     t_move_end = time.time()
#     rospy.loginfo('plan time: ' + str(t_end.to_sec()))
#     rospy.loginfo('real time: ' + str((t_move_end-t_move_start)))
#
#
# display_trajectory(ur_control, plan)
# ur_control.group.execute(plan)
# go_to_pose_goal(ur_control, start_pose)
# plan1 = plan_to_pose(ur_control, start_pose)
# goal_pose = copy.deepcopy(start_pose)
# goal_pose.position.x+=0.1
# plan1 = plan_to_pose(ur_control, goal_pose)
# ur_control.group.execute(plan1)
# rospy.sleep(1)

# (plan, fraction) = plan_cartesian_path(ur_control, goal_pose)
