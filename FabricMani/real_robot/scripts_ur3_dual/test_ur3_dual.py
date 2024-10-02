# script to test ur3_dual (moving to a position)
# via MoveGroupPythonInteface class (MUST run moveit!)
# NOTE: all poses are based on {base_link} frame
#
# Z.Zhang
# 09-2024


import os
import yaml
import rospy
import copy
import sys
import numpy as np
from utils.ur_move import MoveGroupPythonInteface, gen_pose
from ur3_dual_exe import Robot
import moveit_commander
# import rospy
# import actionlib
# import ur_dashboard_msgs.msg # type: ignore

ns_cur = 'ur3_left'

# # # UR APIs -----------------------------------------------------------------
# def play_program(wait_result=True):
#     mode=ur_dashboard_msgs.msg.RobotMode(mode=ur_dashboard_msgs.msg.RobotMode.RUNNING)
#     mode_action_goal = ur_dashboard_msgs.msg.SetModeGoal(target_robot_mode=mode, stop_program=True, play_program=True)
#     client_set_mode.send_goal(mode_action_goal)
#     res=None
#     if wait_result:
#         client_set_mode.wait_for_result(rospy.Duration.from_sec(5.0))
#         res = client_set_mode.get_result()
#     return res

if __name__ == '__main__':
    rospy.init_node('my_unique_node_name', anonymous=True)
    ur_control = MoveGroupPythonInteface(robot_description=f"{ns_cur}/robot_description", ns=ns_cur, sim=False)  #real
    
    # velocity setting
    maxVelScale = 1.0
    ur_control.set_speed_slider(maxVelScale, ns_cur) # set speed in the teach pendant

    # load config file according to the real robot setup
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/../cfg/ur3_dual.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # # init controller client and move to init pose
    # robot_left = Robot(config['robot_left'])
    # robot_left.move_to_init_pose()
    # rospy.sleep(3)

    # test functions from moveit_commander
    # NOTE: run following codes before cancel the instance of Robot class
    # Step 1: virtually hit RUN button in the teach pendant
    # NOTE: Once the connection bewteen reserve interface & robot is done, no need to run "play_program()" again
    # Method 1
    # client_set_mode = actionlib.SimpleActionClient(f"{ns_cur}/ur_hardware_interface/set_mode", ur_dashboard_msgs.msg.SetModeAction)
    # client_set_mode.wait_for_server()
    # play_program()    
    # rospy.sleep(3)
    # Method 2
    ur_control.play_program()
    rospy.sleep(3)

    # It works!
    # init_pose = [0.295, 0.311, 0.103, 0.0, 0.3756404, 0.9181996, -0.1257133]
    # goal_pose = gen_pose(*init_pose)
    # ur_control.go_to_pose_goal(goal_pose)

    # It works!
    home_config = [np.pi / 12.0, -np.pi * 1.0 / 3.0, -np.pi * 2.0 / 3.0, -np.pi, -np.pi * 2.0 / 3.0, 0]
    joint_goal = ur_control.group.get_current_joint_values()
    joint_goal = home_config
    ur_control.group.go(joint_goal, wait=True)
    ur_control.group.stop()
    ur_control.group.clear_pose_targets()

    # It works!
    # waypoints = []
    # wpose = ur_control.group.get_current_pose().pose
    # wpose.position.z += 0.1
    # waypoints.append(copy.deepcopy(wpose))
    # # based on robot base frame
    # (plan, fraction) = ur_control.group.compute_cartesian_path(
    #                                waypoints,   # waypoints to follow
    #                                0.01,        # eef_step
    #                                0.0)
    # ur_control.group.execute(plan, wait=True)

    # It works!
    waypoints = []
    wpose = ur_control.group.get_current_pose().pose # based on {base_link}
    wpose.position.z += 0.1
    waypoints.append(copy.deepcopy(wpose))
    wpose.position.x -= 0.12
    waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    ur_control.group.execute(plan, wait=True)