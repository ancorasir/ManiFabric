import os
import numpy as np
import time

import pyflex

from ADFM.utils.utils import downsample, load_data, load_data_list, store_h5_data, pc_reward_model, draw_target_pos, draw_traj_pos, voxelize_pointcloud
from ADFM.utils.camera_utils import get_observable_particle_index, get_observable_particle_index_old, get_world_coords, get_observable_particle_index_3, get_matrix_world_to_camera
from softgym.utils.visualization import save_numpy_as_gif

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.style'] = 'normal'

class DataCollector(object):
    def __init__(self, args, phase, env):

        self.args = args
        self.env = env
        self.dt = args.dt
        self.data_names = None # just load and save everything
        ratio = self.args.train_valid_ratio

        if self.args.dataf is not None:
            self.data_dir = os.path.join(self.args.dataf, phase)
            os.system('mkdir -p ' + self.data_dir)

        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = int(self.args.n_rollout - int(self.args.n_rollout * ratio))
        else:
            raise AssertionError("Unknown phase")
        self.phase = phase
        self.swing_acc = 2.0
        self.pull_acc = 1.0

    def gen_dataset(self):
        rollout_idx = 0

        while rollout_idx < self.n_rollout:

            print("{} / {}".format(rollout_idx, self.n_rollout))
            rollout_dir = os.path.join(self.data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)
            self.env.reset()
            prev_data, rollout_data = self.get_curr_env_data(env_info=True)
            store_h5_data(self.data_names, rollout_data, os.path.join(rollout_dir, 'rollout_info.h5'))

            if self.args.gen_gif:
                frames_rgb, frames_depth = [prev_data['rgb']], [prev_data['depth']]
                frames_top, frames_side = [], []

            current_config = self.env.get_current_config()
            actions = self._collect_trajectory(prev_data['picker_position'], current_config[
                'target_picker_pos'])  # Get actions for the baseline and platform

            picker_position_list = []
            for j in range(1, len(actions) + 1):

                self.env.action_tool.update_picker_boundary([-0.3, 0, -0.5], [1, 2, 0.5])
                if not self._data_test(prev_data):
                    # raise error
                    raise Exception('_data_test failed: Number of point cloud too small')

                action = actions[j - 1]
                if action[3] > 0.5:
                    self.env.action_tool.picked_particles = self.env._get_drop_point_idx()
                _, reward, done, info = self.env.step(action, record_continuous_video=True, img_size=360)

                curr_data = self.get_curr_env_data()
                picker_position_list.append(curr_data['picker_position'])

                prev_data['velocities'] = (curr_data['positions'] - prev_data['positions']) / self.dt
                prev_data['action'] = action
                store_h5_data(self.data_names, prev_data, os.path.join(rollout_dir, str(j - 1) + '.h5'))
                prev_data = curr_data

                if self.args.gen_gif and rollout_idx<100:
                    frames_rgb.append(prev_data['rgb'])
                    # frames_depth.append(prev_data['depth'])

                    # frames_top.append(info['image_top'])
                    # frames_side.append(info['image_side'])

            # the last step has no action, and is not used in training
            prev_data['action'], prev_data['velocities'] = 0, 0
            store_h5_data(self.data_names, prev_data, os.path.join(rollout_dir, str(len(actions)) + '.h5'))

            if self.args.gen_gif and rollout_idx < 100:
                camera_pos, camera_angle = self.env.get_camera_params()
                matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

                frames_rgb = np.array(np.array(frames_rgb) * 255).clip(0., 255.)
                for t in range(len(frames_rgb)):
                    frames_rgb[t] = draw_target_pos(frames_rgb[t], self.env.get_current_config()['target_pos'],
                                                    matrix_world_to_camera[:3, :],
                                                    self.env.camera_height, self.env.camera_width,
                                                    self.env._get_key_point_idx())

                save_numpy_as_gif(frames_rgb, os.path.join(rollout_dir, 'rgb.gif'))

                # self._save_gifs_picker_traj(rollout_dir, frames_rgb, frames_depth, frames_top, frames_side, picker_position_list, current_config)

            rollout_idx += 1

    def get_curr_env_data(self, env_info=False):

        # Env info that does not change within one episode
        config = self.env.get_current_config()
        cloth_xdim, cloth_ydim = config['ClothSize']
        config_id = self.env.current_config_id
        scene_params = [self.env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

        downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, self.args.down_sample_scale)
        scene_params[1], scene_params[2] = downsample_x_dim, downsample_y_dim

        position = pyflex.get_positions().reshape(-1, 4)[:, :3]
        picker_position = self.env.action_tool.get_picker_pos()

        # Cloth and picker information
        # Get partially observed particle index
        rgbd = self.env.get_rgbd(show_picker=True)
        rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]

        world_coordinates = get_world_coords(rgb, depth, self.env, position)
        # Old way of getting observable index
        downsample_observable_idx = get_observable_particle_index_old(world_coordinates, position[downsample_idx], rgb, depth)

        ## observable_idx is not used anymore
        # observable_idx = get_observable_particle_index_old(world_coordinates, position, rgb, depth)
        observable_idx = []

        world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
        pointcloud = world_coords[depth.flatten() > 0]

        vox_pc = voxelize_pointcloud(pointcloud.astype(np.float32), self.args.voxel_size)

        ret_step = {'positions': position.astype(np.float32),
               'picker_position': picker_position,
               'downsample_observable_idx': downsample_observable_idx,
               'pointcloud': vox_pc} # store vox_pc for reduce memory usage - lead to a fixed voxel size

        if self.args.gen_gif:
            ret_step['rgb'], ret_step['depth'] = rgb, depth

        if env_info:
            ret_env = {'scene_params': scene_params,
                      'downsample_idx': downsample_idx,
                      'target_pos': config['target_pos'],}

            if self.env.current_config['env_shape'] != None:
                ret_env['shape_size'] = config['shape_size']
                ret_env['shape_pos'] = config['shape_pos']
                ret_env['shape_quat'] = config['shape_quat']
                ret_env['env_shape'] = config['env_shape']
            return ret_step, ret_env

        return ret_step

    def _collect_trajectory(self, current_picker_position, target_picker_position):

        """ Policy for collecting data - random sampling"""
        xy_trans = np.random.uniform(0.1, 0.5)
        z_ratio = np.random.uniform(0.1, 0.5)

        norm_direction = np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))
        middle_state = target_picker_position.copy()
        middle_state[:, [0, 2]] = target_picker_position[:, [0, 2]] + xy_trans * norm_direction
        middle_state[:, 1] = current_picker_position[:, 1] + z_ratio * (
                    target_picker_position[:, 1] - current_picker_position[:, 1])

        trajectory_start_to_middle = self._generate_trajectory(current_picker_position, middle_state,
                                                              self.swing_acc, self.dt)

        trajectory_middle_to_target = self._generate_trajectory(middle_state, target_picker_position,
                                                            self.pull_acc, self.dt)

        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        action_list = []
        for step in range(1, trajectory.shape[0]):
            action = np.ones_like(self.env.action_space.sample(), dtype=np.float32)
            action[:3], action[4:7] = trajectory[step, :3] - trajectory[step - 1, :3], trajectory[step,
                                                                                       3:6] - trajectory[step - 1, 3:6]
            action_list.append(action)

        action_list = np.array(action_list)
        action_list[:,[3,7]] = 1

        action_drop = np.zeros((30, 8))
        action_list = np.concatenate((action_list, action_drop), axis=0)
        return action_list

    def _generate_trajectory(self, current_picker_position, target_picker_position, acc_max, dt):

        """ Policy for trajectory generation based on current and target_picker_position"""

        # select column 1 and 3 in current_picker_position and target_picker_position
        initial_vertices_xy = current_picker_position[:, [0, 2]]
        final_vertices_xy = target_picker_position[:, [0, 2]]

        # calculate angle of rotation from initial to final segment in xy plane
        angle = np.arctan2(final_vertices_xy[1, 1] - final_vertices_xy[0, 1],
                           final_vertices_xy[1, 0] - final_vertices_xy[0, 0]) - \
                np.arctan2(initial_vertices_xy[1, 1] - initial_vertices_xy[0, 1],
                           initial_vertices_xy[1, 0] - initial_vertices_xy[0, 0])

        # translation vector: difference between final and initial centers
        translation = (target_picker_position.mean(axis=0) - current_picker_position.mean(axis=0))

        _time_steps = max(np.sqrt(4 * np.abs(translation) / acc_max) / dt)
        steps = np.ceil(_time_steps).max().astype(int)

        # calculate angle of rotation for each step
        rot_steps = angle / steps

        accel_steps = steps // 2
        decel_steps = steps - accel_steps

        v_max = translation * 2 / (steps * dt)
        accelerate = v_max / (accel_steps * dt)
        decelerate = -v_max / (decel_steps * dt)

        # calculate incremental translation
        incremental_translation = [0, 0, 0]

        # initialize list of vertex positions
        positions_xzy = [current_picker_position]


        # apply translation and rotation in each step
        for i in range(steps):
            if i < accel_steps:
                # Acceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + accelerate * self.dt) * self.dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + decelerate * self.dt) * self.dt

            # translate vertices
            vertices = positions_xzy[-1] + incremental_translation

            # calculate rotation matrix for this step
            rotation_matrix = np.array([[np.cos(rot_steps), 0, -np.sin(rot_steps)],
                                        [0, 1, 0],
                                        [np.sin(rot_steps), 0, np.cos(rot_steps)]])

            # rotate vertices
            center = vertices.mean(axis=0)
            vertices = (rotation_matrix @ (vertices - center).T).T + center

            # append vertices to positions
            positions_xzy.append(vertices)

        return positions_xzy

    def _generate_traj_z(self, start_z, mid_z, end_z, steps_s2m, steps_m2e, dt):

        steps = steps_s2m + steps_m2e
        v_max = (end_z - start_z) * 2 / (steps * dt)
        accelerate = v_max**2 / ( 2 * (mid_z - start_z))
        decelerate = -v_max**2 / ( 2 * (end_z - mid_z))

        steps_s2m = int(v_max / accelerate / dt)
        steps_m2e = steps - steps_s2m

        position_z = [start_z]
        _v = 0
        for i in range(steps):
            if i < steps_s2m:
                # Acceleration phase
                _v += accelerate * dt
                position_z.append(position_z[-1] + _v * dt)
            else:
                # Deceleration phase
                _v += decelerate * dt

                position_z.append(position_z[-1] + _v * dt)

        return position_z

    def _data_test(self, data):
        """ Filter out cases where cloth is moved out of the view or when number of voxelized particles is larger than number of partial particles"""
        pointcloud = data['pointcloud']
        if len(pointcloud.shape) != 2 or len(pointcloud) < 20:
            print('_data_test failed: Number of point cloud too small')
            return False
        return True

    def _save_gifs_picker_traj(self, rollout_dir, frames_rgb, frames_depth, frames_top, frames_side, picker_position_list, current_config):
        # save_numpy_as_gif(np.array(frames_depth) * 255., os.path.join(rollout_dir, 'depth.gif'))
        #
        # save_numpy_as_gif(np.array(frames_top), os.path.join(rollout_dir, 'rgb_top.gif'))

        matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=np.array([0, 0, 0]), cam_pos=np.array([0.3,0.3, 1.5]))

        frames_side = np.array(frames_side)
        for t in range(len(frames_side)):
            frames_side[t], mid_index = draw_traj_pos(frames_side[t], np.array(picker_position_list)[t:,1,:],
                                            matrix_world_to_camera[:3, :],
                                            self.env.camera_height, self.env.camera_width)
            if t ==0:
                index = mid_index
        save_numpy_as_gif(frames_side[:], os.path.join(rollout_dir, 'rgb_side.gif'))

        # picker_position_list = np.average(picker_position_list, axis=1)

        # plot the picker position and vel in 3 axis

        # fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 6))
        # ax0.plot(picker_position_list[:, 0])
        # ax0.set_title('pos x')
        # ax1.plot(picker_position_list[:, 1])
        # ax1.set_title('pos z')
        # ax2.plot(picker_position_list[:, 2])
        # ax2.set_title('pos y')
        # ax3.plot(np.diff(np.diff(picker_position_list[:, 0]) / self.dt) / self.dt)
        # ax3.set_title('vel x')
        # ax4.plot(np.diff(np.diff(picker_position_list[:, 1]) / self.dt) / self.dt)
        # ax4.set_title('vel z')
        # ax5.plot(np.diff(np.diff(picker_position_list[:, 2]) / self.dt) / self.dt)
        # ax5.set_title('vel y')
        # plt.savefig(os.path.join(rollout_dir, 'picker_position.png'))
        # plt.close(fig)
        #
        # plt.plot(picker_position_list[:, 0], picker_position_list[:, 1])
        # plt.savefig(os.path.join(rollout_dir, 'X-Z.png'))
        # plt.close()
