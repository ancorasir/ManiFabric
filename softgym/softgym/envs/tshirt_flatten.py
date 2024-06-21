import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation as R
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TshirtFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='tshirt_flatten_init_states.pkl', cloth_type='tshirt-small', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        self.cloth_type = cloth_type
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.shorts_pkl_path = os.path.join(cur_path, '../cached_initial_states/shorts_flatten.pkl')
        self.env_shape = kwargs['env_shape']

        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            x_target = np.random.uniform(0.2, 0.35)
            rot_angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            z_target = 0

            if self.env_shape == 'random':
                env_shape_list = [None, 'platform', 'sphere', 'rod']
                env_shape = env_shape_list[i % 4]
            else:
                env_shape = self.env_shape

            if env_shape is not None:

                config['shape_size'], config['shape_pos'], config['shape_quat'], config['env_shape'] = self._add_shape(x_target=x_target, cloth_dimx=0, rot_angle=rot_angle, env_shape=env_shape)
                z_target = config['shape_size'][1]
                if config['env_shape'] == 'rod':
                    z_target = config['shape_size'][0] + config['shape_pos'][1]
                if config['env_shape'] == 'table':
                    z_target = config['shape_size'][1] + config['shape_pos'][1]

            else:
                config['env_shape'] = None

            config['x_target'] = x_target
            config['rot_angle'] = rot_angle

            # set shape color
            n_shapes = pyflex.get_n_shapes()
            color_shapes = np.zeros((n_shapes, 3))
            color_shapes[:1, :] = [1.0, 1.0, 0.4]
            pyflex.set_shape_color(color_shapes)

            self._set_to_flat(x_target=x_target, delta_z= z_target, rot_angle=rot_angle)

            pickpoints = self._get_drop_point_idx()
            picker_pos = self.set_picker_wait(pickpoints, max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold,
                                              grasped=False)
            config['target_picker_pos'] = picker_pos
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            config['target_pos'] = curr_pos[:, :3]

            ## Set the cloth to initial position and wait to stablize
            self._set_to_vertical(height_high= 0)

            self.set_picker_wait(pickpoints, max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold)

            generated_configs.append(deepcopy(config))
            print('config {}: camera params {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states
    def _set_to_vertical(self,height_high):

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self.default_pos_vertical.reshape(-1,4)
        curr_pos[:, :4] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < height_high:
            curr_pos[:, 1] += height_high - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def set_picker_wait(self, pickpoints, max_wait_step=500, stable_vel_threshold=0.1, grasped=True):

        curr_pos = pyflex.get_positions().reshape(-1, 4)
        # curr_pos[0] += np.random.random() * 0.001  # Add small jittering
        if grasped:
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
        pickpoint_pos = curr_pos[pickpoints, :3]
        pyflex.set_positions(curr_pos.flatten())

        if grasped:
            pickpoint_pos += np.array([[0, 0, -0.02], [0, 0, 0.02]])

        picker_radius = self.action_tool.picker_radius
        self.action_tool.update_picker_boundary([-0.3, 0.01, -0.5], [0.5, 2, 0.5])
        self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))
        picker_pos = pickpoint_pos + np.array([0., picker_radius, 0.])


        # wait to stablize
        for _ in range(max_wait_step):
            pyflex.step()
            pyflex.render()
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_vel = pyflex.get_velocities().reshape((-1, 3))
            if np.alltrue(curr_vel < stable_vel_threshold) and _ > 100:
                break
            if grasped:
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
        if grasped:
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())

        return picker_pos

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0., -1., 0.])
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))

        if 'env_shape' in self.current_config.keys():
            if self.current_config['env_shape'] == 'platform':
                self._add_box(self.current_config['shape_size'], self.current_config['shape_pos'], self.current_config['shape_quat'])

            if self.current_config['env_shape'] == 'sphere':
                self._add_sphere(self.current_config['shape_size'][0], self.current_config['shape_pos'], self.current_config['shape_quat'])

            if self.current_config['env_shape'] == 'rod':
                self._add_rod(self.current_config['shape_size'], self.current_config['shape_pos'], self.current_config['shape_quat'])

            if self.current_config['env_shape'] == 'table':
                self._add_table(self.current_config['shape_size'], self.current_config['shape_pos'], self.current_config['shape_quat'])

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('key_point'):
            # TODO ad action_repeat
            print('Need to add action repeat')
            raise NotImplementedError
            raise DeprecationWarning
            valid_idxs = np.array([0, 63, 31 * 64, 32 * 64 - 1])
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            pyflex.step()

            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 4])
            idxs = np.hstack(action[:, 0])
            updates = action[:, 1:]
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            vels = pyflex.get_velocities()
            cur_pos[:, 3] = 1
            if self.action_mode == 'key_point_pos':
                cur_pos[valid_idxs[idxs.astype(int)], :3] = last_pos[valid_idxs[idxs.astype(int)]][:, :3] + updates
                cur_pos[valid_idxs[idxs.astype(int)], 3] = 0
            else:
                vels = np.array(vels).reshape([-1, 3])
                vels[idxs.astype(int), :] = updates
            pyflex.set_positions(cur_pos.flatten())
            pyflex.set_velocities(vels.flatten())
        else:
            original_inv_mass = self.action_tool.step(action)
            if self.action_mode in ['sawyer', 'franka']:
                pyflex.step(self.action_tool.next_action)
            else:
                pyflex.step()
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[:, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
        return

    def _get_current_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area
        return r

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p

    def _get_info(self):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        performance = -curr_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        IoU = self._get_IoU(particle_pos.reshape((-1, 4))[:, :3], self.get_current_config()['target_pos'], voxel_size=0.0216)
        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'IoU': IoU,}

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    def _get_IoU(self, pc_pos, target_pos, voxel_size=0.0216):
        def voxel_set(point_cloud, voxel_size):
            """
            Create a set of unique voxels occupied by the point cloud.
            """
            voxel_coords = np.floor(point_cloud / voxel_size).astype(int)
            return set(map(tuple, voxel_coords))

        voxel_set_1 = voxel_set(pc_pos, voxel_size)
        voxel_set_2 = voxel_set(target_pos, voxel_size)

        # Compute intersection and union
        intersection = voxel_set_1.intersection(voxel_set_2)
        union = voxel_set_1.union(voxel_set_2)

        # Calculate IoU
        iou = len(intersection) / len(union) if union else 0
        return iou

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker, dtype=np.int32) * -1  # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    def get_picked_particle_new_position(self):
        intermediate_picked_particle_new_pos = self.action_tool.intermediate_picked_particle_pos
        if len(intermediate_picked_particle_new_pos) > 0:
            return np.vstack(intermediate_picked_particle_new_pos)
        else:
            return []

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 3
        if self.cloth_type == 'tshirt':
            cloth_type = 0
        elif self.cloth_type == 'shorts':
            cloth_type = 1
        else:
            cloth_type = 2
        scene_params = np.concatenate(
            [config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
             camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']], [render_mode], [cloth_type]])
        if self.version == 2:
            robot_params = []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            print("set scene")
            pyflex.set_scene(env_idx, scene_params, 0)
            print("after set scene")
        self.rotate_particles([0, 90, 0])
        self.move_to_pos([0, 0.3, 0])
        self.default_pos_vertical = pyflex.get_positions()

        self.rotate_particles([90, 0, 0])
        self.move_to_pos([0, 0.05, 0])

        # pos = pyflex.get_positions().reshape(-1, 4)
        # # plot 3d point cloud
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # _pos = pos[pos[:,0]<-0.06].copy()
        # ax.scatter(_pos[:,0], _pos[:,1], _pos[:,2])
        # idx_x = np.argsort(pos[:, 0])[:100]
        # for i, (x, y, z) in enumerate(pos[:,:3]):
        #     if -0.08<x< -0.06:
        #         # ax.scatter(x, y, z,)
        #         ax.text(x, y, z, f'{i}', color='green', fontsize=10)
        #
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')
        # ax.set_title('3D Point Cloud')
        # plt.show()
        #
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot()
        #
        # _pos = pos[:,[0,2]].copy()
        # ax.scatter(_pos[:,0], _pos[:,1], s=1)
        # for i, (x, y) in enumerate(_pos[:,:2]):
        #     if i%4==0 or i in [1075,1073]:
        #         ax.text(x, y, f'{i}', color='green', fontsize=10)
        #
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # plt.show()

        for _ in range(50):
            # print("after move to pos step {}".format(_))
            pyflex.step()
            pyflex.render()
            # obs = self._get_obs()
            # cv2.imshow("obs at after move to obs", obs)
            # cv2.waitKey()
        self.default_pos = pyflex.get_positions()

        if state is not None:
            self.set_state(state)

        self.current_config = deepcopy(config)

    def get_default_config(self):
        # cam_pos, cam_angle = np.array([0.0, 0.82, 0.00]), np.array([0, -np.pi/2., 0.])
        cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])

        config = {
            'pos': [0.01, 0.2, 0.01],
            'scale': -1,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': 1.0,
            'mass': 0.5 / (40 * 40),
            'radius': self.cloth_particle_radius,  # / 1.8,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.2, 0.7, 0]),
                                   'angle': np.array([np.pi/2, -np.pi/6, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height},
                              'top_down_camera_full': {
                                  'pos': np.array([0, 0.35, 0]),
                                  'angle': np.array([0, -90 / 180 * np.pi, 0]),
                                  'width': self.camera_width,
                                  'height': self.camera_height
                              },
                              },
            'drop_height': 0.0,
            'cloth_type': 0

        }

        return config

    def rotate_particles(self, angle):
        r = R.from_euler('zyx', angle, degrees=True)
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()[:, :3]
        new_pos = r.apply(new_pos)
        new_pos = np.column_stack([new_pos, pos[:, 3]])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _set_to_flat(self, pos=None, x_target=0, delta_z=0, rot_angle=0):
        if pos is None:
            pos = self.default_pos.copy()
        if self.cloth_type == 'shorts':
            with open(self.shorts_pkl_path, 'rb') as f:
                pos = pickle.load(f)
        curr_pos = pos.reshape((-1, 4))
        curr_pos[:,0] += x_target
        curr_pos[:,1] += delta_z
        rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                            [0, 1, 0],
                            [np.sin(rot_angle), 0, np.cos(rot_angle)]])
        curr_pos[:,:3] = (rot_mat @ curr_pos[:,:3].T).T

        pyflex.set_positions(curr_pos)
        # if self.cloth_type != 'shorts':
        #     self.rotate_particles([0, 0, 90])
        pyflex.step()
        return self._get_current_covered_area(pos)

    def move_to_pos(self, new_pos):
        # TODO
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos[:, :3] -= center[:3]
        pos[:, :3] += np.asarray(new_pos)
        pyflex.set_positions(pos)
    def _get_drop_point_idx(self):
        return np.array([1075,1073])

    def _get_key_point_idx(self):
        return np.array([1075,1073, 2532, 2460,436,2404,2308,884,1172,936])

    def _add_shape(self, x_target, cloth_dimx, rot_angle, env_shape):

        if env_shape == 'platform':

            shape_size = np.array([0.2, 0.02, 0.2])
            shape_pos = np.array([x_target + cloth_dimx * self.cloth_particle_radius / 2, 0, 0])
            shape_quat = np.array([0, -np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])

            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                                [0, 1, 0],
                                [np.sin(rot_angle), 0, np.cos(rot_angle)]])
            shape_pos = (rot_mat @ shape_pos.T).T
            self._add_box(box_size=shape_size, box_pos=shape_pos, box_quat=shape_quat)

        elif env_shape == 'sphere':
            shape_size = [0.1,0.1,0.1]
            shape_pos = np.array([x_target + cloth_dimx * self.cloth_particle_radius / 2, 0, 0])
            shape_quat = np.array([0., 0., 0., 1.])

            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                                [0, 1, 0],
                                [np.sin(rot_angle), 0, np.cos(rot_angle)]])
            shape_pos = (rot_mat @ shape_pos.reshape([-1,3]).T).T

            self._add_sphere(sphere_radius=shape_size[0], sphere_position=shape_pos,
                                                 sphere_quat=shape_quat)

        elif env_shape == 'rod':

            shape_size = [0.01, 0.25]
            shape_pos = np.array([x_target + cloth_dimx * self.cloth_particle_radius / 2, 0.1, 0])
            shape_quat = np.array([0, -np.sin((rot_angle + np.pi/2) / 2), 0, np.cos((rot_angle + np.pi/2) / 2)])

            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                                [0, 1, 0],
                                [np.sin(rot_angle), 0, np.cos(rot_angle)]])
            shape_pos = (rot_mat @ shape_pos.T).T

            self._add_rod(rod_size=shape_size, rod_position=shape_pos,
                                           rod_quat=shape_quat)

        elif env_shape =='table':

            shape_size = np.array([0.1, 0.02, 0.1])
            shape_pos = np.array([x_target + cloth_dimx * self.cloth_particle_radius / 2, 0.1, 0])
            shape_quat = np.array([0, -np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])

            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                                [0, 1, 0],
                                [np.sin(rot_angle), 0, np.cos(rot_angle)]])
            shape_pos = (rot_mat @ shape_pos.T).T

            self._add_table(table_size=shape_size, table_pos=shape_pos, table_quat=shape_quat)

        else:
            raise NotImplementedError


        return shape_size, shape_pos, shape_quat, env_shape

    def _add_table(self, table_size, table_pos, table_quat):

        rot_angle = np.arccos(table_quat[3])*2
        rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                            [0, 1, 0],
                            [np.sin(rot_angle), 0, np.cos(rot_angle)]])
        table_pos_inv = (rot_mat.T @ table_pos.reshape([-1,3]).T).T[0]
        rod_size = [0.02, table_pos[1]]
        bias = table_size[0] * 0.8
        rod_pos_1 = table_pos_inv + np.array([bias, -table_pos_inv[1], bias])
        rod_pos_2 = table_pos_inv + np.array([-bias, -table_pos_inv[1], bias])
        rod_pos_3 = table_pos_inv + np.array([-bias, -table_pos_inv[1], -bias])
        rod_pos_4 = table_pos_inv + np.array([bias, -table_pos_inv[1], -bias])

        rod_quat = np.array([0, 0, -np.sin(np.pi / 4), np.cos(np.pi / 4)])

        self._add_box(box_size=table_size, box_pos=table_pos, box_quat=table_quat)

        # add four rod
        pyflex.add_capsule(rod_size, (rot_mat @ rod_pos_1.T).T, rod_quat)
        pyflex.add_capsule(rod_size, (rot_mat @ rod_pos_2.T).T, rod_quat)
        pyflex.add_capsule(rod_size, (rot_mat @ rod_pos_3.T).T, rod_quat)
        pyflex.add_capsule(rod_size, (rot_mat @ rod_pos_4.T).T, rod_quat)


    def _add_rod(self, rod_size, rod_position, rod_quat):

        pyflex.add_capsule(rod_size, rod_position, rod_quat)

        return True

    def _add_box(self, box_size, box_pos, box_quat):

        pyflex.add_box(box_size, box_pos, box_quat)
        self._add_capsules_tobox(box_size, box_pos, box_quat)
        return True

    def _add_sphere(self, sphere_radius, sphere_position, sphere_quat):

        pyflex.add_sphere(sphere_radius, sphere_position, sphere_quat)

        return True

    def _add_capsules_tobox(self, box_size, box_pos, box_quat):

        # add_capsule
        params = np.array([box_size[1]+0.001, box_size[0]])

        initial_quat = R.from_quat(box_quat)
        rpy = initial_quat.as_euler('xyz', degrees=False)
        rotation_angle = -rpy[1]

        # add four capsules to the box
        lower_position = box_pos + np.array([box_size[0]*np.cos(rotation_angle), 0., box_size[0]*np.sin(rotation_angle)])
        z_rot_quat = R.from_euler('y', 90, degrees=True)
        result_quat = z_rot_quat * initial_quat
        quat_capsule = result_quat.as_quat()
        pyflex.add_capsule(params, lower_position, quat_capsule)

        lower_position = box_pos - np.array([box_size[0]*np.cos(rotation_angle), 0., box_size[0]*np.sin(rotation_angle)])
        pyflex.add_capsule(params, lower_position, quat_capsule)

        lower_position = box_pos + np.array([-box_size[0]*np.sin(rotation_angle), 0., box_size[0]*np.cos(rotation_angle)])
        pyflex.add_capsule(params, lower_position, box_quat)

        lower_position = box_pos + np.array([box_size[0]*np.sin(rotation_angle), 0., -box_size[0]*np.cos(rotation_angle)])
        pyflex.add_capsule(params, lower_position, box_quat)


if __name__ == '__main__':
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import copy
    import cv2


    def prepare_policy(env):
        print("preparing policy! ", flush=True)

        # move one of the picker to be under ground
        shape_states = pyflex.get_shape_states().reshape(-1, 14)
        shape_states[1, :3] = -1
        shape_states[1, 7:10] = -1

        # move another picker to be above the cloth
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
        shape_states[0, 7:10] = pos[pp] + [0., 0.06, 0.]
        pyflex.set_shape_states(shape_states.flatten())


    env_name = 'TshirtFlatten'
    env_args = copy.deepcopy(env_arg_dict[env_name])
    env_args['render_mode'] = 'cloth'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 720
    env_args['camera_width'] = 720
    env_args['camera_name'] = 'default_camera'
    env_args['headless'] = False
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['cached_states_path'] = 'tshirt_flatten_init_states_small_2021_05_28_01_16.pkl'
    env_args['num_variations'] = 2
    env_args['use_cached_states'] = True
    env_args['save_cached_states'] = False
    env_args['cloth_type'] = 'tshirt-small'
    # pkl_path = './softgym/cached_initial_states/shorts_flatten.pkl'

    env = SOFTGYM_ENVS[env_name](**env_args)
    print("before reset")
    env.reset()
    print("after reset")
    env._set_to_flat()
    print("after reset")
    # env.move_to_pos([0, 0.1, 0])
    # pyflex.step()
    # i = 0
    # import pickle

    # while (1):
    #     pyflex.step(render=True)
    #     if i % 500 == 0:
    #         print('saving pkl to ' + pkl_path)
    #         pos = pyflex.get_positions()
    #         with open(pkl_path, 'wb') as f:
    #             pickle.dump(pos, f)
    #     i += 1
    #     print(i)

    obs = env._get_obs()
    cv2.imwrite('./small_tshirt.png', obs)
    # cv2.imshow('obs', obs)
    # cv2.waitKey()

    prepare_policy(env)

    particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    n_particles = particle_positions.shape[0]
    # p_idx = np.random.randint(0, n_particles)
    # p_idx = 100
    pos = particle_positions
    ok = False
    while not ok:
        pp = np.random.randint(len(pos))
        if np.any(np.logical_and(np.logical_and(np.abs(pos[:, 0] - pos[pp][0]) < 0.00625, np.abs(pos[:, 2] - pos[pp][2]) < 0.00625),
                                 pos[:, 1] > pos[pp][1])):
            ok = False
        else:
            ok = True
    picker_pos = particle_positions[pp] + [0, 0.01, 0]

    timestep = 50
    movement = np.random.uniform(0, 1, size=(3)) * 0.4 / timestep
    movement = np.array([0.2, 0.2, 0.2]) / timestep
    action = np.zeros((timestep, 8))
    action[:, 3] = 1
    action[:, :3] = movement

    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 7:10] = -1

    shape_states[0, :3] = picker_pos
    shape_states[0, 7:10] = picker_pos

    pyflex.set_shape_states(shape_states)
    pyflex.step()

    obs_list = []

    for a in action:
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    for t in range(30):
        a = np.zeros(8)
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    from softgym.utils.visualization import save_numpy_as_gif

    save_numpy_as_gif(np.array(obs_list), '{}.gif'.format(
        env_args['cloth_type']
    ))
