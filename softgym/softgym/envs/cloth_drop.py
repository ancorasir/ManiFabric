import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon

class ClothDropEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_drop_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        self.vary_cloth_size = kwargs['vary_cloth_size']
        self.vary_stiffness = kwargs['vary_stiffness']
        self.vary_orientation = kwargs['vary_orientation']
        self.vary_mass = kwargs['vary_mass']
        self.env_shape = kwargs['env_shape']
        if 'target_type' in kwargs:
            self.target_type = kwargs['target_type']
        else:
            self.target_type = None
        # self.particle_radius = kwargs['particle_radius']

        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized

    def generate_env_variation(self, num_variations=1):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.05  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()


        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

            if self.vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            if self.vary_mass:
                config['mass'] = np.random.uniform(0.02, 0.2)

            if self.vary_stiffness:
                config['ClothStiff'] = [np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0),
                                                np.random.uniform(0.5, 2.0)]

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            if self.vary_orientation:
                rot_angle = np.random.uniform(-np.pi/6, np.pi/6)
            else:
                rot_angle = 0

            x_target = np.random.uniform(0.05, 0.15)
            z_target = 0

            if self.env_shape == 'random':
                env_shape_list = [None, 'platform', 'sphere', 'rod']
                env_shape = env_shape_list[i % 4]
            elif self.env_shape == 'all':
                env_shape_list = [None, 'platform', 'sphere', 'rod', 'table']
                env_shape = env_shape_list[i % 5]
            else:
                env_shape = self.env_shape

            if env_shape is not None:

                config['shape_size'], config['shape_pos'], config['shape_quat'], config['env_shape'] = self._add_shape(x_target=x_target, cloth_dimx=cloth_dimx, rot_angle=rot_angle, env_shape=env_shape)
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

            ## Set the cloth to target position and wait to stablize
            flat_pos = self._set_to_flat(x_target=x_target, delta_z= z_target, rot_angle=rot_angle)

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            picker_pos = self.set_picker_wait(pickpoints, max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold,
                                              grasped=False)

            config['target_picker_pos'] = picker_pos
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            config['target_pos'] = curr_pos[:, :3]

            ## Set the cloth to initial position and wait to stablize
            self._set_to_vertical(x_low=0, height_low=np.random.uniform(0.1, 0.15), height_high= 0)

            self.set_picker_wait(pickpoints, max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['ClothStiff']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_picker_wait(self, pickpoints, max_wait_step=500, stable_vel_threshold=0.1, grasped=True):

        curr_pos = pyflex.get_positions().reshape(-1, 4)
        # curr_pos[0] += np.random.random() * 0.001  # Add small jittering
        if grasped:
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
        pickpoint_pos = curr_pos[pickpoints, :3]
        pyflex.set_positions(curr_pos.flatten())

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
    def _set_to_flat(self, x_target=0, delta_y=0, delta_z=0, rot_angle=0):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        if self.target_type =='fold':
            flat_pos = self._get_flat_pos(x_target=x_target, delta_z=delta_z, rot_angle=rot_angle, fold=True)
        else:
            flat_pos = self._get_flat_pos(x_target=x_target, delta_z=delta_z, rot_angle=rot_angle)
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

        return flat_pos

    def _get_flat_pos(self, x_target=0, delta_y=0, delta_z=0, rot_angle=0, fold=False):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        x += x_target
        # print(x.mean(),delta_x)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3 + delta_z  # Set specifally for particle radius of 0.00625

        if fold:
            xx = xx.flatten()
            curr_pos[xx < np.mean(xx), 1] += 0.00625
            xx[xx>np.mean(xx)] = np.mean(xx)- (xx[xx>np.mean(xx)]-np.mean(xx))
            curr_pos[:, 0] = xx

        # Rotate the cloth
        rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)],
                            [0, 1, 0],
                            [np.sin(rot_angle), 0, np.cos(rot_angle)]])
        curr_pos = (rot_mat @ curr_pos.T).T
        return curr_pos

    def _set_to_vertical(self, x_low, height_low, height_high):
        if self.env_shape == 'rod':
            height_low+=0.1
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < height_high:
            curr_pos[:, 1] += height_high - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
            # self.action_tool.visualize_picker_boundary()
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
        original_inv_mass = self.action_tool.step(action)
        pyflex.step()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[:, 3] = original_inv_mass
        pyflex.set_positions(curr_pos.flatten())

        return

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        r = - curr_dist
        return r

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

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [0, 0, 0],
            'ClothSize': [48, 48],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            # 'camera_params': {'default_camera':
            #                       {'pos': np.array([1.07199, 0.94942, 1.15691]),
            #                        'angle': np.array([0.633549, -0.397932, 0]),
            #                        'width': self.camera_width,
            #                        'height': self.camera_height}},
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.2, 0.7, 0]),
                                   'angle': np.array([np.pi/2, -np.pi/6, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},

            'flip_mesh': 0,
            'mass': 0.1
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _sample_cloth_size(self):
        return np.random.randint(32, 48), np.random.randint(32, 48)

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
