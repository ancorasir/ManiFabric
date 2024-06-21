import numpy as np
from multiprocessing import pool
import copy
from ADFM.utils.camera_utils import project_to_image, get_target_pos


class MPCPlanner():

    def __init__(self,
                    dynamics, reward_model, normalize_info=None,
                    matrix_world_to_camera=np.identity(4),
                    use_pred_rwd=False, env=None, args=None):
        """
        Random Shooting planner.
        """

        self.normalize_info = normalize_info  # Used for robot experiments to denormalize before action clipping
        self.shooting_number = args.shooting_number
        self.reward_model = reward_model
        self.dynamics = dynamics
        self.gpu_num = args.gpu_num
        self.use_pred_rwd = use_pred_rwd

        num_worker = args.num_worker
        if num_worker > 0:
            self.pool = pool.Pool(processes=num_worker)

        self.num_worker = num_worker
        self.matrix_world_to_camera = matrix_world_to_camera
        self.image_size = (env.camera_height, env.camera_width)

        self.dt = args.dt
        self.env = env
        self.args = args
        self.pred_time_interval = args.pred_time_interval

        self.swing_acc = 2.0
        self.pull_acc = 1.0

        self.picker_traj = None
        self.delta_actions = np.array([[0.1, 0.1, 0.1], [0.1, 0, 0.1], [0.1, -0.1, 0.1],
                                       [0, 0.1, 0], [0, 0, 0], [0, -0.1, 0],
                                       [-0.1, 0.1, -0.1], [-0.1, 0, -0.1], [-0.1, -0.1, -0.1]])
        # self.delta_actions = self.delta_actions*2
        self.shooting_num = len(self.delta_actions)

    def init_traj(self, data, sampling_num, robot_exp=False):

        picker_position = data['picker_position']
        target_position = data['target_picker_pos']
        actions_sampled, step_mid_sampled = [], []
        for _ in range(sampling_num):
            actions, step_mid = self._collect_trajectory(picker_position, target_position)
            actions = np.concatenate((actions, np.zeros((3, 8))), axis=0)
            actions_sampled.append(actions)
            step_mid_sampled.append(step_mid)

        data_cpy = copy.deepcopy(data)

        returns, results = [], []
        for i in range(sampling_num):
            res = self.dynamics.rollout(
                dict(
                    model_input_data=copy.deepcopy(data_cpy), actions=actions_sampled[i],
                    reward_model=self.reward_model, cuda_idx=0, robot_exp=robot_exp,
                )
            )
            results.append(res), returns.append(res['final_ret'])

        highest_return_idx = np.argmax(returns)
        self.actions = actions_sampled[highest_return_idx]
        self.step_mid = step_mid_sampled[highest_return_idx]

        pred_positions = results[highest_return_idx]['model_positions']
        pred_edges = results[highest_return_idx]['mesh_edges']
        pred_shapes = results[highest_return_idx]['shape_positions']

        self.picker_traj = pred_shapes.copy()

        print('Trajectory initialized.')
        # How to evaluate this middle state? add noise to the dynamics model..  or add noise to the action sequence?
        # Todo: delete results after plotting
        return pred_positions, pred_edges, pred_shapes, returns, results

    def update_traj(self, actions, pred_shape, control_seq_idx):
        self.actions = np.concatenate((self.actions[:control_seq_idx], actions))
        self.picker_traj = np.concatenate((self.picker_traj[:control_seq_idx], pred_shape))

    def get_action(self, init_data, control_seq_idx=0,  robot_exp=False, m_name='vsbl'):
        """
        check_mask: Used to filter out place points that are on the cloth.
        init_data should be a list that include:
            ['pointcloud', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'observable_particle_indices]
            note: require position, velocity to be already downsampled

        """
        data = init_data.copy()
        target_picker_pos = self.env.get_current_config()['target_picker_pos']

        # maybe stop here or continue?

        # paralleled version of generating action sequences
        if robot_exp:
            raise NotImplementedError
        else:  # simulation planning

            if control_seq_idx < self.step_mid:
                actions_swing = self.actions[control_seq_idx:self.step_mid]

                ## add delta action to actions_swing

                # extend one dimension
                actions_swing_expanded = np.expand_dims(actions_swing, 0).repeat(self.shooting_num, axis=0)
                for i, delta in enumerate(self.delta_actions):
                    actions_swing_expanded[i, :, :3] += actions_swing[:, :3] * delta
                    actions_swing_expanded[i, :, 4:7] += actions_swing[:, 4:7] * delta

                # Calculate assumed positions for both pickers
                assumed_mid_pos_1 = data['picker_position'][0] + np.sum(actions_swing_expanded[:, :, :3], axis=1)
                assumed_mid_pos_2 = data['picker_position'][1] + np.sum(actions_swing_expanded[:, :, 4:7], axis=1)
                assumed_mid_pos = np.stack((assumed_mid_pos_1, assumed_mid_pos_2), axis=-2)

                # Generate pull actions
                actions_pull_list = [self._generate_pull_actions(mid_pos, target_picker_pos) for mid_pos in
                                     assumed_mid_pos]

                # Combine swing and pull actions
                actions = [np.concatenate((actions_swing_expanded[i], pull_actions, [np.zeros(8) for _ in range(3)]))
                                    for i, pull_actions in enumerate(actions_pull_list)]
            else:
                actions = self.actions[control_seq_idx:]
                actions = np.expand_dims(actions, axis=0)

        # parallely rollout the dynamics model with the sampled action seqeunces
        data_cpy = copy.deepcopy(data)
        if self.num_worker > 0:
            job_each_gpu = self.shooting_number // self.gpu_num
            params = []
            for i in range(self.shooting_number):

                gpu_id = i // job_each_gpu if i < self.gpu_num * job_each_gpu else i % self.gpu_num
                params.append(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=gpu_id, robot_exp=robot_exp,
                    )
                )
            results = self.pool.map(self.dynamics.rollout, params, chunksize=max(1, self.shooting_number // self.num_worker))
            returns = [x['final_ret'] for x in results]
        else: # sequentially rollout each sampled action trajectory
            returns, results = [], []
            for i in range(len(actions)):
                assert actions[i].shape[-1] == 8
                res = self.dynamics.rollout(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=0, robot_exp=robot_exp,
                    )
                )
                results.append(res), returns.append(res['final_ret'])

        highest_return_idx = np.argmax(returns)

        highest_return = returns[highest_return_idx]
        action_seq = actions[highest_return_idx]

        pred_positions = results[highest_return_idx]['model_positions']
        pred_edges = results[highest_return_idx]['mesh_edges']
        pred_shape = results[highest_return_idx]['shape_positions']
        print('highest_return_idx', highest_return_idx)

        self.update_traj(action_seq, pred_shape, control_seq_idx=control_seq_idx)

        return action_seq, pred_positions, highest_return, pred_edges, pred_shape

    def _generate_pull_actions(self, assumed_mid_pos, target_picker_pos):
        traj_pull = self._generate_trajectory(assumed_mid_pos, target_picker_pos, self.pull_acc,
                                              self.dt * self.pred_time_interval)

        actions_pull = [np.ones(8, dtype=np.float32) for _ in range(len(traj_pull) - 1)]
        for i, action in enumerate(actions_pull):
            action[:3], action[4:7] = traj_pull[i + 1][0] - traj_pull[i][0], traj_pull[i + 1][1] - traj_pull[i][1]

        return actions_pull

    def get_action_real(self, init_data, control_seq_idx=0,  robot_exp=True, m_name='vsbl'):

        data = init_data.copy()
        target_picker_pos = data['target_picker_pos']

        if control_seq_idx < self.step_mid:
            actions_swing = self.actions[control_seq_idx:self.step_mid]
            # Todo: get a better action sampling method

            ## add noise to actions_swing
            noise_ratio = np.random.normal(0, 0.2, [self.shooting_number, 3])
            delta_action_list_1 = [actions_swing[1:, :3] * noise_ratio[i] for i in range(self.shooting_number)]
            delta_action_list_2 = [actions_swing[1:, 4:7] * noise_ratio[i] for i in range(self.shooting_number)]

            # extend one dimension
            actions_swing = np.expand_dims(actions_swing, axis=0).repeat(self.shooting_number, axis=0)

            for i in range(self.shooting_number):
                actions_swing[i, 1:, :3] = actions_swing[i, 1:, :3] + delta_action_list_1[i]
                actions_swing[i, 1:, 4:7] = actions_swing[i, 1:, 4:7] + delta_action_list_2[i]

            assumed_mid_pos_1 = data['picker_position'][0] + np.sum(actions_swing[:, :, :3], axis=1)
            assumed_mid_pos_2 = data['picker_position'][1] + np.sum(actions_swing[:, :, 4:7], axis=1)
            assumed_mid_pos = np.concatenate((assumed_mid_pos_1, assumed_mid_pos_2), axis=1)

            actions_pull_list = []
            for i in range(self.shooting_number):
                traj_pull = self._generate_trajectory(assumed_mid_pos[i].reshape(-1, 3), target_picker_pos,
                                                      self.pull_acc, self.dt * self.pred_time_interval)

                actions_pull = []
                for step in range(len(traj_pull) - 1):
                    action = np.ones(8, dtype=np.float32)
                    action[:3], action[4:7] = traj_pull[step + 1][0] - traj_pull[step][0], traj_pull[step + 1][1] - \
                                              traj_pull[step][1]
                    actions_pull.append(action)

                actions_pull_list.append(actions_pull)
            actions = []
            for i in range(self.shooting_number):
                actions.append(np.array(actions_swing[i].tolist() + actions_pull_list[i]))
        else:
            actions = self.actions[control_seq_idx:]
            actions = np.expand_dims(actions, axis=0).repeat(self.shooting_number, axis=0)

        # parallely rollout the dynamics model with the sampled action seqeunces
        data_cpy = copy.deepcopy(data)
        if self.num_worker > 0:
            job_each_gpu = self.shooting_number // self.gpu_num
            params = []
            for i in range(self.shooting_number):

                gpu_id = i // job_each_gpu if i < self.gpu_num * job_each_gpu else i % self.gpu_num
                params.append(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=gpu_id, robot_exp=robot_exp,
                    )
                )
            results = self.pool.map(self.dynamics.rollout, params, chunksize=max(1, self.shooting_number // self.num_worker))
            returns = [x['final_ret'] for x in results]
        else: # sequentially rollout each sampled action trajectory
            returns, results = [], []
            for i in range(self.shooting_number):
                assert actions[i].shape[-1] == 8
                res = self.dynamics.rollout(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=0, robot_exp=robot_exp,
                    )
                )
                results.append(res), returns.append(res['final_ret'])

        highest_return_idx = np.argmax(returns)

        highest_return = returns[highest_return_idx]
        action_seq = actions[highest_return_idx]

        pred_positions = results[highest_return_idx]['model_positions']
        pred_edges = results[highest_return_idx]['mesh_edges']
        pred_shape = results[highest_return_idx]['shape_positions']
        print('highest_return_idx', highest_return_idx)

        self.update_traj(action_seq, control_seq_idx=control_seq_idx)

        return action_seq, pred_positions, highest_return, pred_edges, pred_shape

    def _collect_trajectory(self, current_picker_position, target_picker_position, xy_trans= None, z_ratio=None):

        """ Policy for collecting data - random sampling"""
        if xy_trans == None:
            xy_trans = np.random.uniform(0.1, 0.25)
        if z_ratio == None:
            z_ratio = np.random.uniform(0.1, 0.6)

        norm_direction = np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))
        middle_state = target_picker_position.copy()
        middle_state[:, [0, 2]] = target_picker_position[:, [0, 2]] + xy_trans * norm_direction
        middle_state[:, 1] = current_picker_position[:, 1] + z_ratio * (
                target_picker_position[:, 1] - current_picker_position[:, 1])

        trajectory_start_to_middle = self._generate_trajectory(current_picker_position, middle_state,
                                                               self.swing_acc, self.dt * self.pred_time_interval)
        s2m_steps = len(trajectory_start_to_middle) -1
        trajectory_middle_to_target = self._generate_trajectory(middle_state, target_picker_position,
                                                                self.pull_acc, self.dt * self.pred_time_interval)

        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        action_list = []
        for step in range(1, trajectory.shape[0]):
            action = np.ones(8, dtype=np.float32)
            action[:3], action[4:7] = trajectory[step, :3] - trajectory[step - 1, :3], trajectory[step,
                                                                                       3:6] - trajectory[step - 1, 3:6]
            action_list.append(action)

        action_list = np.array(action_list)
        action_list[:, [3, 7]] = 1
        return action_list, s2m_steps

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

        if accel_steps is 0:
            raise ValueError('accel_steps is 0')

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
                                                     dt) + accelerate * dt) * dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     dt) + decelerate * dt) * dt

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

def pos_in_image(after_pos, matrix_world_to_camera, image_size):
    euv = project_to_image(matrix_world_to_camera, after_pos.reshape((1, 3)), image_size[0], image_size[1])
    u, v = euv[0][0], euv[1][0]
    if u >= 0 and u < image_size[1] and v >= 0 and v < image_size[0]:
        return True
    else:
        return False

def project_3d(self, pos):
    return project_to_image(self.matrix_world_to_camera, pos, self.image_size[0], self.image_size[1])

def rollout_gt(env, episode_idx, actions, pred_time_interval):
    # rollout the env in simulation
    env.reset(config_id=episode_idx)
    actions_executed = []
    for action in actions:
        ac_exe = np.zeros((pred_time_interval, 8))
        ac_exe[:, :] = (action[:]) / pred_time_interval

        ac_exe[ac_exe[:, 3] > 0, 3] = 1
        ac_exe[ac_exe[:, 7] > 0, 7] = 1
        actions_executed.extend(ac_exe)

    frames, frames_top = [], []
    for action in actions_executed:
        env.action_tool.picked_particles = env._get_drop_point_idx()
        _, reward, done, info = env.step(action, record_continuous_video=True, img_size=360)

        frames.extend(info['flex_env_recorded_frames'])
        frames_top.append(info['image_top'])

    return frames, frames_top, reward