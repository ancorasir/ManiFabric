import os
import scipy
import numpy as np
import torch

from torch_geometric.data import Dataset

from ADFM.utils.utils import downsample, load_data, load_data_list, store_h5_data, voxelize_pointcloud, cloth_drop_reward_fuc, draw_target_pos
from ADFM.utils.camera_utils import get_observable_particle_index, get_observable_particle_index_old, get_world_coords, get_observable_particle_index_3, get_matrix_world_to_camera
from ADFM.utils.data_utils import PrivilData
import scipy.spatial.transform as sst

class ClothDataset(Dataset):
    def __init__(self, args, input_types, phase, env, train_mode):
        super(ClothDataset).__init__()
        self.input_types = input_types
        self.train_mode = train_mode
        self.args = args
        self.phase = phase
        self.env = env
        if self.args.dataf is not None:
            self.data_dir = os.path.join(self.args.dataf, phase)
            os.system('mkdir -p ' + self.data_dir)

            ## list all directories in the data_dir and count the data in each directory
            self.all_rollouts = os.listdir(self.data_dir)
            if len(self.all_rollouts) is not 0:
                if 'mid_steps.npy' in self.all_rollouts:
                    self.all_rollouts.pop(self.all_rollouts.index('mid_steps.npy'))
                self.all_rollouts = sorted(self.all_rollouts, key=lambda x: int(x))
                self.num_traj = len(self.all_rollouts)

                rollout_lengths, self.rollout_steps = [],[]
                for rollout_num in self.all_rollouts:
                    rollout_dir = os.path.join(self.data_dir, rollout_num)

                    # count the number of data in each rollout if the first character is a digit
                    rollout_data = [data for data in os.listdir(rollout_dir) if data[0].isdigit()]
                    rollout_lengths.append(len(rollout_data) - self.args.pred_time_interval)
                    self.rollout_steps.append(len(rollout_data))

                self.cumulative_lengths = np.cumsum(rollout_lengths)
            else:
                self.cumulative_lengths=[0]
        else:

            raise ValueError('No data directory is provided')
        self.num_workers = args.num_workers
        self.dt = args.dt
        self.use_fixed_observable_idx = False

        ratio = self.args.train_valid_ratio

        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = int(self.args.n_rollout - int(self.args.n_rollout * ratio))
        else:
            raise AssertionError("Unknown phase")

        self.all_trajs = []

        self.data_names = None
        self.edge = None
        self.skipped = 0

        # self.dataset_check()
    def __getitem__(self, idx):
        all_input = {}
        ori_data = self.prepare_transition(idx, eval=self.phase == 'valid')
        for input_type in self.input_types:
            suffix = '_' + input_type
            data = self.remove_suffix(ori_data, input_type)
            d = self.build_graph(data, input_type=input_type)
            node_attr, neighbors, edge_attr = d['node_attr'], d['neighbors'], d['edge_attr']

            all_input.update({
                'x' + suffix: node_attr,
                'edge_index' + suffix: neighbors,
                'edge_attr' + suffix: edge_attr,
                'gt_accel' + suffix: data['gt_accel'],
                'gt_vel' + suffix: data['gt_vel'],
                'gt_reward_nxt' + suffix: data['gt_reward_nxt']
            })
            if self.train_mode == 'graph_imit' and input_type == 'full':
                all_input.update({'partial_pc_mapped_idx' + suffix: torch.as_tensor(data['partial_pc_mapped_idx'], dtype=torch.long)})
        data = PrivilData.from_dict(all_input)
        return data

    def prepare_transition(self, idx, eval=False):
        """
        Return the raw input for both full and partial point cloud.
        Noise augmentation only support when fd_input = True
        Two modes for input and two modes for output:
            self.args.fd_input = True:
                Calculate vel his by 5-step finite differences
            else:
                Retrieve vel from dataset, which is obtained by 1-step finite differences.
            self.args.fd_output = True:
                Calculate vel_nxt by 5-step finite differences
            else:
                Calculate vel_nxt by retrieving one-step vel at 5 timesteps later.
        """

        pred_time_interval = self.args.pred_time_interval

        while True:

            idx_rollout = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx)
            if idx_rollout > 0:
                idx_timestep = idx - self.cumulative_lengths[idx_rollout - 1]
            else:
                idx_timestep = idx
            # idx_rollout = (idx // (self.args.time_step - self.args.n_his)) % self.n_rollout
            # idx_timestep = max((self.args.n_his - pred_time_interval) + idx % (self.args.time_step - self.args.n_his), 0)

            data_rollout = load_data(self.data_dir, idx_rollout, 'rollout_info', None)
            data_cur = load_data(self.data_dir, idx_rollout, idx_timestep, self.data_names)
            data_nxt = load_data(self.data_dir, idx_rollout, idx_timestep + pred_time_interval, self.data_names)

            vox_pc = data_cur['pointcloud'] # voxelized point cloud

            # vox_pc = voxelize_pointcloud(pointcloud, self.args.voxel_size)
            partial_particle_pos = data_cur['positions'][data_rollout['downsample_idx']][data_cur['downsample_observable_idx']]
            if len(vox_pc) <= len(partial_particle_pos):

                break
            else:
                break

                #  for bi-partite graph matching, not used for now

                self.skipped += 1
                print('Skip idx_rollout: {}, idx_timestep: {}, vox_pc len:{}, partical pos:{}, skipped:{}'.format(idx_rollout, idx_timestep, len(vox_pc), len(partial_particle_pos),self.skipped))
                idx += (self.cumulative_lengths[idx_rollout] - self.cumulative_lengths[idx_rollout-1])

        # accumulate action if we need to predict multiple steps
        action = np.ones(4 * self.args.num_picker)

        for i in range(self.args.num_picker):
            action[i*4:i*4+3] = data_nxt['picker_position'][i,:3] - data_cur['picker_position'][i,:3]

        # Use clean observable point cloud for bi-partite matching
        # particle_pc_mapped_idx: For each point in pc, give the index of the closest point on the visible downsample mesh
        # shape of vox_pc may change
        vox_pc, partial_pc_mapped_idx = get_observable_particle_index_3(vox_pc, partial_particle_pos, threshold=self.args.voxel_size)
        partial_pc_mapped_idx = data_cur['downsample_observable_idx'][
            partial_pc_mapped_idx]  # Map index from the observable downsampled mesh to the downsampled mesh

        # velocity calculation by multi-step finite differences
        # for n_his = n, we need n+1 velocities(including target), and n+2 position
        # full_pos_list: [p(t-25), ... p(t-5), p(t), p(t+5)]
        # full_vel_list: [v(t-20), ... v(t), v(t+5)], v(t) = (p(t) - p(t-5)) / (5*dt)

        downsample_idx = data_rollout['downsample_idx']
        full_pos_cur, full_pos_nxt = data_cur['positions'], data_nxt['positions']
        full_pos_list, full_vel_list = [], []
        # when i minus 0, it will be 0, the interval will not be pred_time_interval; 0.1 incase pred_time_interval is 0
        time_interval = [max(1,min(i,pred_time_interval)) for i in range(idx_timestep - (self.args.n_his * (pred_time_interval)), idx_timestep + pred_time_interval*2, pred_time_interval)]
        for i in range(idx_timestep - self.args.n_his * pred_time_interval, idx_timestep, pred_time_interval):  # Load history data
            t_positions = load_data_list(self.data_dir, idx_rollout, max(0, i), ['positions'])[0]  # max just in case
            full_pos_list.append(t_positions)

        full_pos_list.extend([full_pos_cur, full_pos_nxt])

        # Finite difference
        for i in range(self.args.n_his + 1): full_vel_list.append((full_pos_list[i + 1] - full_pos_list[i]) / (self.args.dt * time_interval[i+1] ))

        # Get velocity history, remove target velocity (last one)
        full_vel_his = full_vel_list[:-1]
        partial_vel_his = [vel[downsample_idx][partial_pc_mapped_idx] for vel in full_vel_his]

        partial_vel_his = np.concatenate(partial_vel_his, axis=1)
        full_vel_his = np.concatenate(full_vel_his, axis=1)

        # Compute info for full cloth, used for IL
        full_gt_vel = torch.FloatTensor(full_vel_list[-1])
        full_gt_accel = torch.FloatTensor((full_vel_list[-1] - full_vel_list[-2]) / (self.args.dt * pred_time_interval))
        partial_gt_vel = full_gt_vel[downsample_idx][partial_pc_mapped_idx]
        partial_gt_accel = full_gt_accel[downsample_idx][partial_pc_mapped_idx]

        gt_reward_crt = torch.FloatTensor([cloth_drop_reward_fuc(full_pos_cur[downsample_idx], data_rollout['target_pos'][downsample_idx])]) if 'target_pos' in data_rollout else None
        gt_reward_nxt = torch.FloatTensor([cloth_drop_reward_fuc(full_pos_nxt[downsample_idx], data_rollout['target_pos'][downsample_idx])]) if 'target_pos' in data_rollout else None


        data = {'pointcloud_vsbl': vox_pc,
                'vel_his_vsbl': partial_vel_his,
                'gt_accel_vsbl': partial_gt_accel,
                'gt_vel_vsbl': partial_gt_vel,

                'pointcloud_full': full_pos_cur[downsample_idx],  # Full dynamics is trained on the downsampled mesh
                'vel_his_full': full_vel_his[downsample_idx],
                'gt_accel_full': full_gt_accel[downsample_idx],
                'gt_vel_full': full_gt_vel[downsample_idx],

                'gt_reward_crt': gt_reward_crt,
                'gt_reward_nxt': gt_reward_nxt,
                'idx_rollout': idx_rollout,
                'picker_position': data_cur['picker_position'],
                'action': action,
                'scene_params': data_rollout['scene_params'],
                'partial_pc_mapped_idx': partial_pc_mapped_idx,
                'target_pos': data_rollout['target_pos'] if 'target_pos' in data_rollout else None,}

        if self.args.env_shape is not None:
            if 'env_shape' in data_rollout.keys():
                data['env_shape'] = data_rollout['env_shape'].astype(str)
                data['shape_size'] = data_rollout['shape_size']
                data['shape_pos'] = data_rollout['shape_pos']
                data['shape_quat'] = data_rollout['shape_quat']


        if self.edge is not None:
            # TODO: support rest dist for full(well, maybe not necessary)
            self.edge.set_mode('eval')
            model_input_data = dict(
                scene_params=data_cur['scene_params'],
                pointcloud=pointcloud,
                cuda_idx=-1,
            )
            mesh_edges = self.edge.infer_mesh_edges(model_input_data)
            data['mesh_edges'] = mesh_edges

            if self.args.__dict__.get('use_rest_distance', False):
                print("computing rest distance", flush=True)
                # scene_params = data_cur[5]
                # _, cloth_xdim, cloth_ydim, _ = scene_params
                # _, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, self.args.down_sample_scale)
                # tri_mesh = self.get_triangle_mesh(downsample_x_dim, downsample_y_dim)
                # particle_pos = data_cur[0][downsample_idx]
                # bc, pc_tri_idx = self.register_pointcloud(pointcloud, particle_pos, tri_mesh)

                # print("computing rest distance from the mapped particles at the first time step")
                if self.args.use_cache:
                    data_init = self._load_data_from_cache(load_names, idx_rollout, 0)
                else:
                    data_path = os.path.join(self.data_dir, str(idx_rollout), '0.h5')
                    data_init = self._load_data_file(load_names, data_path)
                pc_pos_init = data_init[0][downsample_idx][observe_pc_cur].astype(np.float32)

                # pc_pos_init = self.get_interpolated_pc(bc, pc_tri_idx, particle_pos, tri_mesh).astype(np.float32)
                rest_dist = np.linalg.norm(pc_pos_init[mesh_edges[0, :]] - pc_pos_init[mesh_edges[1, :]], axis=-1)
                data['rest_dist'] = rest_dist

        if not eval:
            return data
        else:
            data['downsample_idx'] = data_rollout['downsample_idx']
            # data['observable_idx'] = data_cur['observable_idx'] # not used for now
            return data

    def build_graph(self, data, input_type, robot_exp=False):
        """
        data: positions, vel_history, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        global_feat: fixed, not used for now
        """

        vox_pc, velocity_his = data['pointcloud'], data['vel_his']
        picked_points, picked_status = self._find_and_update_picked_point(data, robot_exp=robot_exp)  # Return index of the picked point

        node_attr = self._compute_node_attr(vox_pc, picked_points, velocity_his, **data)
        edges, edge_attr = self._compute_edge_attr(input_type, data)

        return {'node_attr': node_attr,
                'neighbors': edges,
                'edge_attr': edge_attr,
                'picked_particles': picked_points,
                'picked_status': picked_status}

    def _find_and_update_picked_point(self, data, robot_exp):
        """ Directly change the position and velocity of the picked point so that the dynamics model understand the action"""
        picked_pos = []  # Position of the picked particle
        picked_velocity = []  # Velocity of the picked particle

        action = (data['action'] * self.args.action_repeat).reshape([-1, 4])  # scale to the real action

        vox_pc, picker_pos, velocity_his = data['pointcloud'], data['picker_position'], data['vel_his']

        picked_particles = [-1 for _ in picker_pos]
        pick_flag = action[:, 3] > 0.5
        new_picker_pos = picker_pos.copy()
        if robot_exp:

            for i in range(self.args.num_picker):
                new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]

                if pick_flag[i]:
                    if picked_particles[i] == -1:  # No particle is currently picked and thus need to select a particle to pick
                        dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), vox_pc[:, :3].reshape((-1, 3)))
                        picked_particles[i] = np.argmin(dists)

                    if picked_particles[i] != -1:
                        old_pos = vox_pc[picked_particles[i]]
                        new_pos = vox_pc[picked_particles[i]] + action[i, :3]
                        new_vel = (new_pos - old_pos) / (self.dt*self.args.pred_time_interval)

                        tmp_vel_history = (velocity_his[picked_particles[i]][3:]).copy()
                        velocity_his[picked_particles[i], :-3] = tmp_vel_history
                        velocity_his[picked_particles[i], -3:] = new_vel
                        vox_pc[picked_particles[i]] = new_pos

                        picked_velocity.append(velocity_his[picked_particles[i]])
                        picked_pos.append(new_pos)
                else:
                    picked_particles[i] = -1
        else:
            for i in range(self.args.num_picker):
                new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
                if pick_flag[i]:
                    if picked_particles[i] == -1:  # No particle is currently picked and thus need to select a particle to pick
                        dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), vox_pc[:, :3].reshape((-1, 3)))
                        # idx_dists = np.hstack([np.arange(vox_pc.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
                        # mask = dists.flatten() <= self.args.picker_threshold * self.args.down_sample_scale \
                        #        + self.args.picker_radius + self.args.particle_radius
                        # idx_dists = idx_dists[mask, :].reshape((-1, 2))
                        # if idx_dists.shape[0] > 0:
                        #     pick_id, pick_dist = None, None
                        #     for j in range(idx_dists.shape[0]):
                        #         if idx_dists[j, 0] not in picked_particles and (pick_id is None or idx_dists[j, 1] < pick_dist):
                        #             pick_id = idx_dists[j, 0]
                        #             pick_dist = idx_dists[j, 1]
                        #     if pick_id is not None:  # update picked particles
                        #         picked_particles[i] = int(pick_id)
                        #
                        # else:
                        #     picked_particles[i] = int(-1)
                        #     #rause error
                        #     raise ValueError('No particle is picked')

                        picked_particles[i] = np.argmin(dists)
                    # update the position and velocity of the picked particle
                    if picked_particles[i] != -1:
                        old_pos = vox_pc[picked_particles[i]]
                        new_pos = vox_pc[picked_particles[i]] + new_picker_pos[i, :] - picker_pos[i, :]
                        new_vel = (new_pos - old_pos) / (self.dt * self.args.pred_time_interval)
                        tmp_vel_history = velocity_his[picked_particles[i]][3:].copy()
                        velocity_his[picked_particles[i], :-3] = tmp_vel_history
                        velocity_his[picked_particles[i], -3:] = new_vel

                        vox_pc[picked_particles[i]] = new_pos

                        picked_velocity.append(velocity_his[picked_particles[i]])
                        picked_pos.append(new_pos)
                else:
                    picked_particles[i] = int(-1)
        picked_status = (picked_velocity, picked_pos, new_picker_pos)

        return picked_particles, picked_status

    def _compute_node_attr(self, vox_pc, picked_points, velocity_his, **kwargs):
        # picked particle [0, 1]
        # normal particle [1, 0]
        node_one_hot = np.zeros((len(vox_pc), 2), dtype=np.float32)
        node_one_hot[:, 0] = 1
        for picked in picked_points:
            if picked != -1:
                node_one_hot[picked, 0] = 0
                node_one_hot[picked, 1] = 1

        if "env_shape" in kwargs.keys() and kwargs['env_shape'] is not None and self.args.use_es == True:
            distance_to_shape, vector_to_shape = self._compute_distance_to_shape(vox_pc, kwargs['shape_pos'], kwargs['shape_size'], kwargs['shape_quat'], kwargs['env_shape'])
        else:
            distance_to_shape = torch.from_numpy(vox_pc[:, 1]).view((-1, 1))
            vector_to_shape = torch.zeros([distance_to_shape.shape[0],3])
            vector_to_shape[:, 1] = 1

        node_one_hot = torch.from_numpy(node_one_hot)
        node_attr = torch.from_numpy(velocity_his)
        node_attr = torch.cat([node_attr, distance_to_shape, vector_to_shape, node_one_hot], dim=1)
        return node_attr

    def _compute_distance_to_shape(self, vox_pc, shape_pos, shape_size, shape_quat, env_shape):

        # distance to env shape
        if env_shape == 'platform':
            box_position = shape_pos
            box_size = shape_size
            box_quat = shape_quat
            rot_matrix = sst.Rotation.from_quat(box_quat).as_matrix()
            local_point = np.dot(rot_matrix.T, (vox_pc - box_position).T).T
            clamped_point = np.maximum(np.minimum(local_point, box_size), -box_size)
            global_clamped_point = np.dot(rot_matrix, clamped_point.T).T + box_position

            vector_tobox = vox_pc - global_clamped_point

            # vector_tobox to dtype float32
            distance_to_box = np.linalg.norm(vector_tobox.astype(np.float32), axis=1, keepdims=True)
            vector_tobox = np.array([vector_tobox[i]/distance_to_box[i] if distance_to_box[i] != 0 else np.array([0,0,0]) for i in range(len(vox_pc))])

            #select the min distance between ground and platform
            distance_to_shape = torch.from_numpy(np.minimum(vox_pc[:, 1:2], distance_to_box))
            vector_to_shape = [vector_tobox[i] if distance_to_box[i] <=vox_pc[i,1] else np.array([0,1,0]) for i in range(len(vox_pc))]

        elif env_shape == 'sphere':
            sphere_position = shape_pos
            sphere_radius = shape_size[0]
            vector_to_sphere = (vox_pc - sphere_position).astype(np.float32)
            distance_to_sphere_center = np.linalg.norm(vector_to_sphere, axis=1, keepdims=True)
            distance_to_sphere = distance_to_sphere_center-sphere_radius
            vector_to_sphere = np.array([vector_to_sphere[i]/distance_to_sphere_center[i] if distance_to_sphere_center[i] != 0 else np.array([0,0,0]) for i in range(len(vox_pc))])

            vector_to_shape = [vector_to_sphere[i] if distance_to_sphere[i] <=vox_pc[i,1] else np.array([0,1,0]) for i in range(len(vox_pc))]
            distance_to_shape = torch.from_numpy(np.minimum(vox_pc[:, 1:2], distance_to_sphere))

        elif env_shape == 'rod':
            rod_position = shape_pos
            rod_size = shape_size[0]
            rod_quat = shape_quat
            rot_matrix = sst.Rotation.from_quat(rod_quat).as_matrix()
            rot_matrix_ = sst.Rotation.from_quat(np.array([0, -np.sin((-np.pi/2) / 2), 0, np.cos((-np.pi/2) / 2)])).as_matrix()
            rot_matrix  = np.dot(rot_matrix_, rot_matrix)
            local_point = np.dot(rot_matrix.T, (vox_pc - rod_position).T).T
            clamped_point = local_point.copy()
            clamped_point[:,:2] = 0
            global_clamped_point = np.dot(rot_matrix, clamped_point.T).T + rod_position

            vector_torod = vox_pc - global_clamped_point
            distance_to_rod_center = np.linalg.norm(vector_torod.astype(np.float32), axis=1, keepdims=True)

            # vector_tobox to dtype float32
            distance_to_rod = distance_to_rod_center - rod_size
            vector_torod = np.array([vector_torod[i]/distance_to_rod_center[i] if distance_to_rod_center[i] != 0 else np.array([0,0,0]) for i in range(len(vox_pc))])

            vector_to_shape = [vector_torod[i] if distance_to_rod[i] <=vox_pc[i,1] else np.array([0,1,0]) for i in range(len(vox_pc))]
            distance_to_shape = torch.from_numpy(np.minimum(vox_pc[:, 1:2], distance_to_rod))
        elif env_shape == 'table':
            box_position = shape_pos
            box_size = shape_size
            box_quat = shape_quat
            rot_matrix = sst.Rotation.from_quat(box_quat).as_matrix()
            local_point = np.dot(rot_matrix.T, (vox_pc - box_position).T).T
            clamped_point = np.maximum(np.minimum(local_point, box_size), -box_size)
            global_clamped_point = np.dot(rot_matrix, clamped_point.T).T + box_position

            vector_tobox = vox_pc - global_clamped_point

            # vector_tobox to dtype float32
            distance_to_box = np.linalg.norm(vector_tobox.astype(np.float32), axis=1, keepdims=True)
            vector_tobox = np.array([vector_tobox[i]/distance_to_box[i] if distance_to_box[i] != 0 else np.array([0,0,0]) for i in range(len(vox_pc))])

            #select the min distance between ground and platform
            distance_to_shape = torch.from_numpy(np.minimum(vox_pc[:, 1:2], distance_to_box))
            vector_to_shape = [vector_tobox[i] if distance_to_box[i] <=vox_pc[i,1] else np.array([0,1,0]) for i in range(len(vox_pc))]
        else:
            raise ValueError('Unknown env shape')
        vector_to_shape = np.array(vector_to_shape)
        vector_to_shape = torch.from_numpy(vector_to_shape.astype(np.float32))
        return distance_to_shape, vector_to_shape

    def _compute_edge_attr(self, input_type, data):
        ##### add env specific graph components
        ## Edge attributes:
        # [1, 0] Distance based neighbor
        # [0, 1] Mesh edges
        # Calculate undirected edge list and corresponding relative edge attributes (distance vector + magnitude)
        vox_pc, velocity_his, observable_particle_idx = data['pointcloud'], data['vel_his'], data['partial_pc_mapped_idx']
        _, cloth_xdim, cloth_ydim, _ = data['scene_params']
        rest_dist = data.get('rest_dist', None)

        point_tree = scipy.spatial.cKDTree(vox_pc)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = vox_pc[undirected_neighbors[0, :]] - vox_pc[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1)
            edge_attr = np.concatenate([edge_attr, edge_attr_reverse])
            num_distance_edges = edges.shape[1]
        else:
            num_distance_edges = 0

        # Build mesh edges -- both directions
        if self.args.use_mesh_edge:
            if 'mesh_edges' not in data or data['mesh_edges'] is None:
                if input_type == 'vsbl':
                    mesh_edges = self._get_eight_neighbor(cloth_xdim, cloth_ydim, observable_particle_idx)
                else:
                    mesh_edges = self._get_eight_neighbor(cloth_xdim, cloth_ydim)
                data['mesh_edges'] = mesh_edges  # Pass this back into input data
            else:
                mesh_edges = data['mesh_edges']

            mesh_dist_vec = vox_pc[mesh_edges[0, :]] - vox_pc[mesh_edges[1, :]]
            mesh_dist = np.linalg.norm(mesh_dist_vec, axis=1, keepdims=True)
            mesh_edge_attr = np.concatenate([mesh_dist_vec, mesh_dist], axis=1)
            num_mesh_edges = mesh_edges.shape[1]

            if self.args.use_rest_distance:
                if rest_dist is None:
                    if data.get('idx_rollout', None) is not None:  # training case, without using an edge model to get the mesh edges
                        idx_rollout = data['idx_rollout']
                        positions = load_data_list(self.data_dir, idx_rollout, 0, ['positions'])[0]
                        downsample_idx = load_data_list(self.data_dir, idx_rollout, 'rollout_info', ['downsample_idx'])[0]
                        if input_type == 'vsbl':
                            pc_pos_init = positions[downsample_idx][data['partial_pc_mapped_idx']].astype(np.float32)
                        else:
                            pc_pos_init = positions[downsample_idx].astype(np.float32)
                    else:  # rollout during training
                        assert 'initial_particle_pos' in data
                        pc_pos_init = data['initial_particle_pos']
                    rest_dist = np.linalg.norm(pc_pos_init[mesh_edges[0, :]] - pc_pos_init[mesh_edges[1, :]], axis=-1)

                # rollout during test case, rest_dist should already be computed outwards.
                rest_dist = rest_dist.reshape((-1, 1))
                displacement = mesh_dist.reshape((-1, 1)) - rest_dist
                mesh_edge_attr = np.concatenate([mesh_edge_attr, displacement.reshape(-1, 1)], axis=1)
                if num_distance_edges > 0:
                    edge_attr = np.concatenate([edge_attr, np.zeros((edge_attr.shape[0], 1), dtype=np.float32)], axis=1)

            # concatenate all edge attributes
            edge_attr = np.concatenate([edge_attr, mesh_edge_attr], axis=0) if num_distance_edges > 0 else mesh_edge_attr
            edge_attr, mesh_edges = torch.from_numpy(edge_attr), torch.from_numpy(mesh_edges)

            # Concatenate edge types
            edge_types = np.zeros((num_mesh_edges + num_distance_edges, 2), dtype=np.float32)
            edge_types[:num_distance_edges, 0] = 1.
            edge_types[num_distance_edges:, 1] = 1.
            edge_types = torch.from_numpy(edge_types)
            edge_attr = torch.cat([edge_attr, edge_types], dim=1)

            if num_distance_edges > 0:
                edges = torch.from_numpy(edges)
                edges = torch.cat([edges, mesh_edges], dim=1)
            else:
                edges = mesh_edges
        else:
            if num_distance_edges > 0:
                edges, edge_attr = torch.from_numpy(edges), torch.from_numpy(edge_attr)
            else:
                # manually add one edge for correct processing when there is no collision edges
                print("number of distance edges is 0! adding fake edges")
                edges = np.zeros((2, 2), dtype=np.uint8)
                edges[0][0] = 0
                edges[1][0] = 1
                edges[0][1] = 0
                edges[1][1] = 2
                edge_attr = np.zeros((2, 4), dtype=np.float32)
                edges = torch.from_numpy(edges).bool()
                edge_attr = torch.from_numpy(edge_attr)
                print("shape of edges: ", edges.shape)
                print("shape of edge_attr: ", edge_attr.shape)
        return edges, edge_attr

    @staticmethod
    def _get_eight_neighbor(cloth_xdim, cloth_ydim, observable_particle_idx=None):
        # Connect cloth particles based on the ground-truth edges
        # Cloth index looks like the following:
        # 0, 1, ..., cloth_xdim -1
        # ...
        # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1
        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
        all_idx = np.arange(cloth_xdim * cloth_ydim).reshape([cloth_ydim, cloth_xdim])
        if observable_particle_idx is not None:
            observable_mask = np.zeros(cloth_xdim * cloth_ydim, dtype=np.int)
            observable_mask[observable_particle_idx] = 1
            # the observable particle index is in the downsample range, e.g., downsample_particle_pos[observable_particle_idx],
            # need to change this to be in the range [0, len(observable_particle_idx) - 1]
            edge_map = {}
            for idx, o_idx in enumerate(observable_particle_idx):
                edge_map[o_idx] = idx

        senders = []
        receivers = []

        # Horizontal connections
        idx_s = all_idx[:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1
        senders.append(idx_s)
        receivers.append(idx_r)

        # Vertical connections
        idx_s = all_idx[:-1, :].reshape(-1, 1)
        idx_r = idx_s + cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        # Diagonal connections
        idx_s = all_idx[:-1, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 + cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        idx_s = all_idx[1:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 - cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        if observable_particle_idx is None:
            senders = np.concatenate(senders, axis=0)
            receivers = np.concatenate(receivers, axis=0)
        else:
            obsverable_senders, observable_receivers = [], []
            senders, receivers = np.concatenate(senders).flatten(), np.concatenate(receivers).flatten()
            for s, r in zip(senders, receivers):
                if observable_mask[s] and observable_mask[r]:
                    obsverable_senders.append(s)
                    observable_receivers.append(r)

            senders = [edge_map[x] for x in obsverable_senders]
            receivers = [edge_map[x] for x in observable_receivers]
            senders = np.array(senders, dtype=np.long).reshape((-1, 1))
            receivers = np.array(receivers, dtype=np.long).reshape((-1, 1))

        new_senders = np.concatenate([senders, receivers], axis=0)
        new_receivers = np.concatenate([receivers, senders], axis=0)
        edges = np.concatenate([new_senders, new_receivers], axis=1).T
        assert edges.shape[0] == 2
        return edges

    def _downsample_mapping(self, cloth_ydim, cloth_xdim, idx, downsample):
        """ Given the down sample scale, map each point index before down sampling to the index after down sampling
        downsample: down sample scale
        """
        y, x = idx // cloth_xdim, idx % cloth_xdim
        down_ydim, down_xdim = (cloth_ydim + downsample - 1) // downsample, (cloth_xdim + downsample - 1) // downsample
        down_y, down_x = y // downsample, x // downsample
        new_idx = down_y * down_xdim + down_x
        return new_idx

    def _downsample(self, data, scale=2, test=False):
        if not test:
            pos, vel_his, picked_points, picked_point_pos, scene_params = data
        else:
            pos, vel_his, pciker_positions, actions, picked_points, scene_params, shape_pos = data
            # print("in downsample, picked points are: ", picked_points)

        sphere_radius, cloth_xdim, cloth_ydim, config_id = scene_params
        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
        original_xdim, original_ydim = cloth_xdim, cloth_ydim
        new_idx = np.arange(cloth_xdim * cloth_ydim).reshape((cloth_ydim, cloth_xdim))
        new_idx = new_idx[::scale, ::scale]
        cloth_ydim, cloth_xdim = new_idx.shape
        new_idx = new_idx.flatten()
        pos = pos[new_idx, :]
        vel_his = vel_his[new_idx, :]

        # Remap picked_points
        pps = []
        for pp in picked_points.astype('int'):
            if pp != -1:
                pps.append(self._downsample_mapping(original_ydim, original_xdim, pp, scale))
                assert pps[-1] < len(pos)
            else:
                pps.append(-1)

        scene_params = sphere_radius, cloth_xdim, cloth_ydim, config_id

        if not test:
            return (pos, vel_his, pps, picked_point_pos, scene_params), new_idx
        else:
            return (pos, vel_his, pciker_positions, actions, pps, scene_params, shape_pos), new_idx

    def load_rollout_data(self, idx_rollout, idx_timestep):
        data_cur = load_data(self.data_dir, idx_rollout, idx_timestep, self.data_names)
        data_nxt = load_data(self.data_dir, idx_rollout, idx_timestep + self.args.pred_time_interval, self.data_names)
        rollout_info = load_data(self.data_dir, idx_rollout, 'rollout_info', ['downsample_idx','target_pos'])
        # accumulate action if we need to predict multiple steps
        action = data_cur['action']

        for i in range(self.args.num_picker):
            action[i*4:i*4+3] = data_nxt['picker_position'][i,:3] - data_cur['picker_position'][i,:3]

        data_cur['action'] = action
        data_cur['downsample_idx'] = rollout_info['downsample_idx']
        data_cur['gt_reward_crt'] = cloth_drop_reward_fuc(data_cur['positions'][rollout_info['downsample_idx']],rollout_info['target_pos'][rollout_info['downsample_idx']] )

        return data_cur

    @staticmethod
    def remove_suffix(data, m_name):
        suffix = '_{}'.format(m_name)
        new_data = {}
        for k, v in data.items():
            new_data[k.replace(suffix, '')] = v
        return new_data

    def __len__(self):
        return self.cumulative_lengths[-1]

    def len(self):  # required by torch_geometric.data.dataset
        return len(self)

    def get(self, idx):
        return self.__getitem__(idx)

    def traj_id_to_idx(self, traj_id):
        if traj_id >= self.num_traj:
            raise ValueError('traj_id should be less than num_traj')

        if traj_id >0:
            return self.cumulative_lengths[traj_id-1]
        else:
            return 0

    def dataset_check(self):
        n = len(self.cumulative_lengths)

        for idx_rollout in range(n):
            data_cur = load_data(self.data_dir, idx_rollout, 0, ['positions'])
            len_pc = data_cur['positions'].shape[0]
            if idx_rollout == 0:
                len_rollout = 0
            else:
                len_rollout = self.args.pred_time_interval+self.cumulative_lengths[idx_rollout] - self.cumulative_lengths[idx_rollout-1]
            for idx_timestep in range(len_rollout):
                data_cur = load_data(self.data_dir, idx_rollout, idx_timestep, ['positions'])
                if len_pc != data_cur['positions'].shape[0]:
                    print("dataset_check: ", len_pc, data_cur['positions'].shape[0])
                    self.del_h5(idx_rollout, idx_timestep, len_rollout)
                    break

    def del_h5(self, idx_rollout, steps_start, steps_end):
        for idx_timestep in range(steps_start, steps_end):
            data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
            os.remove(data_path)
