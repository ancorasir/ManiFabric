import numpy as np
import torch
import os.path as osp
from scipy import spatial
from torch_geometric.data import Data

from FabricMani.utils.camera_utils import get_observable_particle_index_3
from FabricMani.module.dataset import ClothDataset
from FabricMani.utils.utils import load_data, voxelize_pointcloud, cloth_drop_reward_fuc


class ClothDatasetCritic(ClothDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not osp.exists(self.data_dir+ '/mid_steps.npy'):
            self.mid_steps = self.get_mid_steps()
            np.save(self.data_dir+ '/mid_steps.npy', self.mid_steps)
        else:
            self.mid_steps = np.load(self.data_dir+ '/mid_steps.npy')
    def __getitem__(self, idx):
        data = self._prepare_transition(idx)
        d = self.build_graph(data)
        return Data.from_dict(d)

    def _prepare_transition(self, idx, eval=False):
        success = False

        while not success:
            idx_rollout = idx
            idx_timestep = self.mid_steps[idx_rollout]
            len_traj = self.get_traj_len(idx_rollout)
            data = load_data(self.data_dir, idx_rollout, idx_timestep, self.data_names)
            data_rollout = load_data(self.data_dir, idx_rollout, 'rollout_info', None)
            data_final = load_data(self.data_dir, idx_rollout, len_traj-1, ['positions'])
            vox_pc = data['pointcloud'].astype(np.float32)

            partial_particle_pos = data['positions'][data_rollout['downsample_idx']][data['downsample_observable_idx']]
            if len(vox_pc) <= len(partial_particle_pos):
                success = True
            else:
                success = True # not require vox_pc len smaller than partial_particle_pos

        vox_pc, partial_pc_mapped_idx = get_observable_particle_index_3(vox_pc, partial_particle_pos, threshold=self.args.voxel_size)
        normalized_vox_pc = vox_pc - np.mean(vox_pc, axis=0)

        if self.args.env_shape is not None:
            distance_to_shape, vector_to_shape = self._compute_distance_to_shape(vox_pc, data_rollout['shape_pos'], data_rollout['shape_size'], data_rollout['shape_quat'])
        else:
            distance_to_shape = torch.from_numpy(vox_pc[:, 1]).view((-1, 1))
            vector_to_shape = torch.zeros([distance_to_shape.shape[0],3])
            vector_to_shape[:, 1] = 1

        downsample_idx = data_rollout['downsample_idx']
        full_pos_cur = data_final['positions']

        gt_reward = torch.FloatTensor([cloth_drop_reward_fuc(full_pos_cur[downsample_idx], data_rollout['target_pos'][downsample_idx])]) if 'target_pos' in data_rollout else None

        ret_data = {
            'scene_params': data_rollout['scene_params'],
            'downsample_observable_idx': data['downsample_observable_idx'],
            'normalized_vox_pc': normalized_vox_pc,
            'partial_pc_mapped_idx': partial_pc_mapped_idx,
            'distance_to_shape': distance_to_shape,
            'vector_to_shape': vector_to_shape,
            'gt_reward': gt_reward,
        }
        if eval:
            ret_data['downsample_idx'] = data_rollout['downsample_idx']
            ret_data['pointcloud'] = vox_pc

        return ret_data

    def _compute_edge_attr(self, vox_pc):
        point_tree = spatial.cKDTree(vox_pc)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = vox_pc[undirected_neighbors[0, :]] - vox_pc[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
            edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
        else:
            print("number of distance edges is 0! adding fake edges")
            edges = np.zeros((2, 2), dtype=np.uint8)
            edges[0][0] = 0
            edges[1][0] = 1
            edges[0][1] = 0
            edges[1][1] = 2
            edge_attr = np.zeros((2, self.args.relation_dim), dtype=np.float32)
            edges = torch.from_numpy(edges).bool()
            edge_attr = torch.from_numpy(edge_attr)
            print("shape of edges: ", edges.shape)
            print("shape of edge_attr: ", edge_attr.shape)

        return edges, edge_attr

    def build_graph(self, data, get_gt_edge_label=True):
        """
        data: positions, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        gt_mesh_edge: 0/1 label for groundtruth mesh edge connection.
        """
        normalized_vox_pc = torch.from_numpy(data['normalized_vox_pc'])
        node_attr = torch.cat([normalized_vox_pc, data['distance_to_shape'], data['vector_to_shape']], dim=1)
        edges, edge_attr = self._compute_edge_attr(data['normalized_vox_pc'])

        gt_value = data['gt_reward'].view((-1, 1))

        return {
            'x': node_attr,
            'edge_index': edges,
            'edge_attr': edge_attr,
            'gt_value': gt_value
        }

    def get_mid_steps(self):
        mid_steps = []
        for idx_rollout in range(len(self.cumulative_lengths)):
            traj = self.get_rollout_traj(idx_rollout)
            # mid step is when x traj reaches max
            mid_step = np.argmax(traj[:, 0])
            mid_steps.append(mid_step)
        return mid_steps

    def get_rollout_traj(self, idx_rollout):
        if idx_rollout == 0:
            traj_len = self.cumulative_lengths[idx_rollout] + self.args.pred_time_interval
        else:
            traj_len = self.cumulative_lengths[idx_rollout] - self.cumulative_lengths[idx_rollout-1] + self.args.pred_time_interval

        traj = []
        for idx_timestep in range(traj_len):
            data = load_data(self.data_dir, idx_rollout, idx_timestep, ['picker_position'])
            traj.append(data['picker_position'][0])
        return np.array(traj)

    def get_traj_len(self, idx_rollout):
        if idx_rollout == 0:
            traj_len = self.cumulative_lengths[idx_rollout] + self.args.pred_time_interval
        else:
            traj_len = self.cumulative_lengths[idx_rollout] - self.cumulative_lengths[idx_rollout-1] + self.args.pred_time_interval
        return traj_len
    def __len__(self):
        return len(self.cumulative_lengths)
