import os.path as osp
import numpy as np
import json

from FabricMani.module.dynamics import Dynamic
from FabricMani.module.edge import Edge

from FabricMani.utils.utils import vv_to_args, voxelize_pointcloud
from FabricMani.utils.camera_utils import get_world_coords, get_observable_particle_index_3

def get_rgbd_and_mask(env, sensor_noise):
    rgbd = env.get_rgbd(show_picker=True)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    if sensor_noise > 0:
        non_cloth_mask = (depth <= 0)
        depth += np.random.normal(loc=0, scale=sensor_noise,
                                  size=(depth.shape[0], depth.shape[1]))
        depth[non_cloth_mask] = 0

    return depth.copy(), rgb, depth

def load_edge_model(edge_model_path, env, args):
    if edge_model_path is not None:
        edge_model_dir = osp.dirname(edge_model_path)
        edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
        edge_model_vv['eval'] = 1
        edge_model_vv['n_epoch'] = 1
        edge_model_vv['edge_model_path'] = edge_model_path
        edge_model_vv['env_shape'] = args.env_shape
        edge_model_args = vv_to_args(edge_model_vv)

        edge = Edge(edge_model_args, env=env)
        print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
    else:
        print("no edge GNN model is loaded")
        edge = None

    return edge


def load_dynamics_model(args, env, edge):
    model_vv_dir = osp.dirname(args.partial_dyn_path)
    model_vv = json.load(open(osp.join(model_vv_dir, 'best_state.json')))

    model_vv[
        'fix_collision_edge'] = args.fix_collision_edge  # for ablation that train without mesh edges, if True, fix collision edges from the first time step during planning; If False, recompute collision edge at each time step
    model_vv[
        'use_collision_as_mesh_edge'] = args.use_collision_as_mesh_edge  # for ablation that train with mesh edges, but remove edge GNN at test time, so it uses first-time step collision edges as the mesh edges
    model_vv['train_mode'] = 'vsbl'
    model_vv['use_wandb'] = False
    model_vv['eval'] = 1
    model_vv['load_optim'] = False
    model_vv['pred_time_interval'] = args.pred_time_interval
    model_vv['cuda_idx'] = args.cuda_idx
    model_vv['partial_dyn_path'] = args.partial_dyn_path
    model_vv['env_shape'] = args.env_shape
    if 'use_es' not in model_vv.keys():
        model_vv['use_es'] = args.use_es
    args = vv_to_args(model_vv)

    dynamics = Dynamic(args, edge=edge, env=env)
    return dynamics

def data_prepration(env, args, config, scene_params, downsample_idx, **kwargs):
    # prepare input data for planning
    cloth_mask, rgb, depth = get_rgbd_and_mask(env, args.sensor_noise)
    world_coordinates = get_world_coords(rgb, depth, env)[:, :, :3].reshape((-1, 3))
    pointcloud = world_coordinates[depth.flatten() > 0].astype(np.float32)

    voxel_pc = voxelize_pointcloud(pointcloud, args.voxel_size)
    voxel_pc, observable_particle_indices = get_observable_particle_index_3(voxel_pc,
                                                                            env.get_state()['particle_pos'].reshape(-1,4)[
                                                                            downsample_idx, :3], args.voxel_size)


    vel_history = np.zeros((len(observable_particle_indices), args.n_his * 3), dtype=np.float32)

    if kwargs:
        gt_positions = kwargs['gt_positions']
        control_seq_idx = kwargs['control_seq_idx']
        if len(gt_positions) > 1:
            for i in range(min(len(gt_positions) - 1, args.n_his - 1)):
                start_index = (min(control_seq_idx, args.n_his)) * (-3) + i * 3
                vel_history[:, start_index:start_index + 3] = (gt_positions[i + 1][0][observable_particle_indices] -
                                                               gt_positions[i][0][observable_particle_indices]) / (
                                                                          args.dt * args.pred_time_interval)
            # -1 since the last position in gt_positions is the current position
            vel_history[:, -3:] = (voxel_pc - gt_positions[-1][0][observable_particle_indices]) / (
                        args.dt * (args.pred_time_interval-1))

        elif len(gt_positions) == 1:
            vel_history[:, -3:] = (voxel_pc - gt_positions[-1][0][observable_particle_indices]) / (
                        args.dt * (args.pred_time_interval-1))

    picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
    data = {
        'pointcloud': voxel_pc,
        'vel_his': vel_history,
        'picker_position': picker_position,
        'action': env.action_space.sample(),  # action will be replaced by sampled action later
        'picked_points': picked_points,
        'scene_params': scene_params,
        'partial_pc_mapped_idx': observable_particle_indices,
        'downsample_idx': downsample_idx,
        'target_pos': config['target_pos'],
        'target_picker_pos': config['target_picker_pos'],
    }
    if config['env_shape'] is not None:
        data['shape_size'] = config['shape_size']
        data['shape_pos'] = config['shape_pos']
        data['shape_quat'] = config['shape_quat']
        data['env_shape'] = config['env_shape']

    return data