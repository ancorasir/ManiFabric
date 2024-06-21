from ADFM.module.planner import MPCPlanner
from chester import logger
import numpy as np
import multiprocessing as mp
import os.path as osp
import pyflex

from ADFM.utils.utils import cloth_drop_reward_fuc, downsample, draw_target_pos, visualize, project_to_image, voxelize_pointcloud, transform_info
from ADFM.utils.camera_utils import get_matrix_world_to_camera, get_world_coords, get_observable_particle_index_3
from ADFM.utils.env_utils import create_env_plan
from ADFM.utils.plot_utils import make_result_gif, plot_performance_curve, draw_gt_trajectory, draw_init_trajectory, plot_figure
from ADFM.utils.plan_utils import get_rgbd_and_mask, load_edge_model, load_dynamics_model, data_prepration

import os
import time
import cv2
import pickle
def plan(args):

    log_dir = args.log_dir
    mp.set_start_method('forkserver', force=True)

    env, render_env = create_env_plan(args)

    edge = load_edge_model(args.edge_model_path, env, args)
    dyn_model = load_dynamics_model(args, env, edge)

    # compute camera matrix
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

    # build random shooting planner
    planner = MPCPlanner(
        dynamics=dyn_model,
        reward_model=cloth_drop_reward_fuc,
        matrix_world_to_camera=matrix_world_to_camera,
        env=env,
        args=args,
    )

    performance_infos = []
    performance_all = []
    for episode_idx in range(args.configurations):

        log_dir_episode = osp.join(log_dir, str(episode_idx))
        os.makedirs(log_dir_episode, exist_ok=True)

        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)
        env.action_tool.update_picker_boundary([-0.3, 0.0, -0.5], [0.5, 2, 0.5])

        config = env.get_current_config()
        config_id = env.current_config_id
        if args.env_name == 'TshirtFlatten':
            cloth_xdim, cloth_ydim = 40,40
            scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]
            scene_params[1] = cloth_xdim
            scene_params[2] = cloth_ydim
            downsample_idx = np.array(range(config['target_pos'].shape[0]))[::9]
        else:
            cloth_xdim, cloth_ydim = config['ClothSize']

            scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

            downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, args.down_sample_scale)
            scene_params[1] = downsample_x_dim
            scene_params[2] = downsample_y_dim

        # change the color of the env
        n_shapes = pyflex.get_n_shapes()
        color_shapes = np.zeros((n_shapes, 3))
        color_shapes[:1, :] = [1.0, 1.0, 0.4]
        pyflex.set_shape_color(color_shapes)

        # Forsight planning - Stage 1

        data = data_prepration(env, args, config, scene_params, downsample_idx)

        pred_positions, pred_edges, pred_shapes, returns, results = planner.init_traj(data, args.sampling_num)

        # Stage 2
        # env.reset(config_id=episode_idx)
        rewards, infos, frames, frames_top = [], [], [], []
        gt_positions, gt_shape_positions = [], []
        pred_particle_poses, pred_edges_all, pred_performances, pred_shapes = [], [], [], []
        control_seq_idx = 0

        while True:
            if control_seq_idx >= len(planner.actions):
                break
            if args.baseline == None:
                data = data_prepration(env, args, config, scene_params, downsample_idx, gt_positions=gt_positions, control_seq_idx = control_seq_idx)
                action_seq, pred_particle_pos, highest_return, predicted_edges, pred_shape \
                    = planner.get_action(data, control_seq_idx=control_seq_idx)
                print("config {} control sequence idx {}".format(config_id, control_seq_idx), flush=True)

                pred_particle_poses.append(pred_particle_pos)
                pred_edges_all.append(predicted_edges)
                pred_performances.append(highest_return)
                pred_shapes.append(pred_shape)
                # action_seq = planner.actions[control_seq_idx].reshape(1, -1)

            else:
                action_seq = planner.actions[control_seq_idx].reshape(1, -1)

            # decompose the large action to be small 1-step actions to execute
            if args.pred_time_interval >= 2:
                action_exe= np.zeros((args.pred_time_interval, 8))
                action_exe[:, :] = (action_seq[0, :]) / args.pred_time_interval

                action_exe[action_exe[:, 3] > 0, 3] = 1
                action_exe[action_exe[:, 7] > 0, 7] = 1
            else:
                action_exe = action_seq
            gt_positions.append(np.zeros((len(action_exe), len(downsample_idx), 3)))
            gt_shape_positions.append(np.zeros((len(action_exe), 2, 3)))

            for t_idx, ac in enumerate(action_exe):
                grasped = ac[[3, 7]] > 0
                grasped_points = env._get_drop_point_idx()
                env.action_tool.picked_particles = [grasped_points[i] if grasped[i] else None for i in
                                                    range(len(grasped))]

                _, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                frames.extend(info['flex_env_recorded_frames'])
                frames_top.append(info['image_top'])

                info.pop("flex_env_recorded_frames")
                info.pop("image_top")
                info.pop("image_side")

                rewards.append(reward)

                gt_positions[control_seq_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[downsample_idx, :3]
                shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(2):
                    gt_shape_positions[control_seq_idx][t_idx][k] = shape_pos[k][:3]

            control_seq_idx += 1
            infos.append(info)

        #####################
        # SAVE EVERYTHING
        #####################

        # if args.baseline==None:
        #     plot_figure(pred_positions, pred_shapes, returns, results,
        #             env, render_env, config_id, matrix_world_to_camera, log_dir_episode, episode_idx, args,
        #             planner, pyflex, downsample_idx, log_dir, gt_positions, gt_shape_positions, frames, frames_top)

        # plot init result
        # draw_init_trajectory(res_init['pred_positions'], res_init['pred_shapes'], render_env, config_id,
        #                matrix_world_to_camera, log_dir_episode, episode_idx, env)
        time_cost = len(infos) * args.dt * args.pred_time_interval
        transformed_info = transform_info([infos])
        plot_performance_curve(transformed_info, log_dir_episode, episode_idx, pred_performances, gt_shape_positions)
        performance_infos.append([transformed_info['performance'][0,-1], transformed_info['normalized_performance'][0,-1],
                                 transformed_info['IoU'][0,-1], time_cost])
        performance_all.append([transformed_info['performance'][0,:], transformed_info['normalized_performance'][0,:],
                                 transformed_info['IoU'][0,:]])
        # make gif from two views
        make_result_gif(frames, env, matrix_world_to_camera, episode_idx, logger, args, frames_top)

        # # draw the groundtruth trajectory
        # draw_gt_trajectory(gt_positions, gt_shape_positions, render_env, config_id, downsample_idx,
        #                matrix_world_to_camera, log_dir_episode, episode_idx, env)

        print('episode {} finished'.format(episode_idx))


    np.save(osp.join(log_dir, 'performance_infos.npy'), np.array(performance_infos))
    np.save(osp.join(log_dir, 'performance_all.npy'), np.array(performance_all))

    print('Average performance: {}'.format(np.mean(np.array(performance_infos), axis=0)))
    print('std performance: {}'.format(np.std(np.array(performance_infos), axis=0)))
    # end
    print('planning finished')
