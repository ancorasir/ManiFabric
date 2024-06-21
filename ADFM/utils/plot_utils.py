import numpy as np
import pyflex
import os
import os.path as osp
import copy

from softgym.utils.visualization import save_numpy_as_gif
import matplotlib.pyplot as plt
from ADFM.utils.camera_utils import get_matrix_world_to_camera

from ADFM.utils.utils import (
    visualize, draw_edge,cem_make_gif,draw_target_pos, project_to_image
)

def seg_3d_figure(data: np.ndarray, labels: np.ndarray, labelmap=None, sizes=None, fig=None):
    import plotly.colors as pc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.figure_factory as ff

    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)

    # Colormap.
    cols = np.array(pc.qualitative.Alphabet)
    labels = labels.astype(int)
    for label in np.unique(labels):
        subset = data[np.where(labels == label)]
        subset = np.squeeze(subset)
        if sizes is None:
            subset_sizes = 1.5
        else:
            subset_sizes = sizes[np.where(labels == label)]
        color = cols[label % len(cols)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker={"size": subset_sizes, "color": color, "line": {"width": 0}},
                x=subset[:, 0],
                y=subset[:, 1],
                z=subset[:, 2],
                name=legend,
            )
        )
    fig.update_layout(showlegend=True)

    # This sets the figure to be a cube centered at the center of the pointcloud, such that it fits
    # all the points.
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
            yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
            zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75),
    )
    return fig

def draw_target_gif(env, matrix_world_to_camera, log_dir_episode, env_name, episode_idx):
    target_pos = env.get_current_config()['target_pos']
    curr_pos = pyflex.get_positions().reshape((-1, 4))
    curr_pos[:, :3] = target_pos
    pyflex.set_positions(curr_pos)
    _img = env.get_image(env.camera_width, env.camera_height)
    _imgs = [_img, _img]
    for i in range(len(_imgs)):
        _imgs[i] = draw_target_pos(_imgs[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                   env.camera_height, env.camera_width, env._get_key_point_idx())
    cem_make_gif([_imgs], log_dir_episode, env_name + '{}_target.gif'.format(episode_idx))

def draw_prediction_step(actual_control_num, gt_positions, gt_shape_positions,render_env,
                         model_pred_particle_poses,model_pred_shape_poses, config_id, downsample_idx,
                            predicted_edges_all, matrix_world_to_camera, log_dir_episode, episode_idx,
                         args, predicted_all_results, env):
    # make gif visuals of the model predictions and groundtruth rollouts
    for control_sequence_idx in range(0, actual_control_num):
        # subsample the actual rollout (since we decomposed the 5-step action to be 1-step actions)
        subsampled_gt_pos, subsampled_shape_pos = [], []
        for t in range(5):
            subsampled_gt_pos.append(gt_positions[control_sequence_idx][t])
            subsampled_shape_pos.append(gt_shape_positions[control_sequence_idx][t])

        frames_model = visualize(render_env, model_pred_particle_poses[control_sequence_idx],
                                 model_pred_shape_poses[control_sequence_idx],
                                 config_id, range(model_pred_particle_poses[control_sequence_idx].shape[1]))
        frames_gt = visualize(render_env, subsampled_gt_pos, subsampled_shape_pos, config_id, downsample_idx)

        # visualize the infered edge from edge GNN
        predicted_edges = predicted_edges_all[control_sequence_idx]
        frames_edge = copy.deepcopy(frames_model)
        pointcloud_pos = model_pred_particle_poses[control_sequence_idx]
        for t in range(len(frames_edge)):
            frames_edge[t] = draw_edge(frames_edge[t], predicted_edges, matrix_world_to_camera[:3, :],
                                       pointcloud_pos[t], env.camera_height, env.camera_width)

        # save the gif
        for name in ['gt', 'model', 'edge']:
            _frames = np.array(eval('frames_{}'.format(name)))
            for t in range(len(_frames)):
                _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'],
                                             matrix_world_to_camera[:3, :],
                                             env.camera_height, env.camera_width, env._get_key_point_idx())

            save_numpy_as_gif(_frames,
                              osp.join(log_dir_episode, '{}-{}-{}.gif'.format(episode_idx, control_sequence_idx, name)))

        if control_sequence_idx == 0:
            model_pred_particle_poses_all = []
            for i in range(args.shooting_number):

                frames_model_sample_action = visualize(render_env,
                                                       predicted_all_results[control_sequence_idx][i]['model_positions'],
                                                       model_pred_shape_poses[control_sequence_idx],
                                                       config_id,
                                                       range(model_pred_particle_poses[control_sequence_idx].shape[1]))

                _frames = np.array(frames_model_sample_action)
                for t in range(len(_frames)):
                    _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'],
                                                 matrix_world_to_camera[:3, :],
                                                 env.camera_height, env.camera_width, env._get_key_point_idx())

                log_dir_episode_sample_action = osp.join(log_dir_episode,
                                                         'sample_action_{}'.format(control_sequence_idx))
                os.makedirs(log_dir_episode_sample_action, exist_ok=True)
                save_numpy_as_gif(_frames, osp.join(log_dir_episode_sample_action,
                                                    '{}-{}-ActionSampling{}.gif'.format(episode_idx,
                                                                                        control_sequence_idx, i)))
def draw_gt_trajectory(gt_positions, gt_shape_positions, render_env, config_id, downsample_idx,
                       matrix_world_to_camera, log_dir_episode, episode_idx, env):
    _gt_positions = np.array(gt_positions).reshape(-1, len(downsample_idx), 3)
    _gt_shape_positions = np.array(gt_shape_positions).reshape(-1, 2, 3)
    frames_gt = visualize(render_env, _gt_positions, _gt_shape_positions, config_id, downsample_idx)
    for t in range(len(frames_gt)):
        frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'],
                                       matrix_world_to_camera[:3, :],
                                       env.camera_height, env.camera_width, env._get_key_point_idx())
    save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}-{}.gif'.format(episode_idx, 'gt')))

def draw_init_trajectory(pred_positions, pred_shape_positions, render_env, config_id,
                       matrix_world_to_camera, log_dir_episode, episode_idx, env):
    frames_gt = visualize(render_env, pred_positions, pred_shape_positions, config_id)
    for t in range(len(frames_gt)):
        frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'],
                                       matrix_world_to_camera[:3, :],
                                       env.camera_height, env.camera_width, env._get_key_point_idx())
    save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}-{}.gif'.format(episode_idx, 'init')), fps=10)

def draw_middle_state(gt_positions, gt_shape_positions, render_env, config_id,
                       matrix_world_to_camera, log_dir_episode,idx, env):
    _gt_positions = np.array(gt_positions)
    _gt_shape_positions = np.array(gt_shape_positions)
    frames_gt = visualize(render_env, _gt_positions, _gt_shape_positions, config_id)
    for t in range(len(frames_gt)):
        frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'],
                                       matrix_world_to_camera[:3, :],
                                       env.camera_height, env.camera_width, env._get_key_point_idx())
    save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}.gif'.format(idx)))


def plot_performance_curve(transformed_info, log_dir_episode, episode_idx, predicted_performances, gt_shape_positions):
    performance_gt = transformed_info['performance'][0]
    performance_pre = np.array(predicted_performances)
    plt.plot(performance_gt, label='gt')
    plt.plot(performance_pre, label='pre')
    plt.legend()
    plt.savefig(osp.join(log_dir_episode, '{}-performance.png'.format(episode_idx)))
    plt.close()

    _gt_shape_positions = np.array(gt_shape_positions).reshape(-1, 2, 3)
    plt.plot(_gt_shape_positions[:, 0, 0], _gt_shape_positions[:, 0, 1])
    plt.savefig(os.path.join(log_dir_episode, 'X-Z.png'))
    plt.close()

def make_result_gif(frames, env, matrix_world_to_camera, episode_idx, logger, args,frames_top):
    for i in range(len(frames)):
        frames[i] = draw_target_pos(frames[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                    env.camera_height, env.camera_width, env._get_key_point_idx())

    cem_make_gif([frames], logger.get_dir(), args.env_name + '{}.gif'.format(episode_idx))


    matrix_world_to_camera_ = get_matrix_world_to_camera(cam_angle=np.array([1.57, -1.57, 0]),
                                                         cam_pos=np.array([0.2, 1.5, 0]))
    for i in range(len(frames_top)):
        frames_top[i] = draw_target_pos(frames_top[i], env.get_current_config()['target_pos'],
                                        matrix_world_to_camera_[:3, :],
                                        env.camera_height, env.camera_width, env._get_key_point_idx())
    cem_make_gif([frames_top], logger.get_dir(), args.env_name + '{}_top.gif'.format(episode_idx))

def plot_figure(pred_positions, pred_shapes, returns, results,
                env, render_env, config_id, matrix_world_to_camera, log_dir_episode, episode_idx, args,
                planner, pyflex, downsample_idx, log_dir,
                gt_positions, gt_shape_positions, frames, frames_top):
    import cv2
    import pickle
    def filter_img(image):
        # filter all write or grey pixels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV range for blue and yellow
        lower_blue = np.array([100, 150, 20])
        upper_blue = np.array([140, 255, 255])
        lower_yellow = np.array([20, 0, 0])
        upper_yellow = np.array([70, 255, 220])

        lower_background = np.array([0, 0, 0])
        upper_background = np.array([255, 20, 255])

        # Create masks for blue and yellow
        background_mask = cv2.inRange(hsv, lower_background, upper_background)

        image[np.where(background_mask == 255)] = [255, 255, 255]

        return image

    def draw_vel_pred(image, particle_pos, pred_pos, matrix_world_to_camera, camera_height, camera_width):
        # draw velocity prediction
        u, v = project_to_image(matrix_world_to_camera, particle_pos, camera_height, camera_width)
        u_pred, v_pred = project_to_image(matrix_world_to_camera, pred_pos, camera_height, camera_width)
        for i in range(len(u)):
            cv2.arrowedLine(image, (u[i], v[i]), (u_pred[i], v_pred[i]), (255, 0, 0), 1)
        return image

    pred_positions_all = [result['model_positions'] for result in results]
    pred_shapes_all = [result['shape_positions'] for result in results]
    highest_return_idx = np.argmax(returns)
    # plot every figure in results for each traj in sampling
    tasks = [0, 1, 2, 3, 4, 5, 6, 7]

    for task in tasks:
        target_ = task//4
        vel_ = task in [1,3,5,7]
        filter_ = task in [2,3,6,7]
        for i in range(len(results)):
            if i != highest_return_idx:
                continue
            pred_position = pred_positions_all[i]
            pred_shape = pred_shapes_all[i]
            pred_frames = visualize(render_env, pred_position, pred_shape, config_id)
            traj_dir = osp.join(log_dir_episode, 'traj_{}_target_{}_vel_{}_filter_{}'.format(i, target_, vel_, filter_))
            os.makedirs(traj_dir, exist_ok=True)
            for j in range(0, len(pred_frames)):
                if target_:
                    pred_frames[j] = draw_target_pos(pred_frames[j], env.get_current_config()['target_pos'],
                                                matrix_world_to_camera[:3, :],
                                                env.camera_height, env.camera_width, env._get_key_point_idx())
                if vel_:
                    if j < len(pred_frames) - 1:
                        pred_frames[j] = draw_vel_pred(pred_frames[j], pred_position[j], pred_position[j + 1], matrix_world_to_camera,
                                                  env.camera_height, env.camera_width)

                pred_frames[j] = pred_frames[j][:, :, ::-1]
                if filter_:
                    pred_frames[j] = filter_img(pred_frames[j])

                cv2.imwrite(osp.join(traj_dir, 'step_{}.png'.format(j)), pred_frames[j])

    tasks = [0, 1, 2, 3, 4, 5, 6, 7]

    for task in tasks:
        target_ = task//4
        vel_ = task%2
        filter_ = 1 - task%2

        gt_position = [gt[0] for gt in gt_positions]
        gt_shape = [gt[0] for gt in gt_shape_positions]

        frames_gt_particle = visualize(render_env, gt_position, gt_shape, config_id)
        traj_dir = osp.join(log_dir_episode, 'particle_gt_target_{}_vel_{}_filter_{}'.format(target_,vel_, filter_))
        os.makedirs(traj_dir, exist_ok=True)
        for j in range(0, len(frames_gt_particle)):
            if target_:
                frames_gt_particle[j] = draw_target_pos(frames_gt_particle[j], env.get_current_config()['target_pos'],
                                                        matrix_world_to_camera[:3, :],
                                                        env.camera_height, env.camera_width, env._get_key_point_idx())
            if vel_:
                if j < len(frames_gt_particle) - 1:
                    frames_gt_particle[j] = draw_vel_pred(frames_gt_particle[j], gt_position[j], gt_position[j+1],matrix_world_to_camera, env.camera_height, env.camera_width)

            frames_gt_particle[j] = frames_gt_particle[j][:, :, ::-1]
            if filter_:
                frames_gt_particle[j] = filter_img(frames_gt_particle[j])

            cv2.imwrite(osp.join(traj_dir, 'step_{}.png'.format(j)), frames_gt_particle[j])

    for task in ['no_filter','filter']:
        for target_ in [0,1]:
            frames_copy = copy.deepcopy(frames)
            frames_top_copy = copy.deepcopy(frames_top)
            # get frame every 10 steps
            gt_frames, gt_frames_top = [], []
            for i in range(0, len(frames), args.pred_time_interval):
                gt_frames.append(frames_copy[i])
                gt_frames_top.append(frames_top_copy[i])

            log_dir_gt = osp.join(log_dir_episode, 'gt_'+task+'_target_{}'.format(target_))
            os.makedirs(log_dir_gt, exist_ok=True)

            for i in range(len(gt_frames)):
                if target_:
                    gt_frames[i] = draw_target_pos(gt_frames[i], env.get_current_config()['target_pos'],
                                                   matrix_world_to_camera[:3, :],
                                                   env.camera_height, env.camera_width, env._get_key_point_idx())
                gt_frames[i] = gt_frames[i][:, :, ::-1]
                if target_:
                    matrix_world_to_camera_ = get_matrix_world_to_camera(cam_angle=np.array([1.57, -1.57, 0]),
                                                                         cam_pos=np.array([0.2, 1.5, 0]))

                    gt_frames_top[i] = draw_target_pos(gt_frames_top[i], env.get_current_config()['target_pos'],
                                                       matrix_world_to_camera_[:3, :],
                                                       env.camera_height, env.camera_width, env._get_key_point_idx())
                gt_frames_top[i] = gt_frames_top[i][:, :, ::-1]

                if task == 'filter':
                    gt_frames[i] = filter_img(gt_frames[i])
                    gt_frames_top[i] = filter_img(gt_frames_top[i])

                cv2.imwrite(osp.join(log_dir_gt, 'frames_{}.png'.format(i)), gt_frames[i])
                cv2.imwrite(osp.join(log_dir_gt, 'frames_top_{}.png'.format(i)), gt_frames_top[i])


    plot_data = {"pred_positions": pred_positions,
                 "pred_shapes": pred_shapes,
                 "returns": returns,
                 "results": results,
                 "gt_positions": gt_positions,
                 "gt_shape_positions": gt_shape_positions,
                 "gt_frames": gt_frames,
                 "gt_frames_top": gt_frames_top,
                 "frames": frames,
                 "frames_top": frames_top,
                 }
    with open(osp.join(log_dir, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)

    with open(osp.join(log_dir, 'plot_data.pkl'), 'rb') as f:
        plot_data_load = pickle.load(f)
