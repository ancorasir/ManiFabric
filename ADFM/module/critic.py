from ADFM.module.models import GNN

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from omegaconf import OmegaConf
import os.path as osp
from chester import logger
import numpy as np
import torch_geometric

from ADFM.module.dataset_critic import ClothDatasetCritic
from ADFM.utils.utils import extract_numbers
from ADFM.utils.data_utils import AggDict
import json
from tqdm import tqdm
import pyflex
import cv2

class Critic(object):
    def __init__(self, args, env=None):
        self.args = args
        self.env = env
        self.model = GNN(args.model, decoder_output_dim=1, name='CriticGNN')  # Predict performace
        self.device = torch.device(self.args.cuda_idx)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.param(), lr=args.lr, betas=(args.beta1, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.8, patience=3, verbose=True)
        if self.args.critic_model_path is not None:
            self.load_model(self.args.load_optim)

        self.datasets = {phase: ClothDatasetCritic(args.dataset, 'vsbl', phase, env, 'vsbl') for phase in ['train', 'valid']}
        follow_batch = 'x_'
        self.dataloaders = {
            x: torch_geometric.data.DataLoader(
                self.datasets[x], batch_size=args.batch_size, follow_batch=follow_batch,
                shuffle=True if x == 'train' else False, drop_last=True,
                num_workers=args.num_workers, prefetch_factor=8)
            for x in ['train', 'valid']
        }

        self.log_dir = logger.get_dir()
        self.mse_loss =  torch.nn.MSELoss()
        self.load_epoch = 0

        if self.args.use_wandb and args.eval == 0:
            # To use wandb, you need to create an account and run 'wandb login'.
            wandb.init(project='ADFM', entity='yanglh14', name=args.exp_name, resume='allow',
                       id=None, settings=wandb.Settings(start_method='thread'))
            print('Weights & Biases is initialized with run name {}'.format(args.exp_name))
            wandb.config.update(args, allow_val_change=True)
    def train(self):

        # Training loop
        st_epoch = self.load_epoch
        best_valid_loss = np.inf
        for epoch in range(st_epoch, self.args.n_epoch):
            phases = ['train', 'valid'] if self.args.eval == 0 else ['valid']
            for phase in phases:
                self.set_mode(phase)
                epoch_info = AggDict(is_detach=True)

                for i, data in tqdm(enumerate(self.dataloaders[phase]), desc=f'Epoch {epoch}, phase {phase}'):
                    data = data.to(self.device).to_dict()
                    iter_info = AggDict(is_detach=False)
                    last_global = torch.zeros(self.args.batch_size, self.args.model.global_size, dtype=torch.float32, device=self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        data['u'] = last_global
                        pred_ = self.model(data)
                        loss = self.mse_loss(pred_['value'], data['gt_value'])
                        iter_info.add_item('loss', loss)

                    if phase == 'train':
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    epoch_info.update_by_add(iter_info)
                    iter_info.clear()

                    epoch_len = len(self.dataloaders[phase])
                    if i == len(self.dataloaders[phase]) - 1:
                        avg_dict = epoch_info.get_mean('{}/'.format(phase), epoch_len)
                        avg_dict['lr'] = self.optim.param_groups[0]['lr']
                        for k, v in avg_dict.items():
                            logger.record_tabular(k, v)

                        logger.dump_tabular()

                    if phase == 'train' and i == len(self.dataloaders[phase]) - 1:
                        suffix = '{}'.format(epoch)
                        self.model.save_model(self.log_dir, 'vsbl', suffix, self.optim)

                print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
                      (phase, epoch, self.args.n_epoch, avg_dict[f'{phase}/loss'], best_valid_loss))

                if phase == 'valid':
                    cur_loss = avg_dict[f'{phase}/loss']
                    self.scheduler.step(cur_loss)
                    if (cur_loss < best_valid_loss):
                        best_valid_loss = cur_loss
                        state_dict = OmegaConf.to_container(self.args, resolve=True)
                        state_dict['best_epoch'] = epoch
                        state_dict['best_valid_loss'] = cur_loss
                        with open(osp.join(self.log_dir, 'best_state.json'), 'w') as f:
                            json.dump(state_dict, f, indent=2, sort_keys=True)
                        self.model.save_model(self.log_dir, 'vsbl', 'best', self.optim)

                if self.args.use_wandb and self.args.eval == 0:
                    wandb.log(epoch_info, step=epoch)

    def visualize(self):
        phase = 'valid'
        dataset = self.datasets[phase]
        traj_ids = np.random.randint(0, len(dataset), self.args.plot_num)

        for i, traj_id in enumerate(traj_ids):
            pred_value, data_visual= self.load_data_and_predict(traj_id, dataset)
            image = self.visual(*data_visual)
            # RGB to BGR
            image = image[:, :, ::-1]
            pred_value = pred_value.detach().cpu().numpy()
            cv2.imwrite(osp.join(self.log_dir, 'vis_{}_{}.png'.format(i, pred_value)), image)
    def load_data_and_predict(self, traj_id, dataset):

        self.set_mode('eval')

        data_ori = dataset._prepare_transition(traj_id, eval=True)
        data = dataset.build_graph(data_ori)
        gt_reward = data['gt_value'].detach().cpu().numpy()

        with torch.no_grad():
            data['x_batch'] = torch.zeros(data['x'].size(0), dtype=torch.long, device=self.device)
            data['u'] = torch.zeros([1, self.args.model.global_size], device=self.device)
            for key in ['x', 'edge_index', 'edge_attr']:
                data[key] = data[key].to(self.device)
            pred_value = self.model(data)['value']

        return pred_value, (self.env, data_ori['pointcloud'])

    def visual(self, env, particle_position):

        """ Render point cloud trajectory without running the simulation dynamics"""
        env.reset()
        particle_pos = particle_position
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = [0., -0.1, 0.]  # All particles moved underground
        p[:len(particle_pos), :3] = particle_pos
        pyflex.set_positions(p)
        rgb = env.get_image(env.camera_width, env.camera_height)

        return rgb

    def to(self, cuda_idx):
        self.model.to(torch.device("cuda:{}".format(cuda_idx)))

    def set_mode(self, mode='train'):
        self.model.set_mode('train' if mode == 'train' else 'eval')

    def load_model(self, load_optim=False):
        self.model.load_model(self.args.critic_model_path, load_optim=load_optim, optim=self.optim)
        self.load_epoch = extract_numbers(self.args.critic_model_path)[-1]
