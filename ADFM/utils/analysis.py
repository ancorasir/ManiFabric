###
# this file is to analysis the data
###

import numpy as np
import os

from ADFM.utils.utils import load_data
def main():
    data_dir = '../data/dia_v2/train'
    if os.path.exists('rollout_data.npy'):
        data = np.load('rollout_data.npy')
    else:
        data = []
        for idx_rollout in range(450):
            mid_state_step, x_bias, z_mid, reward = load_rollout_data(data_dir, idx_rollout, None)
            data.append([mid_state_step, x_bias, z_mid, reward])
        data = np.array(data)
        np.save('rollout_data.npy', data)
    import pandas as pd
    df = pd.DataFrame(data, columns=['mid_state_step', 'x_bias', 'z_mid', 'performance'])
    correlation_matrix = df.corr()
    # plot correlation matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    data[:,3] -= data[:,3].min()
    data[:, 3] /= data[:, 3].max()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3])
    fig.colorbar(p)

    ax.set_xlabel('mid_state_step')
    ax.set_ylabel('x_bias')
    ax.set_zlabel('z_mid')
    plt.show()

def load_rollout_data(data_dir, idx_rollout, names):
    picker_positions = []
    for idx_step in range(100):
        data_cur = load_data(data_dir, idx_rollout, idx_step, names)
        picker_positions.append(data_cur['picker_position'][0])
        if idx_step ==99:
            cur_position = data_cur['positions']
            target_position = data_cur['target_pos']
            reward = -np.linalg.norm(cur_position - target_position)

    picker_positions = np.array(picker_positions)
    mid_state_step = np.argmax(picker_positions[:,0])
    x_bias = np.max(picker_positions[:,0]) - picker_positions[-1,0]
    z_mid = picker_positions[mid_state_step, 1]

    return mid_state_step, x_bias, z_mid, reward

if __name__ == '__main__':
    main()
