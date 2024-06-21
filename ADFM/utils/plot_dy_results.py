import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-whitegrid')

# Define your font settings
font = {
    'family': 'Times New Roman',  # Font family (e.g., 'serif', 'sans-serif', 'monospace')
    'weight': 'normal',  # Font weight ('normal', 'bold', 'light', 'heavy')
    'size': 32,  # Font size
    'style': 'normal'  # Font style ('normal', 'italic', 'oblique')
}

# Set the font properties
plt.rcParams['font.family'] = font['family']
plt.rcParams['font.weight'] = font['weight']
plt.rcParams['font.size'] = font['size']
plt.rcParams['font.style'] = font['style']

loss_dir = {}

sourece_folde = '../data/'
exp_names = ['dia_v4_pick_drop','dia_platform2', 'dia_sphere']
for exp_name in exp_names:
    exp_dir = os.path.join(sourece_folde, exp_name)
    trials = os.listdir(exp_dir)
    trials = [trial for trial in trials if 'dy' in trial]
    for trial in trials:
        #load a csv file
        csv_file = os.path.join(exp_dir, trial, 'progress.csv')

        # get the data from the csv file and keys
        data = np.genfromtxt(csv_file, delimiter=',', names=True)

        # get the data of vsblvalidsqrt_accel_loss
        vsblvalidsqrt_accel_loss = data['vsblvalidaccel_loss'][1::2]
        vsblvalidsqrt_accel_loss = data['vsblvalidrollout_pos_error'][1::2]

        loss_dir[os.path.join(exp_name,trial)] = vsblvalidsqrt_accel_loss

# plot the data
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['Flat Scenario', 'Platform Scenario', 'Sphere Scenario', 'Rod Scenario']
i=0
for key, value in loss_dir.items():
    ax.plot(value[:min(value.shape[0],50)], label=labels[i],linewidth=3.0)
    i+=1
ax.set_xlabel('Iteration(Epoch)')
ax.set_ylabel('Acceleration Loss(m)')

ax.set_ylabel('Rollout Pos Error(m)')
# ax.set_title('Rollout Pos Error')
ax.legend()
# Add a grid
plt.grid(True, alpha=0.5)

plt.show()
