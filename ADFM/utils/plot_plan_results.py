import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.style'] = 'normal'
#make grid in figure
plt.rcParams['axes.grid'] = True

def get_print_performance(task):

    performance_infos = np.load('../data/plan_all/log_plan_{}/performance_infos.npy'.format(task), allow_pickle=True)
    performance_all = np.load('../data/plan_all/log_plan_{}/performance_all.npy'.format(task), allow_pickle=True)
    MPE = performance_infos[:,0]
    iou = performance_infos[:, 2]
    NI = performance_infos[:,1]
    MPE_flat = MPE[::5]
    MPE_platform = MPE[1::5]
    MPE_sphere = MPE[2::5]
    MPE_rod = MPE[3::5]
    MPE_table = MPE[4::5]
    MPE_generalized = np.concatenate((MPE_platform, MPE_sphere, MPE_rod))
    print('MPE----------------MPE----------------MPE')
    print('task-{}:mean and std of flat: {}, {}'.format(task, np.mean(MPE_flat), np.std(MPE_flat)))
    print('task-{}:mean and std of platform: {}, {}'.format(task, np.mean(MPE_platform), np.std(MPE_platform)))
    print('task-{}:mean and std of sphere: {}, {}'.format(task, np.mean(MPE_sphere), np.std(MPE_sphere)))
    print('task-{}:mean and std of rod: {}, {}'.format(task, np.mean(MPE_rod), np.std(MPE_rod)))
    print('task-{}:mean and std of generalized: {}, {}'.format(task, np.mean(MPE_generalized), np.std(MPE_generalized)))
    print('task-{}:mean and std of table: {}, {}'.format(task, np.mean(MPE_table), np.std(MPE_table)))

    iou_flat = iou[::5]
    iou_platform = iou[1::5]
    iou_sphere = iou[2::5]
    iou_rod = iou[3::5]
    iou_table = iou[4::5]
    iou_generalized = np.concatenate((iou_platform, iou_sphere, iou_rod))
    print('IOU----------------IOU----------------IOU')
    print('task-{}:mean and std of flat: {}, {}'.format(task, np.mean(iou_flat), np.std(iou_flat)))
    print('task-{}:mean and std of platform: {}, {}'.format(task, np.mean(iou_platform), np.std(iou_platform)))
    print('task-{}:mean and std of sphere: {}, {}'.format(task, np.mean(iou_sphere), np.std(iou_sphere)))
    print('task-{}:mean and std of rod: {}, {}'.format(task, np.mean(iou_rod), np.std(iou_rod)))
    print('task-{}:mean and std of generalized: {}, {}'.format(task, np.mean(iou_generalized), np.std(iou_generalized)))
    print('task-{}:mean and std of table: {}, {}'.format(task, np.mean(iou_table), np.std(iou_table)))

    NI_flat = NI[::5]
    NI_platform = NI[1::5]
    NI_sphere = NI[2::5]
    NI_rod = NI[3::5]
    NI_table = NI[4::5]
    NI_generalized = np.concatenate((NI_platform, NI_sphere, NI_rod))
    print('NI----------------NI----------------NI')
    print('task-{}:mean and std of flat: {}, {}'.format(task, np.mean(NI_flat), np.std(NI_flat)))
    print('task-{}:mean and std of platform: {}, {}'.format(task, np.mean(NI_platform), np.std(NI_platform)))
    print('task-{}:mean and std of sphere: {}, {}'.format(task, np.mean(NI_sphere), np.std(NI_sphere)))
    print('task-{}:mean and std of rod: {}, {}'.format(task, np.mean(NI_rod), np.std(NI_rod)))
    print('task-{}:mean and std of generalized: {}, {}'.format(task, np.mean(NI_generalized), np.std(NI_generalized)))
    print('task-{}:mean and std of table: {}, {}'.format(task, np.mean(NI_table), np.std(NI_table)))

    success_rate_flat = np.sum(NI_flat > 0.95) / len(NI_flat)
    success_rate_platform = np.sum(NI_platform > 0.95) / len(NI_platform)
    success_rate_sphere = np.sum(NI_sphere > 0.95) / len(NI_sphere)
    success_rate_rod = np.sum(NI_rod > 0.95) / len(NI_rod)
    success_rate_table = np.sum(NI_table > 0.95) / len(NI_table)
    success_rate_generalized = np.sum(NI_generalized > 0.95) / len(NI_generalized)
    print('success rate----------------success rate----------------success rate')
    print('task-{}:success rate of flat: {}'.format(task, success_rate_flat))
    print('task-{}:success rate of platform: {}'.format(task, success_rate_platform))
    print('task-{}:success rate of sphere: {}'.format(task, success_rate_sphere))
    print('task-{}:success rate of rod: {}'.format(task, success_rate_rod))
    print('task-{}:success rate of generalized: {}'.format(task, success_rate_generalized))
    print('task-{}:success rate of table: {}'.format(task, success_rate_table))


    NI_all = [performance[1] for performance in performance_all]

    performance_all_flat = NI_all[::5]
    performance_all_platform = NI_all[1::5]
    performance_all_sphere = NI_all[2::5]
    performance_all_rod = NI_all[3::5]
    performance_all_table = NI_all[4::5]

    for i in range(4):
        performance_all_flat[i] = np.concatenate((performance_all_flat[i],np.repeat(performance_all_flat[i][-1:],20)))[:20]
        performance_all_platform[i] = np.concatenate((performance_all_platform[i],np.repeat(performance_all_platform[i][-1:],20)))[:20]
        performance_all_sphere[i] = np.concatenate((performance_all_sphere[i],np.repeat(performance_all_sphere[i][-1:],20)))[:20]
        performance_all_rod[i] = np.concatenate((performance_all_rod[i],np.repeat(performance_all_rod[i][-1:],20)))[:20]
        performance_all_table[i] = np.concatenate((performance_all_table[i],np.repeat(performance_all_table[i][-1:],20)))[:20]

    # Calculate means and standard deviations
    mean1 = np.mean(performance_all_flat, axis=0)
    std1 = np.std(performance_all_flat, axis=0)

    mean2 = np.mean(performance_all_platform, axis=0)
    std2 = np.std(performance_all_platform, axis=0)

    mean3 = np.mean(performance_all_sphere, axis=0)
    std3 = np.std(performance_all_sphere, axis=0)

    mean4 = np.mean(performance_all_rod, axis=0)
    std4 = np.std(performance_all_rod, axis=0)

    mean5 = np.mean(performance_all_table, axis=0)
    std5 = np.std(performance_all_table, axis=0)
    return mean1, std1, mean2, std2, mean3, std3, mean4, std4, mean5, std5

tasks = ['flingbot','irp1','irp2','irp3', 'general']

# Set up the subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
labels = ['Flingbot','IRP1','IRP2','IRP3','ADFM(Ours)']
for i, task in enumerate(tasks):
    mean1, std1, mean2, std2, mean3, std3, mean4, std4, mean5, std5 = get_print_performance(task)
    # axs[0, 0].plot(mean1, label=labels[i], linewidth=2)
    # axs[0, 0].fill_between(range(len(mean1)), mean1-std1, mean1+std1, alpha=0.1)
    # axs[0, 0].set_title('Flat Scenario')
    # axs[0, 0].legend()

    axs[0, 0].plot(mean2, label=labels[i], linewidth=2)
    axs[0, 0].fill_between(range(len(mean2)), mean2-std2, mean2+std2, alpha=0.1)
    axs[0, 0].set_title('Platform Scenario')

    axs[0, 1].plot(mean3, label=labels[i], linewidth=2)
    axs[0, 1].fill_between(range(len(mean3)), mean3-std3, mean3+std3, alpha=0.1)
    axs[0, 1].set_title('Hemisphere Scenario')

    axs[1, 0].plot(mean4, label=labels[i], linewidth=2)
    axs[1, 0].fill_between(range(len(mean4)), mean4-std4, mean4+std4, alpha=0.1)
    axs[1, 0].set_title('Pole Scenario')

    axs[1, 1].plot(mean5, label=labels[i], linewidth=2)
    axs[1, 1].fill_between(range(len(mean5)), mean5-std5, mean5+std5, alpha=0.1)
    axs[1, 1].set_title('Table Scenario')

    # axs[0, 0].legend()
    # axs[0, 1].legend()
    # axs[1, 0].legend()
    # axs[1, 1].legend()


# Set up the subplots
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot for the first set

# axs[0, 0].plot(flingbot1, label='Flingbot', color='pink', linewidth=2)
# axs[0, 0].fill_between(range(len(mean1)), flingbot1-flingbot1_std, flingbot1+flingbot1_std, alpha=0.1, color='pink')
# axs[0, 0].plot(irp1, label='IRP', color='orange', linewidth=2)
# axs[0, 0].fill_between(range(len(mean1)), irp1-irp1_std, irp1+irp1_std, alpha=0.1, color='orange')
# axs[0, 0].plot(simple1, label='SSDM', color='blue', linewidth=2)
# axs[0, 0].fill_between(range(len(mean1)), simple1-simple1_std, simple1+simple1_std, alpha=0.1, color='blue')
#
# axs[0, 0].plot(mean1, label='GSDM(Ours)', color='green', linewidth=2)
# axs[0, 0].fill_between(range(len(mean1)), mean1-std1, mean1+std1, alpha=0.1, color='green')
#
# axs[0, 0].set_title('Flat Scenario')
# axs[0, 0].legend()
#
# # Plot for the second set
#
# axs[0, 1].plot(flingbot2, label='Flingbot', color='pink', linewidth=2)
# axs[0, 1].fill_between(range(len(mean2)), flingbot2-flingbot2_std, flingbot2+flingbot2_std, alpha=0.1, color='pink')
# axs[0, 1].plot(irp2, label='IRP', color='orange', linewidth=2)
# axs[0, 1].fill_between(range(len(mean2)), irp2-irp2_std, irp2+irp2_std, alpha=0.1, color='orange')
# axs[0, 1].plot(simple2, label='SSDM', color='blue', linewidth=2)
# axs[0, 1].fill_between(range(len(mean2)), simple2-simple2_std, simple2+simple2_std, alpha=0.1, color='blue')
# axs[0, 1].plot(mean2, label='GSDM(Ours)',color='green', linewidth=2)
# axs[0, 1].fill_between(range(len(mean2)), mean2-std2, mean2+std2, alpha=0.2, color='green')
# axs[0, 1].set_title('Platform Scenario')
# axs[0, 1].legend()
#
# # Plot for the third set
# axs[1, 0].plot(flingbot3, label='Flingbot', color='pink', linewidth=2)
# axs[1, 0].fill_between(range(len(mean3)), flingbot3-flingbot3_std, flingbot3+flingbot3_std, alpha=0.1, color='pink')
# axs[1, 0].plot(irp3, label='IRP', color='orange', linewidth=2)
# axs[1, 0].fill_between(range(len(mean3)), irp3-irp3_std, irp3+irp3_std, alpha=0.1, color='orange')
# axs[1, 0].plot(simple3, label='SSDM', color='blue', linewidth=2)
# axs[1, 0].fill_between(range(len(mean3)), simple3-simple3_std, simple3+simple3_std, alpha=0.1, color='blue')
# axs[1, 0].plot(mean3, label='GSDM(Ours)',color='green', linewidth=2)
# axs[1, 0].fill_between(range(len(mean3)), mean3-std3, mean3+std3, alpha=0.2, color='green')
# axs[1, 0].set_title('Sphere Scenario')
# axs[1, 0].legend()
#
# # Plot for the fourth set
# axs[1, 1].plot(flingbot4, label='Flingbot', color='pink', linewidth=2)
# axs[1, 1].fill_between(range(len(mean4)), flingbot4-flingbot4_std, flingbot4+flingbot4_std, alpha=0.1, color='pink')
# axs[1, 1].plot(irp4, label='IRP', color='orange', linewidth=2)
# axs[1, 1].fill_between(range(len(mean4)), irp4-irp4_std, irp4+irp4_std, alpha=0.1, color='orange')
# axs[1, 1].plot(simple4, label='SSDM', color='blue', linewidth=2)
# axs[1, 1].fill_between(range(len(mean4)), simple4-simple4_std, simple4+simple4_std, alpha=0.1, color='blue')
# axs[1, 1].plot(mean4, label='GSDM(Ours)',color='green', linewidth=2)
# axs[1, 1].fill_between(range(len(mean4)), mean4-std4, mean4+std4, alpha=0.2, color='green')
# axs[1, 1].set_title('Rod Scenario')
# axs[1, 1].legend()

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()

# Adjust layout to prevent overlap
fig.tight_layout()

# Ticks formatting
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.legend(fontsize=10)

# Setting the grid
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Minor ticks
# plt.minorticks_on()

# Save the figure
plt.savefig('experiment_trajectories.png')

# Show the figure
plt.show()
