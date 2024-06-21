import imageio
import os
images = []

for i in range(1,100):
    filename = '../data/realrobot/episode_stool/img_target_{}.png'.format(i)
    if os.path.isfile(filename):
        images.append(imageio.imread(filename))

imageio.mimsave('../data/video/stool.mp4', images, fps=10)