<h1> One Fling to Goal: Environment-aware Dynamics for Goal-conditioned Fabric Flinging</h1>

[PDF](https://arxiv.org/pdf/2406.14136)

# Table of Contents
- 1 [Simulation](#simulation)
    - 1.1 [Setup](#setup)
    - 1.2 [Data Collection](#data-collection)
- 2 [Real Robot](#real-robot)
    - 2.1 [Setup](#real-setup) 
    - 
----
# Simulation

## Setup

This project is based on softgym simulation, consider to follow the [SoftGym](https://danieltakeshi.github.io/2021/02/20/softgym/) setup.

You can clone an individual [softgym](https://github.com/Xingyu-Lin/softgym) to build the pyflex(C++ part). We do not revise the pyflex part in this project. But you have to use the softgym python part in this project(especially the [envs](softgym%2Fsoftgym%2Fenvs) folder) to run the simulation.

```commandline

- Create conda env
```commandline
$ cd ${Project Path}/softgym
$ conda env create -f environment.yml
```
- Install docker (if not installed) refer to [Docker](https://docs.docker.com/engine/install/ubuntu/)
```commandline
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

- Install nvidia-docker (if not installed) refer to [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
maybe not used?
```commandline
$ apt-get install ...
```

- Install pyflex ( for not ubuntu 22.04: use nvidia-docker instead of docker, add --gpus all)
```commandline
conda activate adfm
docker pull xingyu/softgym

sudo docker run -v /home/yang/Projects/softgym:/workspace/softgym 
-v /home/yang/anaconda3/:/home/yang/anaconda3/ 
-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY 
-e QT_X11_NO_MITSHM=1 -it xingyu/softgym:latest bash

cd softgym/
export PATH="/home/yang/anaconda3/bin:$PATH"
. ./prepare_1.0.sh
. ./compile_1.0.sh
```

- back to ubuntu, add these to the .bashrc

```commandline
echo 'export PYFLEXROOT=${PROJECT_DIR}/PyFlex' >> ~/.bashrc
echo 'export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

- install other packages

```commandline
pip install hydra-core --upgrade
pip install h5py
pip install wandb==0.15.4

# please revise revision accordingly 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cu113.html

conda install -c sirokujira python-pcl --channel conda-forge 
```

- debug: ImportError: libboost_system.so.1.54.0: cannot open shared object file: No such file or directory

```commandline
cd anaconda3/envs/adfm/lib
ln -s libboost_system.so.1.64.0 libboost_system.so.1.54.0
ln -s libboost_filesystem.so.1.64.0 libboost_filesystem.so.1.54.0
ln -s libboost_thread.so.1.64.0 libboost_thread.so.1.54.0
ln -s libboost_iostreams.so.1.64.0 libboost_iostreams.so.1.54.0
```

## Data Collection

- collect generalized dataset

```commandline
python3 main.py exp_name=gen_data task=gen_data dataf=./data/data_general log_dir=./data/data_general gen_gif=1 n_rollout=10000 cached_states_path=adfm_general.pklnum_variations=1000 vary_cloth_size=true vary_stiffness=true vary_orientation=true vary_mass=true env_shape=random
```

- train generalized dynamic model 
```commandline
python3 main.py exp_name=train_dy task=train_dy dataf=./data/data_general log_dir=./data/data_general/train_dy gen_gif=1 n_rollout=10000 cached_states_path=adfm_general.pkl num_variations=1000 env_shape=random task.use_wandb=false
```

- plan
```commandline
python3 main.py exp_name=plan task=plan dataf=./data/data_general log_dir=./data/plan/ cached_states_path=adfm_all.pkl num_variations=20 env_shape=all task.partial_dyn_path=./data/data_general/train_dy/vsbl_dyn_best.pth vary_cloth_size=true vary_stiffness=true vary_orientation=true vary_mass=true
```

# Real Robot

## Real Setup
1. [Install ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) (Unbuntu 20.04)

3. [Install Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
```commandline
sudo apt install ros-noetic-ur-robot-driver
```
Or build from source
```
cd ${Project Path}/real_robot
# clone the driver
$ git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver

# clone the description. Currently, it is necessary to use the melodic-devel branch.
$ git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git src/universal_robot

# install dependencies
$ sudo apt update -qq
$ rosdep update
$ rosdep install --from-paths src --ignore-src -y

# build the workspace
$ catkin_make

# activate the workspace (ie: source it)
$ source devel/setup.bash
```

   - Set up the URCup driver on the robot and network conenction
   - Calibration
   ```
    roslaunch ur_calibration calibration_correction.launch robot_ip:=192.168.50.8 target_filename:="${HOME}/my_robot_calibration.yaml"
   ```

3. [Install Realsense ROS Driver](https://github.com/IntelRealSense/realsense-ros/blob/ros1-legacy/README.md#installation-instructions)
      
  - Step 1: Install the latest Intel&reg; RealSense&trade; SDK 2.0
     -  install from [Linux Debian Installation Guide](https://github.com/IntelRealSense/realsense-ros/blob/ros1-legacy/README.md#installation-instructions)
     -  Or Build from sources by downloading the latest [Intel&reg; RealSense&trade; SDK 2.0](https://github.com/IntelRealSense/librealsense/releases/tag/v2.50.0) and follow the instructions under [Linux Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
```commandline

#Register the server's public key:
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
#Make sure apt HTTPS support is installed: sudo apt-get install apt-transport-https

#Add the server to the list of repositories:

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

#Install the libraries (see section below if upgrading packages):
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils

#The above two lines will deploy librealsense2 udev rules, build and activate kernel modules, runtime library and executable demos and tools.
#Optionally install the developer and debug packages:
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg

```
  - Step 2: Install Intel&reg; RealSense&trade; ROS from Sources

```
$ git clone https://github.com/IntelRealSense/realsense-ros.git
$ cd realsense-ros/
$ git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
$ cd ..

$ catkin_init_workspace
$ cd ..
$ catkin_make clean
$ catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
$ catkin_make install
```

```commandline
echo "source /home/yang/Projects/ADFM/ADFM/real_robot/devel/setup.bash" >> ~/.bashrc
```

4. Install gripper driver
Pre: [Setting up the tool communication on an e-Series robot](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/setup_tool_communication.md)
- Only instsall the rs485 urcap; not the gripper urcap

```commandline
# clone the gripper driver
git clone https://github.com/rxjia/robotiq/tree/noetic-devel

# run the gripper driver
roslaunch robotiq_2f_gripper_action_server robotiq_2f_gripper_as_client.launch port:=/tmp/ttyUR16
```

5. Hand-Eye Calibration
```commandline
roslaunch robot_control ur10e.launch 
roslaunch ur10e_moveit_config moveit_planning_execution.launch 
roslaunch ur10e_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur10e_moveit_config)/launch/moveit.rviz
```

## Run Experiments

### To start the camera node in ROS:

```
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

### To start the robot node in ROS:

```commandline
roslaunch robot_control ur10e.launch 
roslaunch robot_control ur16e.launch 
roslaunch robot_control gripper_ur16e.launch 
roslaunch robot_control gripper_ur10e.launch 
```

# Troubleshooting

1. When try to ```catkin_make``` on ubuntu20.04 get error:
    Unable to find either executable ‘empy’ or Python module ‘em’…
Use alternate python intepreter:
```commandline
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```
2. substitution args not supported:  No module named 'rospkg'

```commandline
pip install -U rosdep rosinstall_generator wstool rosinstall six vcstools defusedxml
```

3. catkin_make error: CMake Error at /opt/ros/noetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package): Could not find a package configuration file provided by "moveit_core" with any of the following names: moveit_coreConfig.cmake moveit_core-config.cmake

```commandline
# install the missed package, for example
sudo apt-get install ros-noetic-moveit-core
```
4. Debugger error with OmegaConf package - 'TypeError: OmegaConfUserResolver.get_str() takes 2 positional arguments but 3 were given'

Line 72 of the _pydevd_bundle/pydevd_extension_api.py file in pycharm2023.3 has an extra do_trim parameter, but the get_str function in https://github.com/fabioz/PyDev.Debugger does not have this parameter.
image1.png

Simple solution：
Find the pydevd_plugin_omegaconf.py file in the pydevd_plugins package, and add the *arg, **kwargs parameters to the get_str function on line 100.
For example, this is the location of my pydevd_plugin_omegaconf.py file："/miniconda3/envs/llm-base/Lib/site-packages/pydevd_plugins/extensions/pydevd_plugin_omegaconf.py"

ref:
https://youtrack.jetbrains.com/issue/PY-64588/Debugger-error-with-OmegaConf-package-TypeError-OmegaConfUserResolver.getstr-takes-2-positional-arguments-but-3-were-given

# References
If you run into problems setting up SoftGym, Daniel Seita wrote a nice blog that may help you get started on SoftGym: https://danieltakeshi.github.io/2021/02/20/softgym/

# Citation
If you find this work useful, please consider citing:
```
@article{yang2024one,
  title={One Fling to Goal: Environment-aware Dynamics for Goal-conditioned Fabric Flinging},
  author={Yang, Linhan and Yang, Lei and Sun, Haoran and Zhang, Zeqing and He, Haibin and Wan, Fang and Song, Chaoyang and Pan, Jia},
  journal={arXiv preprint arXiv:2406.14136},
  year={2024}
}
```
