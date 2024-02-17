# MAVRL
MAVRL: Learn to Fly in Cluttered Environments with Varying Speed

# 1. Introduction
Many existing obstacle avoidance algorithms overlook the crucial balance between safety and agility, especially in environments of varying complexity. In our study, we introduce an obstacle avoidance pipeline based on reinforcement learning. This pipeline enables drones to adapt their flying speed according to the environmental complexity. Moreover, to improve the obstacle avoidance performance in cluttered environments, we propose a novel latent space. The latent space in this representation is explicitly trained to retain memory of previous depth map observations. Our findings confirm that varying speed leads to a superior balance of success rate and agility in cluttered environments. Additionally, our memory-augmented latent representation outperforms the latent representation commonly used in reinforcement learning. Finally, after minimal fine-tuning, we successfully deployed our network on a real drone for enhanced obstacle avoidance.

# 2. Installation

## 2.1 Install AvoidBench
Please refer to [AvoidBench](https://github.com/tudelft/AvoidBench) and check the dependency of installation. Run the following commands to setup:
``` bash
sudo apt update
sudo apt install libzmqpp-dev libopencv-dev unzip python3-catkin-tools
sudo apt install libgoogle-glog-dev protobuf-compiler ros-noetic-octomap-msgs ros-noetic-octomap-ros python3-vcstool
git clone git@github.com:tudelft/AvoidBench.git
cd AvoidBench
git checkout mavrl_version
echo "export AVOIDBENCH_PATH=path_to_this_project/AvoidBench/src/avoidbench" >> ~/.bashrc
```

## 2.1 Install MAVRL
Get the mavrl ros package:
``` bashs
cd AvoidBench/src
git clone git@github.com:tudelft/mavrl.git
```
