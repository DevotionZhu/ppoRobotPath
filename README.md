# ppoRobotPath
Notice:
This repository is the codes for my recent paper: **Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning** which is submitted for **ICRA 2019**

This code employs an state-of-the-art deep reinforcement learning approach, Proximal Policy Optimization (PPO), for online trajectory generation of industrial robotic arms. We also introduced the RobotPath simulation environment for kinematic and collision simulation of industrial robots.

## Simulation results:

<p align="center">
    <img src="https://github.com/kavehkamali/ppoRobotPath/blob/master/train.gif" width="370">
    <img src="https://github.com/kavehkamali/ppoRobotPath/blob/master/test.gif" width="430">
</p>

## Experiments on a real ABB robot:

<p align="center">
    <img src="https://github.com/kavehkamali/ppoRobotPath/blob/master/experiment.jpg" width="380">
    <img src="https://github.com/kavehkamali/ppoRobotPath/blob/master/demo.gif" width="460">
</p>


## Installation:
1- Install pybullet:

```
pip install pybullet
```
2- Install OpenAI baselines:

https://github.com/openai/baselines
For installing baselines you need to run:

```
pip install -e .
```
Note 1: OpenAI baselines will install gym which needs MuJoco license. We do not need MuJoco so just remove the words: "mujoco,robotics" from "setup.py" before running the above command.

Note 2: Atary module of gym cannot be installed on windows. So remove the word: "Atari" from "setup.py" before running it.

3- install pyquaternion:

```
pip install pyquaternion
```
4- Install MPI on your system and put its path in the PATH environmental variable.

## Training:
Run RobotPath training:

```
mpirun -np 40 python train.py
```
The above command is for 40 cpu but you can choose any number. We recomment to use at least 25 cpus.


## Test:
Run RobotPath test:

```
python test.py
```
