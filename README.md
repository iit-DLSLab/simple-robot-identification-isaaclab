  <div style="display: flex; justify-content: space-around;">
    <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-blue" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
    <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-blue" alt="Mujoco v3.7.0" style="margin-bottom: 1px;">
    <img src="./gifs/z1_real.gif" alt="Z1Real" width="32%">
    <img src="./gifs/z1_sim.gif" alt="Z1Sim" width="29.3%">
    <img src="./gifs/sim-to-real.gif" alt="Sim-to-Real" width="32%">
  </div>


## Overwiew

A joint calibration routine for IsaacLab and Mujoco, to estimate leg motor parameters. It provides scripts for data collection on the real robot (the robot should be in the air with the base fixed). 
This repo interfaces directly with [Pace](https://github.com/leggedrobotics/pace-sim2real) and the new sys-id functionality of Mujoco.



The models (usd and xml) identified so far can be found in the folder **robot_model**. 


## Run a collection
This repo works best with [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for communicating with unitree go2, b2, a2, and z1 robots. Soon, will support agilex piper arms using [piper-ros2-dls2](https://github.com/iit-DLSLab/piper-ros2-dls2).

1. Choose the robot and the gains in the  [config file](https://github.com/iit-DLSLab/sim2real-robot-identification/blob/main/config.py)

2. In the xml of your robot, add two keyframe (sys_id_1, sys_id_2) to define the start and end point of the chirp trajectory (see [here](https://github.com/iit-DLSLab/sim2real-robot-identification/blob/60e7e48a382dc4293e80062e2bd3f9dc70b7cfc8/robot_model/go2/go2.xml#L252) for an example)

3. Runs one of the following files
```bash
python3 run_collection_quadruped_ros2.py
python3 run_collection_manipulator_ros2.py
```
modifying inside USE_MUJOCO_RENDER and USE_MUJOCO_SIMULATION depending on your usecase


4. Visualize your trajectory running
```bash
python3 datasets/replay_dataset_quadruped.py
python3 datasets/replay_dataset_manipulator.py
```

## Run a calibration in IsaacLab

```bash
python3 sysid_isaaclab/my_fit.py --headless
```

## Run a calibration in Mujoco

```bash
python3 sysid_mujoco/my_fit.py 
```

## How to contribute

PRs are very welcome (search for **TODO** in the issue, or add what you like)!


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi) and [Lorenzo Amatucci](https://github.com/lorenzo96-cmd).
