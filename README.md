## Overwiew

A joint calibration routine for IsaacLab and Mujoco, to estimate leg motor parameters. It provides scripts for data collection on the real robot (the robot should be in the air with the base fixed). 
This repo interfaces directly with [Pace](https://github.com/leggedrobotics/pace-sim2real) and the new sys-id functionality of Mujoco.


Work in progress for supporting mujoco identification, PRs are very welcome!


## Run a collection
This repo works best with [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for communicating with unitree go2, b2, a2, and z1 robots. Soon, will support agilex piper arms using [piper-ros2-dls2](https://github.com/iit-DLSLab/piper-ros2-dls2).

1. Choose the robot and the gains in the config file

2. Runs one of the following files
```bash
python3 run_collection_quadruped_ros2.py
python3 run_collection_manipulator_ros2.py
```
modifying inside USE_MUJOCO_RENDER and USE_MUJOCO_SIMULATION depending on your usecase


3. Visualize your trajectory runnining
```bash
python3 datasets/replay_dataset_quadruped.py
python3 datasets/replay_dataset_manipulator.py
```

## Run a calibration in IsaacLab

```bash
python3 sysid_isaaclab/my_fit.py --headless
```


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).
