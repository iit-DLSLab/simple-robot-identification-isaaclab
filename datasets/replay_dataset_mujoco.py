# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
import config

import numpy as np
import copy
import time
import mujoco

import torch

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    simulation_dt = 0.002


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=simulation_dt,
        base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    )


    env.reset(random=False)
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType

    freezed_base_position = np.array([0, 0, 0.4])
    freezed_base_orientation = np.array([1, 0, 0, 0])
    freezed_base_linear_velocity = np.array([0, 0, 0.])
    freezed_base_angular_velocity = np.array([0, 0, 0])
    

    # Load datasets for calibration
    datasets_path = dir_path + "/" + config.robot + "/traj_0.pt"
    data = torch.load(datasets_path)
    dataset_actual_joint_pos = data["dof_pos"]
    dataset_time = data["time"]

    timestep = 0

    while True:
        
        print(f"Running timestep: {timestep}")
        
        if(timestep >= dataset_actual_joint_pos.shape[0] - 1):
            print("End of dataset reached, resetting to the beginning.")
            break
        
        joint_pos = dataset_actual_joint_pos[timestep]
        
        env.mjData.qpos[0:3] = copy.deepcopy(freezed_base_position)
        env.mjData.qpos[3:7] = copy.deepcopy(freezed_base_orientation)
        env.mjData.qvel[0:3] = copy.deepcopy(freezed_base_linear_velocity)
        env.mjData.qvel[3:6] = copy.deepcopy(freezed_base_angular_velocity)
        env.mjData.qpos[7:19] = copy.deepcopy(joint_pos)
        env.mjModel.opt.timestep = simulation_dt


        mujoco.mj_forward(env.mjModel, env.mjData) 
        timestep += 1

        env.render()
        #time.sleep(float(dataset_time[timestep]))