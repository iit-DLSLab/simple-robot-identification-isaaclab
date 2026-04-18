"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

import numpy as np
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab.assets import Articulation

# Import extensions to set up environment tasks
import basic_locomotion_dls_isaaclab.tasks  # noqa: F401

import utility
import config
from isaaclab.managers import SceneEntityCfg


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )


    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )


    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)


    # reset environment
    env_ids = torch.arange(args_cli.num_envs, device=env.unwrapped.device)  # Updated to include all env IDs


    expected_joint_order = [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ]
    datasets_path = config.datasets_path
    datasets = utility.load_datasets(datasets_path, expected_joint_order)
    all_dataset_actual_joint_pos = datasets["all_dataset_actual_joint_pos"]
    all_dataset_actual_joint_vel = datasets["all_dataset_actual_joint_vel"]
    all_dataset_desired_joint_pos = datasets["all_dataset_desired_joint_pos"]
    all_dataset_desired_joint_vel = datasets["all_dataset_desired_joint_vel"]
    dataset_fps = datasets["dataset_fps"]


    freezed_base_positions = torch.tensor([0, 0, 0.8], dtype=torch.float32, device=env.device).repeat(args_cli.num_envs,1)
    freezed_base_velocities = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32, device=env.device).repeat(args_cli.num_envs,1)
    freezed_base_orientations = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=env.device).repeat(args_cli.num_envs,1)

    # space the base positions over a 20m x 20m square grid
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(args_cli.num_envs, dtype=torch.float32))))
    dimension_world = 200.0 #meters
    spacing = dimension_world / (grid_size - 1) if grid_size > 1 else 0.0
    
    for i in range(args_cli.num_envs):
        row = i // grid_size
        col = i % grid_size
        freezed_base_positions[i, 0] += col * spacing - dimension_world/2.  # Center around origin 
        freezed_base_positions[i, 1] += row * spacing - dimension_world/2.  # Center around origin 
    freezed_base_orientations = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)



    def single_iteration(num_best_candidates):
        timestep = 0

        # Sample different Kp and Kd values for each environment
        nominal_kp = config.Kp
        nominal_kd = config.Kd

        if(config.optimize_gain):
            # Sample in steps withing the bounds
            search_kp_bounds = config.search_Kp_bounds
            search_kd_bounds = config.search_Kd_bounds
            kp_min = nominal_kp + search_kp_bounds[0]
            kp_max = nominal_kp + search_kp_bounds[1]
            kd_min = nominal_kd + search_kd_bounds[0]
            kd_max = nominal_kd + search_kd_bounds[1]

            # with a given sampling interval
            kp_num_steps = int((kp_max - kp_min) / config.Kp_sampling_interval) + 1
            kd_num_steps = int((kd_max - kd_min) / config.Kd_sampling_interval) + 1

            # Sample random step indices
            kp_step_indices = torch.randint(0, kp_num_steps, (args_cli.num_envs, 3), device=env.device)
            kd_step_indices = torch.randint(0, kd_num_steps, (args_cli.num_envs, 3), device=env.device)

            # Convert to actual values
            kp_values = kp_min + kp_step_indices * config.Kp_sampling_interval
            kd_values = kd_min + kd_step_indices * config.Kd_sampling_interval
        else:
            #kp_values = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat(args_cli.num_envs,1)
            #kd_values = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat(args_cli.num_envs,1)
            kp_values = torch.tensor(nominal_kp, device=env.device).repeat(args_cli.num_envs,3)
            kd_values = torch.tensor(nominal_kd, device=env.device).repeat(args_cli.num_envs,3)

        # Apply the Kp and Kd values to the robot's joints
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        asset: Articulation = env.unwrapped.scene[asset_cfg.name]
        asset.actuators["hip"].stiffness = kp_values[:, 0].unsqueeze(1).repeat(1, 4)
        asset.actuators["thigh"].stiffness = kp_values[:, 1].unsqueeze(1).repeat(1, 4)
        asset.actuators["calf"].stiffness = kp_values[:, 2].unsqueeze(1).repeat(1, 4)
        asset.actuators["hip"].damping = kd_values[:, 0].unsqueeze(1).repeat(1, 4)
        asset.actuators["thigh"].damping = kd_values[:, 1].unsqueeze(1).repeat(1, 4)
        asset.actuators["calf"].damping = kd_values[:, 2].unsqueeze(1).repeat(1, 4)



        # Sample different friction static and dynamic values for each environment
        nominal_friction_static = config.friction_static
        nominal_friction_dynamic = config.friction_dynamic
        
        if(config.optimize_friction):
            # Sample in steps withing the bounds
            search_friction_static_bounds = config.search_friction_static_bounds
            search_friction_dynamic_bounds = config.search_friction_dynamic_bounds
            friction_static_min = nominal_friction_static + search_friction_static_bounds[0]
            friction_static_max = nominal_friction_static + search_friction_static_bounds[1]
            friction_dynamic_min = nominal_friction_dynamic + search_friction_dynamic_bounds[0]
            friction_dynamic_max = nominal_friction_dynamic + search_friction_dynamic_bounds[1]

            # with a given sampling interval
            friction_static_num_steps = int((friction_static_max - friction_static_min) / config.friction_static_sampling_interval) + 1
            friction_dynamic_num_steps = int((friction_dynamic_max - friction_dynamic_min) / config.friction_dynamic_sampling_interval) + 1

            # Sample random step indices
            friction_static_step_indices = torch.randint(0, friction_static_num_steps, (args_cli.num_envs,), device=env.device)
            friction_dynamic_step_indices = torch.randint(0, friction_dynamic_num_steps, (args_cli.num_envs,), device=env.device)
            
            # Convert to actual values
            friction_static_values = friction_static_min + friction_static_step_indices * config.friction_static_sampling_interval
            friction_dynamic_values = friction_dynamic_min + friction_dynamic_step_indices * config.friction_dynamic_sampling_interval
        else:
            friction_static_values = torch.tensor(nominal_friction_static, device=env.device).repeat(args_cli.num_envs)
            friction_dynamic_values = torch.tensor(nominal_friction_dynamic, device=env.device).repeat(args_cli.num_envs)
        
        # Apply the friction values to the robot
        asset.actuators["hip"].friction_static = friction_static_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["thigh"].friction_static = friction_static_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["calf"].friction_static = friction_static_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["hip"].friction_dynamic = friction_dynamic_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["thigh"].friction_dynamic = friction_dynamic_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["calf"].friction_dynamic = friction_dynamic_values.unsqueeze(1).repeat(1, 4)


        if(config.optimize_armature):
            # Sample in steps withing the bounds
            search_armature_bounds = config.search_armature_bounds
            armature_min = config.armature + search_armature_bounds[0]
            armature_max = config.armature + search_armature_bounds[1]

            # with a given sampling interval
            armature_num_steps = int((armature_max - armature_min) / config.armature_sampling_interval) + 1

            # Sample random step indices
            armature_step_indices = torch.randint(0, armature_num_steps, (args_cli.num_envs,), device=env.device)
            
            # Convert to actual values
            armature_values = armature_min + armature_step_indices * config.armature_sampling_interval
        else:
            armature_values = torch.tensor(config.armature, device=env.device).repeat(args_cli.num_envs)
        # Apply the armature values to the robot
        asset.actuators["hip"].armature = armature_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["thigh"].armature = armature_values.unsqueeze(1).repeat(1, 4)
        asset.actuators["calf"].armature = armature_values.unsqueeze(1).repeat(1, 4)

        error_joint_pos = torch.zeros(
            (args_cli.num_envs, len(env.unwrapped._robot.joint_names)), dtype=torch.float32, device=env.device 
        )
        error_joint_vel = torch.zeros(
            (args_cli.num_envs, len(env.unwrapped._robot.joint_names)), dtype=torch.float32, device=env.device
        )

        
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                
                print(f"Running timestep: {timestep}")
                
                if(timestep >= all_dataset_actual_joint_pos.shape[0] - 1):
                    print("End of dataset reached, resetting to the beginning.")
                    break
                
                env.unwrapped._robot.write_root_pose_to_sim(
                        torch.cat([freezed_base_positions, freezed_base_orientations], dim=-1), env_ids=env_ids
                )

                env.unwrapped._robot.write_root_velocity_to_sim(
                    freezed_base_velocities, env_ids=env_ids
                )
                
                joint_pos = torch.tensor(
                    all_dataset_actual_joint_pos[timestep], dtype=torch.float32, device=env.device
                )

                if(joint_pos == torch.tensor([-10.0] * joint_pos.shape[0], device=env.device)).all():
                    print("End of motion reached, reset initial robot configuration.")
                    
                    # Reset the robot to its initial configuration
                    joint_pos = torch.tensor(
                        all_dataset_actual_joint_pos[timestep+1], dtype=torch.float32, device=env.device
                    )
                    env.unwrapped._robot.write_joint_state_to_sim(
                        joint_pos, joint_pos*0.0, env_ids=env_ids
                    )

                    env.env.render()
                    time.sleep(1.0)
                
                else:
        
                    # Perform a simulation step   
                    desired_joint_pos = torch.tensor(
                        all_dataset_desired_joint_pos[timestep], dtype=torch.float32, device=env.device
                    )
                    
                    """env.unwrapped._robot.write_joint_state_to_sim(
                        joint_pos, joint_vel, env_ids=env_ids
                    )"""
                    if(desired_joint_pos == torch.tensor([-20.0] * joint_pos.shape[0], device=env.device)).all():
                        desired_joint_pos = desired_joint_pos*0.0
                        # the robot has zero control and is falling!
                        env.unwrapped._robot.set_joint_position_target(desired_joint_pos)

                        decimation = int((1/config.frequency_collection)/env.unwrapped.physics_dt)

                        #for _ in range(env.unwrapped.cfg.decimation):
                        for _ in range(decimation):
                            env.unwrapped.scene.write_data_to_sim()
                            # simulate
                            env.unwrapped.sim.step(render=False)
                            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)                        
                    else:
                        # control the robot with the desired joint positions
                        env.unwrapped._robot.set_joint_position_target(desired_joint_pos)

                        decimation = int((1/config.frequency_collection)/env.unwrapped.physics_dt)
                        
                        # We may not use the decimation of the environment, 
                        # because maybe we collect at a different frequency
                        #for _ in range(env.unwrapped.cfg.decimation):
                        for _ in range(decimation):
                            env.unwrapped.scene.write_data_to_sim()
                            # simulate
                            env.unwrapped.sim.step(render=False)
                            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

                    # And check the errors
                    joint_pos = torch.tensor(
                        all_dataset_actual_joint_pos[timestep+1], dtype=torch.float32, device=env.device
                    )

                    joint_vel = torch.tensor(
                        all_dataset_actual_joint_vel[timestep+1], dtype=torch.float32, device=env.device
                    )

                    if(joint_pos == torch.tensor([-10.0] * joint_pos.shape[0], device=env.device)).all():
                        print("##################")
                    else:
                        # Compute error between desired and actual joint positions
                        error_joint_pos[env_ids] += torch.abs(joint_pos - env.unwrapped._robot.data.joint_pos)
                        error_joint_vel[env_ids] += torch.abs(joint_vel - env.unwrapped._robot.data.joint_vel)
                    

                timestep += 1

                # env stepping
                env.env.render()

                time.sleep(1.0 / dataset_fps)

        
        # Print the average errors
        avg_error = error_joint_pos.mean(dim=1)# + error_joint_vel.mean(dim=1)
        #print("Average Joint Error:", avg_error)
        #print("Minimum Joint Error: ", avg_error.min())
        #print(f"Best Kp iteration: ", kp_values[avg_error.argmin()])
        #print(f"Best Kd iteration: ", kd_values[avg_error.argmin()])
        #print(f"Best Friction Static iteration: ", friction_static_values[avg_error.argmin()])
        #print(f"Best Friction Dynamic iteration: ", friction_dynamic_values[avg_error.argmin()])

        top_errors, top_indices = torch.topk(avg_error, k=min(num_best_candidates, args_cli.num_envs), largest=False)

        return top_errors, kp_values[top_indices], kd_values[top_indices], friction_static_values[top_indices], friction_dynamic_values[top_indices], armature_values[top_indices]



    num_iterations = config.num_iterations
    num_best_candidates = config.num_best_candidates

    # Initialize buffers to store the best candidates across all iterations
    best_errors_buffer = []
    best_kp_buffer = []
    best_kd_buffer = []
    best_friction_static_buffer = []
    best_friction_dynamic_buffer = []
    best_armature_buffer = []
    
    for j in range(num_iterations):
        print("Iteration: ", j)
        errors, \
        kp, \
        kd, \
        friction_static, \
        friction_dynamic, \
        armature = single_iteration(num_best_candidates)
        
        # Add new candidates to the buffer
        for i in range(len(errors)):
            candidate = {
                'error': errors[i].item(),
                'kp': kp[i].tolist(),
                'kd': kd[i].tolist(), 
                'friction_static': friction_static[i].item(),
                'friction_dynamic': friction_dynamic[i].item(),
                'armature': armature[i].item()
            }
            
            # Check if this candidate already exists in the buffer
            is_duplicate = False
            for existing in zip(best_kp_buffer, best_kd_buffer, best_friction_static_buffer, best_friction_dynamic_buffer, best_armature_buffer):
                if (torch.allclose(torch.tensor(candidate['kp']), torch.tensor(existing[0]), atol=1e-6) and
                    torch.allclose(torch.tensor(candidate['kd']), torch.tensor(existing[1]), atol=1e-6) and
                    abs(candidate['friction_static'] - existing[2]) < 1e-6 and
                    abs(candidate['friction_dynamic'] - existing[3]) < 1e-6 and
                    abs(candidate['armature'] - existing[4]) < 1e-6):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                best_errors_buffer.append(candidate['error'])
                best_kp_buffer.append(candidate['kp'])
                best_kd_buffer.append(candidate['kd'])
                best_friction_static_buffer.append(candidate['friction_static'])
                best_friction_dynamic_buffer.append(candidate['friction_dynamic'])
                best_armature_buffer.append(candidate['armature'])

        # Sort by error and keep only the best num_best_candidates
        if len(best_errors_buffer) > num_best_candidates:
            # Create indices sorted by error
            sorted_indices = sorted(range(len(best_errors_buffer)), key=lambda i: best_errors_buffer[i])
            
            # Keep only the best num_best_candidates
            best_errors_buffer = [best_errors_buffer[i] for i in sorted_indices[:num_best_candidates]]
            best_kp_buffer = [best_kp_buffer[i] for i in sorted_indices[:num_best_candidates]]
            best_kd_buffer = [best_kd_buffer[i] for i in sorted_indices[:num_best_candidates]]
            best_friction_static_buffer = [best_friction_static_buffer[i] for i in sorted_indices[:num_best_candidates]]
            best_friction_dynamic_buffer = [best_friction_dynamic_buffer[i] for i in sorted_indices[:num_best_candidates]]
            best_armature_buffer = [best_armature_buffer[i] for i in sorted_indices[:num_best_candidates]]

    print("Final Results:")
    print("Best Errors: ", best_errors_buffer)
    print("Best Kp: ", best_kp_buffer)
    print("Best Kd: ", best_kd_buffer)
    print("Best Friction Static: ", best_friction_static_buffer)
    print("Best Friction Dynamic: ", best_friction_dynamic_buffer)
    print("Best Armature: ", best_armature_buffer)

    # take bounds of the best candidates
    bound_kp = (torch.tensor(best_kp_buffer, device=env.device).min(dim=0)[0], torch.tensor(best_kp_buffer, device=env.device).max(dim=0)[0])
    bound_kd = (torch.tensor(best_kd_buffer, device=env.device).min(dim=0)[0], torch.tensor(best_kd_buffer, device=env.device).max(dim=0)[0])
    bound_friction_static = (torch.tensor(best_friction_static_buffer, device=env.device).min(dim=0)[0], torch.tensor(best_friction_static_buffer, device=env.device).max(dim=0)[0])
    bound_friction_dynamic = (torch.tensor(best_friction_dynamic_buffer, device=env.device).min(dim=0)[0], torch.tensor(best_friction_dynamic_buffer, device=env.device).max(dim=0)[0])
    bound_armature = (torch.tensor(best_armature_buffer, device=env.device).min(dim=0)[0], torch.tensor(best_armature_buffer, device=env.device).max(dim=0)[0])

    print("Bounds founds:")
    print("Best Kp: ", bound_kp)
    print("Best Kd: ", bound_kd)
    print("Best Friction Static: ", bound_friction_static)
    print("Best Friction Dynamic: ", bound_friction_dynamic)
    print("Best Armature: ", bound_armature)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
