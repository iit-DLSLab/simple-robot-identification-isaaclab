# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi

import rclpy 
from rclpy.node import Node 
from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import copy
import os 
import torch

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

import mujoco
import mujoco.viewer
import config


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_MUJOCO_RENDER = True
USE_MUJOCO_SIMULATION = True


CONTROL_FREQ = config.frequency_collection # Hz 


class Data_Collection_Node(Node):
    def __init__(self):
        super().__init__('Data_Collection_Node')
        # Subscribers and Publishers
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state", self.get_blind_state_callback, 1)
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1)
        self.timer = self.create_timer(1.0/CONTROL_FREQ, self.compute_control)


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_joints_arrived = False 

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None
        self.start_collection_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.feet_contact = np.zeros(4)


        # Create the environment -----------------------------------------------------------
        self.mjModel = mujoco.MjModel.from_xml_path(dir_path + "/robot_model/" + config.robot + "/scene_flat.xml")
        self.mjData = mujoco.MjData(self.mjModel)

        if(USE_MUJOCO_RENDER):
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.last_render_time = time.time()
        

        self.stand_up_and_down_actions = np.zeros(12)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions = goDown_qpos[7:29]
        self.stand_up_and_down_actions[2] += 0.3
        self.stand_up_and_down_actions[5] += 0.3
        self.stand_up_and_down_actions[8] += 0.3
        self.stand_up_and_down_actions[11] += 0.3
        
        
        self.Kp_stand_up_and_down = config.Kp
        self.Kd_stand_up_and_down = config.Kd

        self.calibration_reference_joint_positions = None
        

        # Chirp Trajectory only variables
        self.chirp_traj_time = 3.0
        self.calibration_reference_calf_trajectory = None
        self.calibration_reference_thigh_trajectory = None
        self.calibration_reference_hip_trajectory = None
        self.hip_setpoint2 = 0.6
        self.thigh_setpoint2 = 0.5
        self.calf_setpoin2 = -1.2
        self.hip_setpoint1 = self.stand_up_and_down_actions[0]
        self.thigh_setpoint1 = self.stand_up_and_down_actions[1]
        self.calf_setpoint1 = self.stand_up_and_down_actions[2]
        
        
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.num_traj_saved = 0


        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def get_blind_state_callback(self, msg):
        
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)

        self.first_message_joints_arrived = True

        
    def _initialize_calibration_setpoint(self):
        """Initialize calibration setpoint with random values"""
        print("Generating first a setpoint..")
        hip_setpoint = np.random.uniform(-0.0, 0.5)
        thigh_setpoint = np.random.uniform(-1.0, 0.5)
        calf_setpoint = np.random.uniform(-0., 1.5)
        self.calibration_reference_joint_positions = np.zeros((4, 3))
        self.calibration_reference_joint_positions[0] = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[1] = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[2] = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[3] = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.start_collection_time = time.time()

    def _initialize_calibration_trajectory(self):
        """Initialize calibration trajectory with random values"""
        print("Generating first a trajectory..")

        # Generate a linear trajectory between actual joint positions and two setpoint

        t = np.linspace(0, self.chirp_traj_time, num=100)

        self.calibration_reference_hip_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.hip_setpoint1, self.hip_setpoint2, self.hip_setpoint1]
        )

        self.calibration_reference_thigh_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.thigh_setpoint1, self.thigh_setpoint2, self.thigh_setpoint1]
        )

        self.calibration_reference_calf_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.calf_setpoint1, self.calf_setpoin2, self.calf_setpoint1]
        )

        self.start_collection_time = time.time()


    def _get_desired_positions_and_gains(self, ):
        """Get desired joint positions and control gains based on collection type"""
        if self.console.setpoint_collection:
            # Setpoint collection: maintain target position
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0] = copy.deepcopy(self.calibration_reference_joint_positions[0])
            desired_joint_pos[1] = copy.deepcopy(self.calibration_reference_joint_positions[1])
            desired_joint_pos[2] = copy.deepcopy(self.calibration_reference_joint_positions[2])
            desired_joint_pos[3] = copy.deepcopy(self.calibration_reference_joint_positions[3])
            Kp = self.Kp_stand_up_and_down
            Kd = self.Kd_stand_up_and_down
            
        elif self.console.falling_collection:
            # Falling collection: target first, then free fall
            random_coin = np.random.randint(0, 4)
            if(random_coin == 0):
                random_time = 0.0 #No waiting time, immediate fall
            else:
                random_time = 1.8
            if time.time() - self.start_collection_time > (2.0-random_time):
                # Free falling!!
                Kp = 0.0
                Kd = 0.0
                desired_joint_pos = np.zeros((4, 3))
                desired_joint_pos[0] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[1] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[2] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[3] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
            else:
                # Reach the target
                Kp = self.Kp_stand_up_and_down
                Kd = self.Kd_stand_up_and_down
                desired_joint_pos = np.zeros((4, 3))
                desired_joint_pos[0] = copy.deepcopy(self.calibration_reference_joint_positions[0])
                desired_joint_pos[1] = copy.deepcopy(self.calibration_reference_joint_positions[1])
                desired_joint_pos[2] = copy.deepcopy(self.calibration_reference_joint_positions[2])
                desired_joint_pos[3] = copy.deepcopy(self.calibration_reference_joint_positions[3])
        
        elif self.console.trajectory_collection:
            # Trajectory collection: follow the reference trajectory
            Kp = self.Kp_stand_up_and_down
            Kd = self.Kd_stand_up_and_down

            time_traj = self.start_collection_time - time.time()
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0, 0] = self.calibration_reference_hip_trajectory[int((time_traj/self.chirp_traj_time)*100)]
            desired_joint_pos[0, 1] = self.calibration_reference_thigh_trajectory[int((time_traj/self.chirp_traj_time)*100)]
            desired_joint_pos[0, 2] = self.calibration_reference_calf_trajectory[int((time_traj/self.chirp_traj_time)*100)]
            desired_joint_pos[1] = copy.deepcopy(desired_joint_pos[0])
            desired_joint_pos[1, 0] = -desired_joint_pos[1, 0]
            desired_joint_pos[2] = copy.deepcopy(desired_joint_pos[0])
            desired_joint_pos[3] = copy.deepcopy(desired_joint_pos[0])
            desired_joint_pos[3, 0] = -desired_joint_pos[3, 0]  
        return desired_joint_pos, Kp, Kd

    def _collect_trajectory_data(self, joints_pos, joints_vel, desired_joint_pos):
        """Collect trajectory data by concatenating and storing joint information"""
        concatenated_actual_joints_position = np.concatenate([joints_pos[0], joints_pos[1],
                                                            joints_pos[2], joints_pos[3]])
        concatenated_actual_joints_velocity = np.concatenate([joints_vel[0], joints_vel[1],
                                                            joints_vel[2], joints_vel[3]])
        concatenated_desired_joints_position = np.concatenate([desired_joint_pos[0], desired_joint_pos[1],
                                                            desired_joint_pos[2], desired_joint_pos[3]])
        concatenated_desired_joints_velocity = np.concatenate([joints_vel[0]*0.0, joints_vel[1]*0.0,
                                                            joints_vel[2]*0.0, joints_vel[3]*0.0])

        if self.saved_actual_joints_position is None:
            self.saved_actual_joints_position = concatenated_actual_joints_position
            self.saved_actual_joints_velocity = concatenated_actual_joints_velocity
            self.saved_desired_joints_position = concatenated_desired_joints_position
            self.saved_desired_joints_velocity = concatenated_desired_joints_velocity
        else:
            self.saved_actual_joints_position = np.vstack([self.saved_actual_joints_position, concatenated_actual_joints_position])
            self.saved_actual_joints_velocity = np.vstack([self.saved_actual_joints_velocity, concatenated_actual_joints_velocity])
            self.saved_desired_joints_position = np.vstack([self.saved_desired_joints_position, concatenated_desired_joints_position])
            self.saved_desired_joints_velocity = np.vstack([self.saved_desired_joints_velocity, concatenated_desired_joints_velocity])

    def _check_collection_complete(self, joints_pos, desired_joint_pos):
        """Check if data collection is complete based on collection type"""
        if self.console.setpoint_collection:
            # Complete when target is reached or timeout
            target_reached = (np.linalg.norm(desired_joint_pos[0] - joints_pos[0]) < 0.1 and
                            np.linalg.norm(desired_joint_pos[1] - joints_pos[1]) < 0.1 and
                            np.linalg.norm(desired_joint_pos[2] - joints_pos[2]) < 0.1 and
                            np.linalg.norm(desired_joint_pos[3] - joints_pos[3]) < 0.1)
            timeout = time.time() - self.start_collection_time > 2.0
            return target_reached or timeout
            
        elif self.console.falling_collection:
            # Complete after falling phase timeout
            return time.time() - self.start_collection_time > 2.5

        elif self.console.trajectory_collection:
            # Complete after trajectory duration
            return time.time() - self.start_collection_time > self.chirp_traj_time

    def _save_trajectory_data(self):
        """Save collected trajectory data to file"""

        # HACK
        num_steps = self.saved_actual_joints_position.shape[0]
        duration = num_steps/CONTROL_FREQ
        time_data = torch.linspace(0, duration, steps=num_steps, device="cpu")
        dof_pos_buffer = torch.zeros(num_steps, 12, device="cpu")
        dof_target_pos_buffer = torch.zeros(num_steps, 12, device="cpu")

        dof_pos_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_position)
        dof_target_pos_buffer[:] = torch.from_numpy(self.saved_desired_joints_position)

        torch.save({
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
        }, "datasets/" + config.robot + f"/traj_{self.num_traj_saved}.pt")

        self.num_traj_saved += 1
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None

        input("Press enter to continue.")


    def compute_control(self):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time

        # Safety check to not do anything until a first base and blind state are received
        if(not USE_MUJOCO_SIMULATION and self.first_message_joints_arrived==False):
            return


        # Update the mujoco model
        if(not USE_MUJOCO_SIMULATION):
            self.mjData.qpos[0:3] = copy.deepcopy(self.position)
            self.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
            self.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
            self.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
            self.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
            self.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
            self.mjModel.opt.timestep = simulation_dt
            mujoco.mj_forward(self.mjModel, self.mjData)  
        else:
            self.mjData.qpos[0:3] = np.array([0, 0, 0.4])
            self.mjData.qpos[3:7] = np.array([1, 0, 0, 0])
            self.mjData.qvel[0:3] = np.array([0, 0, 0.])
            self.mjData.qvel[3:6] = np.array([0, 0, 0.0])


        qpos, qvel = self.mjData.qpos, self.mjData.qvel


        joints_pos = np.zeros((4, 3))
        joints_pos[0] = qpos[7:10]
        joints_pos[1] = qpos[10:13]
        joints_pos[2] = qpos[13:16]
        joints_pos[3] = qpos[16:19]

        joints_vel = np.zeros((4, 3))
        joints_vel[0] = qvel[6:9]
        joints_vel[1] = qvel[9:12]
        joints_vel[2] = qvel[12:15]
        joints_vel[3] = qvel[15:18]
    
        if(not self.console.isActivated):
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0] = self.stand_up_and_down_actions[0:3]
            desired_joint_pos[1] = self.stand_up_and_down_actions[3:6]
            desired_joint_pos[2] = self.stand_up_and_down_actions[6:9]
            desired_joint_pos[3] = self.stand_up_and_down_actions[9:12]

            # Impedence Loop
            Kp = self.Kp_stand_up_and_down
            Kd = self.Kd_stand_up_and_down
            

        elif(self.console.isActivated and (self.console.setpoint_collection or self.console.falling_collection)):
            
            # Initialize setpoint if needed
            if self.calibration_reference_joint_positions is None:
                self._initialize_calibration_setpoint()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains()
            
            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)
            
            # Check if collection is complete
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)
            if collection_complete:
                self.calibration_reference_joint_positions = None
                self._save_trajectory_data()

        elif(self.console.isActivated and self.console.trajectory_collection):
            # Initialize setpoint if needed
            if self.calibration_reference_hip_trajectory is None:
                self._initialize_calibration_trajectory()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains()

            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)

            # Check if collection is complete            
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)
            if collection_complete:
                self.calibration_reference_hip_trajectory = None
                self.calibration_reference_thigh_trajectory = None
                self.calibration_reference_calf_trajectory = None
                self.chirp_traj_time -= 0.2 # Reduce trajectory time for next trajectory
                if(self.chirp_traj_time < 0.4):
                    self._save_trajectory_data()
                    self.console.trajectory_collection = False
                    print("Trajectory collection completed.")
        else:
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0] = self.stand_up_and_down_actions[0:3]
            desired_joint_pos[1] = self.stand_up_and_down_actions[3:6]
            desired_joint_pos[2] = self.stand_up_and_down_actions[6:9]
            desired_joint_pos[3] = self.stand_up_and_down_actions[9:12]

            # Impedence Loop
            Kp = self.Kp_stand_up_and_down*0.0
            Kd = self.Kd_stand_up_and_down*0.0
            
        
        if USE_MUJOCO_SIMULATION:
            for j in range(10): #Hardcoded for now, if RL is 50Hz, this runs the simulation at 500Hz
                qpos, qvel = self.mjData.qpos, self.mjData.qvel
                joints_pos[0] = qpos[7:10]
                joints_pos[1] = qpos[10:13]
                joints_pos[2] = qpos[13:16]
                joints_pos[3] = qpos[16:19]

                joints_vel[0] = qvel[6:9]
                joints_vel[1] = qvel[9:12]
                joints_vel[2] = qvel[12:15]
                joints_vel[3] = qvel[15:18]

                error_joints_pos = np.zeros((4, 3))
                error_joints_pos[0] = desired_joint_pos[0] - joints_pos[0]
                error_joints_pos[1] = desired_joint_pos[1] - joints_pos[1]
                error_joints_pos[2] = desired_joint_pos[2] - joints_pos[2]
                error_joints_pos[3] = desired_joint_pos[3] - joints_pos[3]
                
                tau = np.zeros((4, 3))
                tau[0] = Kp * (error_joints_pos[0]) - Kd * joints_vel[0]
                tau[1] = Kp * (error_joints_pos[1]) - Kd * joints_vel[1]
                tau[2] = Kp * (error_joints_pos[2]) - Kd * joints_vel[2]
                tau[3] = Kp * (error_joints_pos[3]) - Kd * joints_vel[3]
                
                action = np.zeros(self.mjModel.nu)
                action = tau.flatten()
                self.mjData.ctrl = action
                mujoco.mj_step(self.mjModel, self.mjData)


        # Publish the desired joint positions to the trajectory generator --------------------------------
        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.joints_position = np.array([desired_joint_pos[0], desired_joint_pos[1], desired_joint_pos[2], desired_joint_pos[3]]).flatten().tolist()
        trajectory_generator_msg.joints_velocity = np.zeros(12).tolist()
        trajectory_generator_msg.kp = (np.ones(12) * Kp).tolist()
        trajectory_generator_msg.kd = (np.ones(12) * Kd).tolist()
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)
        
        
        
        # Render the simulation -----------------------------------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30
            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
                self.viewer.sync()




#---------------------------
if __name__ == '__main__':
    print('Hello from your lovely data_collection routine.')
    rclpy.init()
    data_collection_node = Data_Collection_Node()

    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()

    print("Data-Collection-Node is stopped")
    exit(0)