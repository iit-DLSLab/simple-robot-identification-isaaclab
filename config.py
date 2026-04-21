import numpy as np

robot = 'z1'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal2', 'z1' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])
                   
    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "go2"):
    # Order FL, FR, RL, RR
    Kp = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])
                   
    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "b2"):
    # Order FL, FR, RL, RR
    Kp = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])

    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "hyqreal2"):
    # Order FL, FR, RL, RR
    Kp = np.array([175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.])

    Kd = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])

elif(robot =="z1"):
    # Order joint1-joint2-joint3-joint4-joint5-joint6-gripper
    Kp = np.array([80., 120., 120.,
                   80., 80., 80.,
                   80.])

    Kd = np.array([2.5, 2.5, 2.5,
                   2.5, 2.5, 2.5,
                   2.5])

else:
    raise ValueError(f"Robot {robot} not supported")

frequency_collection = 200#hz