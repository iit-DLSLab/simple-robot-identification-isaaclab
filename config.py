robot = 'z1'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal2', 'z1' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp = 25.
    Kd = 2.

elif(robot == "go2"):
    Kp = 20.
    Kd = 1.5

elif(robot == "b2"):
    Kp = 20.
    Kd = 1.5

elif(robot == "hyqreal2"):
    Kp = 175.
    Kd = 20.

elif(robot =="z1"):
    Kp = 30.
    Kd = 2.

else:
    raise ValueError(f"Robot {robot} not supported")

frequency_collection = 200#hz