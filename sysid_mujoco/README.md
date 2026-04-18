# MuJoCo SysID

This directory contains a MuJoCo-based system identification pipeline
The workflow is:`

* Fit per-joint `damping`, `armature`, and `frictionloss` on the recorded trajectory, from the repo main folder:

```bash
python sysid_mujoco/fit.py --dataset <path/to/the/dataset> --robot"name of the robot"
```

Notes:

-  fit.py creates a model of the robot with fixed base, and used mujoco integrated PD controller, it will save the new values in the result/<robot name> as an html report, where you can check the tracking and the parameters values
