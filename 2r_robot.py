import mujoco
import mujoco.viewer 
import numpy as np
import time

m = mujoco.MjModel.from_xml_path("pemdulem.xml")
m.opt.timestep = 0.01
d = mujoco.MjData(m)
step_count = 0
d.qpos[0] = - np.pi /2 + 0.02
step_interval = 10

with mujoco.viewer.launch(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
