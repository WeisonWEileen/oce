import mujoco
import mujoco.viewer 
import numpy as np
import time


m = mujoco.MjModel.from_xml_path("pemdulem.xml")
m.opt.timestep = 0.01
d = mujoco.MjData(m)
# with mujoco.viewer.launch_passive(m, d) as viewer:
# viewer = mujoco.viewer.launch_passive(m, d)
# while True:
# start = time.time()
step_count = 0
# render_interval = 100
# for i in range(13000):
# print(d.qpos)
d.qpos[0] = - np.pi /2 + 0.02
# print(d.qpos)
step_interval = 10
# print(d.qpos)
# exit()

with mujoco.viewer.launch(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        print(d.qpos)
        # if step_count % step_interval == 0:
        mujoco.mj_step(m, d)
        viewer.sync()
        # step_count += 1

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
