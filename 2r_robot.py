import mujoco
import mujoco.viewer 
import numpy as np
import time


m = mujoco.MjModel.from_xml_path("pemdulem.xml")
m.opt.timestep = 0.01
data = mujoco.MjData(m)
# with mujoco.viewer.launch_passive(m, data) as viewer:
viewer = mujoco.viewer.launch(m, data)
# while True:
# start = time.time()
step_count = 0
# render_interval = 100
# for i in range(13000):
while True:
    step_start = time.time()
    print(data.qpos)
    mujoco.mj_step(m, data)
    # if step_count % render_interval == 0:
    viewer.render()
    step_count += 1
