import mujoco
import numpy as np
import time
import os

import mujoco.viewer as viewer
modelxml_path = 'mjcf/carpole.xml'
model = mujoco.MjModel.from_xml_path(modelxml_path)
data = mujoco.MjData(model)

os.environ["QT_MAC_WANTS_LAYER"] = "1"



# define customized controller which returns the feedback control action
# you can implement your controller here 
def myControl(model, data):
    x = np.hstack((data.qpos,data.qvel))
    xref = np.array([0, np.pi, 0, 0])
    x_error = x-xref
    K = np.array([-0.5, 0.5, 0, 0])
    u = K@x_error
    data.ctrl = u
    return u
    

mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0

# 
# with viewer.launch_passive(model, data) as viewer:  
with viewer.launch_passive(model, data) as viewer:  
  # launch_passive means all the simulation should be done by the user 
  
  start = time.time()
  while viewer.is_running() and time.time() - start < 10:
    step_start = time.time()
    data.ctrl = 0.2*myControl(model,data)
    mujoco.mj_step(model, data)

  # let viewer show updated info
    viewer.sync()
    
  # #  make sure the while loop is called every sampling period 
    # computation inside the loop may take some nontrivial time. 
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)