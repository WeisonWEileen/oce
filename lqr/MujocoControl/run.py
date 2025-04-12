import mujoco
import numpy as np
import time
import os
os.environ["QT_MAC_WANTS_LAYER"] = "1"
import mujoco.viewer as viewer
from scipy.linalg import solve_discrete_are

class LQR_Perdulum():
  def __init__(self, xml_model_path):
    self.model = mujoco.MjModel.from_xml_path(xml_model_path)
    self.data = mujoco.MjData(self.model)
    self.input_limit = 1000
    self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Cart")
    self.pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pole")

  def compute_K(self):
    m_c = self.model.body_mass[self.cid]
    m_p = self.model.body_mass[self.pid]
    l_pos = self.model.body_pos[self.pid]
    length = np.abs(l_pos[2])
    g = 9.81
    print(f"Cart mass: {m_c}")
    print(f"Pole mass: {m_p}")
    print(f"Length of pendulem {length}")

    # 
    A_con = np.array([
      [0,                        0,      1,      0],
      [0,                        0,      0,     -1],
      [0,               -m_p*g/m_c,      0,      0],
      [0, -(m_c+m_p)*g/(length*m_c),     0,      0]]
    )

    B_con = np.array([
        [0],
        [0],
        [1/m_c],
        [1/(length*m_c)]
    ])

    dt = self.model.opt.timestep

    # distretize
    A = np.eye(4) + A_con * dt
    B = B_con * dt 

    Q = np.diag([1, 100, 10, 10])  # state weight
    R = np.array([[5]])        # input weight
    P = solve_discrete_are(A, B, Q, R) # Riccati solution
    self.K = np.linalg.inv(R) @ B.T @ P # feedback metrics
    print(f"timestep is {dt}")
    print("A is")
    print(A)
    print("B is")
    print(B)
    print("K is")
    print(self.K)

  def control(self):
      x_error = np.array([
          self.data.qpos[0],       
          np.pi - self.data.qpos[1],
          self.data.qvel[0],       
          self.data.qvel[1]        
      ])
      u = -self.K@x_error
      self.data.ctrl = u
      
      return u
      
  def runLoop(self):
    mujoco.mj_resetDataKeyframe(self.model, self.data, 0) 
    with viewer.launch_passive(self.model, self.data) as viewer_:  
      while True:
        step_start = time.time()
        # self.data.ctrl = np.clip(u, -self.input_limit, self.input_limit)
        self.control()
        mujoco.mj_step(self.model, self.data)
        print(self.data.qpos[1])
        # print(self.data.sensordata[1])
        viewer_.cam.lookat[:] = self.data.body(self.cid).xpos
        viewer_.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)

if __name__ == '__main__':
  modelxml_path = 'mjcf/carpole.xml'
  # mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
  lqr = LQR_Perdulum(modelxml_path)
  lqr.compute_K()
  lqr.runLoop()
  