from typing import List
import time
from scipy.spatial.transform import Rotation as R_scipy
import numpy as np
import pinocchio as pin
import proxsuite

import sys
sys.path.append("/home/rstaion/Documents/FR3Py")

from FR3Py.robot.interface import FR3Real
from fr3_gripper import Gripper
from FR3Py.robot.model import PinocchioModel


class GripperCommand:
    OPEN = 'open'
    CLOSE = 'close'
    NO_CHANGE = 'no change'


class DiffIK:
    def __init__(self):
        self.epsilon = 0.01
        self.kp_joint_centering = 0.01

        self.n = 9
        self.n_eq = 0
        self.n_ieq = 0
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.qp_initialized = False

        self.q_nominal = np.array(
            [
                0.0,
                -0.785398163,
                0.0,
                -2.35619449,
                0.0,
                1.57079632679,
                0.785398163397,
                0.0,
                0.0,
            ]
        )[:, np.newaxis]

    def __call__(self, V_des, q, J):
        self.H, self.g = self.compute_params(V_des, q, J)

        if not self.qp_initialized:
            self.qp.init(H=self.H, g=self.g)
            self.qp.settings.eps_abs = 1.0e-6
            self.qp.settings.max_iter = 20
            self.qp_initialized = True
        else:
            self.qp.update(H=self.H, g=self.g)

        self.qp.solve()

        return self.qp.results.x

    def compute_params(self, V_des, q, J):
        nullspace_mat = np.eye(self.n) - np.linalg.pinv(J) @ J
        nullspace_quadratic = nullspace_mat.T @ nullspace_mat

        H = 2 * (J.T @ J + self.epsilon * nullspace_quadratic)

        dq_nominal = self.kp_joint_centering * (self.q_nominal - q[:, np.newaxis])
        g = -2 * (V_des.T @ J + self.epsilon * dq_nominal.T @ nullspace_quadratic).T

        return H, g
    
class Actuator:
    def __init__(self) -> None:
        self.controller = DiffIK()
        self.robot = FR3Real(robot_id='fr3')
        # Be sure there communication is established with the arm
        state = self.robot.getJointStates()
        print(state)
        assert state is not None

        # self.gripper = Gripper("10.42.0.4")
        self.gripper = Gripper("192.168.123.250")
        assert self.gripper.homing()
        self.robot_model = PinocchioModel()
        info = self.robot_model.getInfo(state['q'], state['dq'])
        # info = self.robot_model.getInfo(np.hstack([state['q'], np.zeros(2)]), np.hstack([state['dq'], np.zeros(2)]))
        self.Rinit = info['R_EE']
        
    def __call__(self, Rtarget, ptarget: List, duration: float, gripper_command: str)->None:
        state = self.robot.getJointStates()
        info = self.robot_model.getInfo(state['q'], state['dq'])
        # info = self.robot_model.getInfo(np.hstack([state['q'], np.zeros(2)]), np.hstack([state['dq'], np.zeros(2)]))
        R0 = info['R_EE']
        P0 = info['P_EE']
        T0 = pin.SE3(R0, P0)
        # Ttarget = pin.SE3(self.Rinit, np.array(ptarget))
  
        Ttarget = pin.SE3(np.array(Rtarget), np.array(ptarget))
        start_time = time.time()
        while time.time()-start_time < duration:
            time.sleep(0.01)
            state = self.robot.getJointStates()
            assert state is not None
            info = self.robot_model.getInfo(state['q'], state['dq'])
            # info = self.robot_model.getInfo(np.hstack([state['q'], np.zeros(2)]), np.hstack([state['dq'], np.zeros(2)]))
            # print(info["q"][0:7])
            info["J_EE"] = np.hstack([info["J_EE"] , np.zeros((6,2))])

            # breakpoint()

            R = info['R_EE']
            P = info['P_EE']
            T = pin.SE3(R, P)
            t = time.time() - start_time
            T_temp = pin.SE3.Interpolate(T0, Ttarget,t/duration)
            v_error = pin.log6(T_temp @ T.inverse().homogeneous).vector
            v_des = 2.0 * v_error
            # print(v_des)
            # breakpoint()
            Δdq = self.controller(v_des, info["q"], info["J_EE"])
            # print(Δdq[0:7])
            # breakpoint()
            self.robot.setCommands(Δdq[0:7])
        self.robot.setCommands(np.zeros(7))

        if gripper_command == GripperCommand.OPEN:
            time.sleep(0.5)
            self.gripper.move(0.08, 0.05)
            time.sleep(0.5)
        elif gripper_command == GripperCommand.CLOSE:
            time.sleep(0.5)
            self.gripper.grasp(0.0085, 0.05, 25, 0.01, 0.01)
            time.sleep(0.5)
    def close(self):
        self.robot.close()
class CapturePosActuator:
    def __init__(self) -> None:
        self.controller = DiffIK()
        self.robot = FR3Real()
        # Be sure there communication is established with the arm
        state = self.robot.getJointStates()
        assert state is not None

        self.gripper = Gripper("10.42.0.4")
        assert self.gripper.homing()
        self.robot_model = PinocchioModel()
        info = self.robot_model.getInfo(state['q'], state['dq'])
        self.Rinit = info['R_EE']
        
    def __call__(self, ptarget: List, duration: float)->None:
        state = self.robot.getStates()
        info = self.robot_model.getInfo(state['q'], state['dq'])
        R0 = info['R_EE']
        P0 = info['P_EE']
        T0 = pin.SE3(R0, P0)

  
        Ttarget = pin.SE3(R0, np.array(ptarget))
        start_time = time.time()
        while time.time()-start_time < duration:
            time.sleep(0.01)
            state = self.robot.getStates()
            assert state is not None
            info = self.robot_model.getInfo(state['q'], state['dq'])
            R = info['R_EE']
            P = info['P_EE']
            T = pin.SE3(R, P)
            t = time.time() - start_time
            T_temp = pin.SE3.Interpolate(T0, Ttarget,t/duration)
            v_error = pin.log6(T_temp @ T.inverse().homogeneous).vector
            v_des = 2.0 * v_error
            Δdq = self.controller(v_des, info["q"], info["J_EE"])
            self.robot.sendCommands(Δdq)
        self.robot.sendCommands(np.zeros(9))

    def run_pose(self,Rtarget, ptarget: List,duration: float):
        state = self.robot.getStates()
        info = self.robot_model.getInfo(state['q'], state['dq'])

        t_base_T_hand = np.array([[0.117, -0.332, 0.654]])
        quat_base_T_hand = np.array([[0.883, -0.437, 0.136, 0.106]])
        r_base_T_hand = R_scipy.from_quat(quat_base_T_hand).as_matrix()
        base_T_hand = np.zeros((4,4))
        base_T_hand[:3,:3] = r_base_T_hand
        base_T_hand[0:3,3] = t_base_T_hand  
        base_T_hand[3,3] = 1

        hand_T_camera = np.array([[-0.028,  1.   , -0.001,  0.05 ],
                                [-1.   , -0.028, -0.005, -0.008],
                                [-0.005,  0.001,  1.   ,  0.006],
                                [ 0.   ,  0.   ,  0.   ,  1.   ]])

        base_T_camera = base_T_hand @ hand_T_camera

        R0 = info['R_EE']
        P0 = info['P_EE']
        T0 = np.eye(4)
        T0[:3,:3] = R0
        T0[:3,3] = np.array(P0)
        T01 = pin.SE3(R0, np.array(P0))
        #T12 = pin.SE3(Rtarget,np.array(ptarget))
        T1 = np.eye(4)
        T1[:3,:3] = Rtarget
        T1[:3,3] = np.array(ptarget)
        #T02 = np.matmul(T01,T12)
        T2 = T0 @ hand_T_camera @ T1
        T2[2][3] = 0.005
        print("Final Transformation Matrix obtained")
        print(T2)
        input()
        Ttarget = pin.SE3(R0,T2[:3,3])
        start_time = time.time()
        while time.time()-start_time < duration:
            time.sleep(0.01)
            state = self.robot.getStates()
            assert state is not None
            info = self.robot_model.getInfo(state['q'], state['dq'])
            R = info['R_EE']
            P = info['P_EE']
            T = pin.SE3(R, P)
            t = time.time() - start_time
            T_temp = pin.SE3.Interpolate(T01, Ttarget,t/duration)
            v_error = pin.log6(T_temp @ T.inverse().homogeneous).vector
            v_des = 2.0 * v_error
            Δdq = self.controller(v_des, info["q"], info["J_EE"])
            self.robot.sendCommands(Δdq)
        self.robot.sendCommands(np.zeros(9))



def language_controller(command:str, position:list, Rtarget:list):
    if command == 'grasp':
        # output = [(position[0:2]+ [position[2] + 0.1] , 5.0, GripperCommand.NO_CHANGE), #PRE-GRASP
        #           (position[0:2]+ [position[2] + 0.1] , 0.5, GripperCommand.NO_CHANGE), #STABILIZE PRE-GRASP
        #            (position[0:2]+ [position[2] + 0.0050] , 2.0, GripperCommand.NO_CHANGE), # GRASP
        #            (position[0:2]+ [position[2] + 0.0050] , 1.5, GripperCommand.CLOSE), #STABILIZE GRASP
        #            (position[0:2]+ [position[2] + 0.1] , 2.0, GripperCommand.NO_CHANGE), #POST GRASP
                #    (position[0:2]+ [position[2] + 0.1] , 0.5, GripperCommand.NO_CHANGE) #STABILIZE POST GRASP
                #    ] 
        output = [(Rtarget, position[0:2]+ [position[2] + 0.1] , 10.0, GripperCommand.NO_CHANGE), #PRE-GRASP
            (Rtarget, position[0:2]+ [position[2] + 0.1] , 1.0, GripperCommand.NO_CHANGE), #STABILIZE PRE-GRASP
            (Rtarget, position[0:2]+ [position[2]+ 0.0050 ] , 5.0, GripperCommand.NO_CHANGE), # GRASP
            (Rtarget, position[0:2]+ [position[2] + 0.0050] , 1.5, GripperCommand.CLOSE), #STABILIZE GRASP
            (Rtarget, position[0:2]+ [position[2] + 0.1] , 2.0, GripperCommand.NO_CHANGE), #POST GRASP
        ]
    elif command == 'place':
        output = [(Rtarget, position[0:2] + [position[2] + 0.1] , 8.0, GripperCommand.NO_CHANGE), # PRE-PLACE
                  (Rtarget, position[0:2] + [position[2] + 0.1] , 0.5, GripperCommand.NO_CHANGE), # PRE-PLACE STABILIZE
                   (Rtarget, position[0:2] + [position[2] + 0.0050] , 2.0, GripperCommand.NO_CHANGE), # PLACE
                   (Rtarget, position[0:2] + [position[2] + 0.0050] , 1.0, GripperCommand.OPEN), # PLACE STABILIZE
                   (Rtarget, position[0:2] + [position[2] + 0.1], 2.0, GripperCommand.NO_CHANGE), # POST place
                #    (position[0:2] + [position[2] + 0.1], 0.5, GripperCommand.NO_CHANGE), # POST place STABILIZE
                   ]
    elif command == 'return':
        output = [(Rtarget, position, 5.0, GripperCommand.NO_CHANGE),
                  (Rtarget, position, 3.0, GripperCommand.NO_CHANGE)]
    return output
