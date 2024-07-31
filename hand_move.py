import pickle
import time
import numpy as np

from scipy.spatial.transform import Rotation as R
from FR3Py.robot.interface import FR3Real
from FR3Py.robot.model import PinocchioModel
import time

from actuator import Actuator, language_controller
import numpy as np

robot = FR3Real(robot_id='fr3')
model = PinocchioModel()

time.sleep(0.2)

def getHandPose(robot):
        state = robot.getStates()

        if state is not None:
            q, dq = state['q'], state['dq']
            print(q)
            info = model.getInfo(np.hstack([q, np.zeros(2)]), np.hstack([dq, np.zeros(2)]))
            print(info)
            R, t = info['R_HAND'], info['P_HAND']

            base_T_hand = np.vstack([np.hstack([R, t.reshape(3,1)]),
                                    np.array([0,0,0,1])])
            return base_T_hand
        else:
            return None

with open('hand_pose.pkl', 'rb') as file: 
    
    camera1_T_Pose = pickle.load(file) 

print(camera1_T_Pose)
# r_base_T_Pose= base_T_Pose[:3,:3]
# # # t_base_T_Pose = base_T_Pose[0:3,3]

# hand_T_camera = np.array([[-0.   , -1.   , -0.011,  0.052],
#                         [ 1.   , -0.   , -0.005, -0.032],
#                         [ 0.005, -0.011,  1.   , -0.001],
#                         [ 0.   ,  0.   ,  0.   ,  1.   ]])

hand_T_camera = np.array([[-0.003, -1.   , -0.014,  0.057],
                        [ 1.   , -0.003,  0.001, -0.043],
                        [-0.001, -0.014,  1.   , -0.005],
                        [ 0.   ,  0.   ,  0.   ,  1.   ]])
# Pose_T_aligned = np.array([[-0.00696832 ,-0.21676174 , 0.97616185 , 0.07412773],
#                         [-0.1174239,  -0.96887506, -0.21612347, -0.00396809],
#                         [ 0.99267646 ,-0.11614281, -0.01851179,  0.00957253],
#                         [ 0. ,         0.   ,       0.    ,      1.        ]])

# Pose_T_aligned = np.array([[-0.00696832 ,-0.21676174 , 0.97616185 , 0.05012773],
#                         [-0.1174239,  -0.96887506, -0.21612347, -0.00396809],
#                         [ 0.99267646 ,-0.11614281, -0.01851179,  0.00457253],
#                         [ 0. ,         0.   ,       0.    ,      1.        ]])

# Pose_T_aligned = np.array([[ 0.78270388 , 0.43622219 , 0.44381137 , 0.07560212],
#                             [-0.1824615 ,  0.84264039 ,-0.50649464 ,-0.06304643],
#                             [-0.59502619 , 0.31549961 , 0.73915931 ,-0.04881841],
#                             [ 0.    ,      0.   ,       0.  ,        1.        ]])

# Pose_T_aligned = np.array([[ 0.21879083, -0.02051317 , 0.9754621,   0.02808009],
#                             [ 0.97563615 ,-0.01155858, -0.21900942 ,-0.00558704],
#                             [ 0.01571814 , 0.99962713 , 0.01747833 , 0.00320354],
#                             [ 0.    ,      0.  ,        0.         , 1.        ]])

Pose_T_aligned = np.array([[ 1, 0 , 0,    0.02808009],
                            [ 0 ,1, 0 ,-0.00558704],
                            [ 0 , 0, 1, 0.00320354],
                            [ 0,0.,0, 1]])

base_T_hand = getHandPose(robot)

print(base_T_hand)




# rotation_90_deg_y = R.from_euler('y', -90, degrees=True).as_matrix()

# rotY_grasp = np.dot(rotation_90_deg_y, camera1_T_Pose[:3,:3])

# camera1_T_Pose[:3,:3] = rotY_grasp

base_T_aligned = base_T_hand @ hand_T_camera @ camera1_T_Pose @ Pose_T_aligned 
print("base_T_aligned computed by post multiplying= ", base_T_aligned)

# quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
# print("Quaternion of t_base_T_aligned= " , t_base_T_aligned)
# print("translation t_base_T_aligned= " , t_base_T_aligned)


r_base_T_aligned = base_T_aligned[:3,:3]
t_base_T_aligned = base_T_aligned[0:3,3]


rot_angle = R.from_matrix(r_base_T_aligned)
angle_rad = rot_angle.magnitude()
angle_deg = np.degrees(angle_rad)

print(angle_deg)
rotateY = np.zeros((4,4))
r = R.from_euler('y', (angle_deg/2) , degrees=True)


rotateY[:3,:3]  = r.as_matrix()
rotateY[0:3,3] = [0,0,0]
rotateY[3,3] = 1
print("base_T_aligned_rotate Before rotation = ", rotateY)

base_T_aligned_rotated = base_T_aligned @  rotateY

print("base_T_aligned_rotated after rotation along Y by ang deg= " , base_T_aligned_rotated)


r_base_T_aligned = base_T_aligned_rotated[:3,:3]
t_base_T_aligned = base_T_aligned_rotated[0:3,3]



# ### Define Experiments

i = 0
Rtarget = []


print(t_base_T_aligned)
print(r_base_T_aligned)


# breakpoint()

actuator = Actuator()


# ### Robot Actions
'''
APPEND THE BASE_T_ALIGN ROT MATRICES TO THE Rtarget which contains Pick up and Place locations
'''


#Pick Up
Rtarget.append(r_base_T_aligned)    
Rtarget.append([[ 0.99723639 ,-0.06815166 ,-0.02957922],
                [-0.06779145 ,-0.99761461 , 0.01301558],
                [-0.03039569 ,-0.01097439 ,-0.9994777 ]])
# Rtarget.append([[1.0 , 0.0, 0.0],
#                 [0.0, -1.0, 0.0],
#                 [0.0, 0.0, -1.0]])
# Rtarget.append(np.array(r_base_T_aligned))
#Place location (Delivery)

# breakpoint()

actions = language_controller(command='grasp', position=list(t_base_T_aligned),Rtarget=Rtarget[0])
# actions += language_controller(command='place', position=[0.27, 0.0, 0.12284176],Rtarget=Rtarget[1])
actions += language_controller(command='place', position=[0.348, 0.041, 0.227] ,Rtarget=Rtarget[1])
# actions += language_controller(command='return', position= [0.3, 0.0, 0.5],Rtarget=Rtarget[2])
# print("i= ",i )
print(actions)
# breakpoint()
for action in actions:
    actuator(Rtarget=action[0],ptarget=action[1],
            duration=action[2],
            gripper_command=action[3])
    time.sleep(1.5)

actuator.close()
robot.close()
print("DONE")
