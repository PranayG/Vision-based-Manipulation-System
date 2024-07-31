import pickle
import time
import numpy as np

from actuator import Actuator, language_controller

# robot = FR3Real(robot_id='fr3')
# model = PinocchioModel()

time.sleep(0.2)

with open('pose.pkl', 'rb') as file: 
    
    base_T_Pose = pickle.load(file) 

print(base_T_Pose)
# r_base_T_Pose= base_T_Pose[:3,:3]
# # # t_base_T_Pose = base_T_Pose[0:3,3]
# Pose_T_aligned = np.array([[ 0.12168533,  0.0056451 ,  0.99255269 , 0.03976486],
#                             [-0.00560859 , 0.99997177, -0.0049997 ,  0.01692713],
#                             [-0.99255292 ,-0.00495843 , 0.12171357 , 0.01268308],
#                             [ 0. ,         0. ,         0.,          1.        ]])


Pose_T_aligned = np.array([[ 0.12168533,  0.0056451 ,  0.99255269 , 0.04976486],
                            [-0.00560859 , 0.99997177, -0.0049997 ,  0.01692713],
                            [-0.99255292 ,-0.00495843 , 0.12171357 , 0.02268308],
                            [ 0. ,         0. ,         0.,          1.        ]])

base_T_aligned = base_T_Pose @ Pose_T_aligned

print("base_T_aligned computed by post multiplying= ", base_T_aligned)

r_base_T_aligned= base_T_aligned[:3,:3]
t_base_T_aligned = base_T_aligned[0:3,3]



# ### Define Experiments

i = 0
Rtarget = []


print(t_base_T_aligned)
print(r_base_T_aligned)


actuator = Actuator()


# ### Robot Actions
'''
APPEND THE BASE_T_ALIGN ROT MATRICES TO THE Rtarget which contains Pick up and Place locations
'''


#Pick Up
Rtarget.append(r_base_T_aligned)    
Rtarget.append(r_base_T_aligned)
Rtarget.append([[1.0 , 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]])
# Rtarget.append(np.array(r_base_T_aligned))
#Place location (Delivery)

# breakpoint()

actions = language_controller(command='grasp', position=list(t_base_T_aligned),Rtarget=Rtarget[0])
actions += language_controller(command='place', position=[0.27, 0.0, 0.12284176],Rtarget=Rtarget[1])
actions += language_controller(command='return', position= [0.3, 0.0, 0.5],Rtarget=Rtarget[2])
# print("i= ",i )
print(actions)
# breakpoint()
for action in actions:
    actuator(Rtarget=action[0],ptarget=action[1],
            duration=action[2],
            gripper_command=action[3])
    time.sleep(1.5)

actuator.close()
print("DONE")
