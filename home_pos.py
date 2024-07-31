import pickle
import time
import numpy as np

from actuator import Actuator, language_controller

# robot = FR3Real(robot_id='fr3')
# model = PinocchioModel()

time.sleep(0.2)



# ### Define Experiments

i = 0
Rtarget = []


actuator = Actuator()


# ### Robot Actions
'''
APPEND THE BASE_T_ALIGN ROT MATRICES TO THE Rtarget which contains Pick up and Place locations
'''


#Pick Up
Rtarget.append([[ 0.9259283,  0.03434064 , 0.37613496],
                [ 0.01667528 ,-0.99860389 , 0.05012186],
                [ 0.37733105 ,-0.04013709 ,-0.92520824]])
# Rtarget.append(np.array(r_base_T_aligned))
#Place location (Delivery)

# breakpoint()

actions = language_controller(command='return', position= [0.020, 0.417, 0.523], Rtarget=Rtarget[0])
# print("i= ",i )
print(actions)

for action in actions:
    actuator(Rtarget=action[0],ptarget=action[1],
            duration=action[2],
            gripper_command=action[3])
    time.sleep(1.5)

actuator.close()
print("DONE")
