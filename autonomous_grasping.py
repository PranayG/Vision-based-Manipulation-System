
import sys
import time 
from actuator import Actuator, language_controller
from llm import BrainLLM, BodyLLM
from SimpleHandEye.interfaces.cameras import RealSenseCamera
from FR3Py.robot.interface import FR3Real
from FR3Py.robot.model import PinocchioModel

import cv2
import time
import numpy as np

import os
import argparse
import torch
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

robot = FR3Real()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


i = 0
Rtarget = []

sys.path.append('/usr/local/lib/python3.8/site-packages')

def showImage(color_frame, depth_frame, ir1_frame, ir2_frame):
    cv2.imshow('image', color_frame)
    cv2.waitKey(33)

camera = RealSenseCamera(callback_fn=showImage)

intrinsics_params = camera.getIntrinsics()
K = intrinsics_params['RGB']['K']
D = intrinsics_params['RGB']['D']

model = PinocchioModel()
time.sleep(1)

def getHandPose(robot):
  state = robot.getStates()

  if state is not None:
    q, dq = state['q'], state['dq']
    info = model.getInfo(q, dq)
    R, t = info['R_HAND'], info['P_HAND']

    base_T_hand = np.vstack([np.hstack([R, t.reshape(3,1)]),
                            np.array([0,0,0,1])])
    return base_T_hand
  else:
    return None

anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
object_len = 4
'''
CLICK AND SAVE RGB IMAGE AND DEPTH
'''
while(object_len > 0):

    # #Open camera 1 :
    camera1 = RealSenseCamera(camera_serial_no=str(318122303427), VGA=True)
    color1 , depth1 = camera1.saveAlignedDepth() 

    cv2.imwrite("grasp_detection/example_data/camera1RGB_test.png", color1)
    cv2.imwrite("grasp_detection/example_data/camera1Depth_test.png", depth1.astype(np.uint16))

    colors = np.array(cv2.imread('examples/test2/color_4body_test2.png'),dtype=np.float32) / 255.0
    depths = np.array(cv2.imread('examples/test2/depth_4body_test2.png'))

    '''
    CHOOSE THE BEST SCORES OF MULTIPLE OBJECTS IN THE SCENE AND STORE GRASP ROT AND TRANS ---> ANYGRASP
    '''
    # get camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0

    # set workspace
    xmin, xmax = -0.2,0.2
    ymin, ymax = -0.2,0.2
    zmin, zmax = 0.175, 0.7
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # remove outlier
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    # get prediction
    gg, cloud = anygrasp.get_grasp(points, colors, lims)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        # o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


    '''
    FROM THE CHOSEN GRASP POSES OF THE OBJECTS TRASNFORM ALL POSES TO BASE_T_ALIGN FROM TRANSFORMS.PY FILE
    IN A WHILE LOOP 
    '''
    #FORWARD KINEMATICS

    # rossubscriber or read through FR3Py from states 
    base_T_hand = getHandPose(robot)
    
    print(base_T_hand)

    r_base_T_hand = base_T_hand[:3,:3] 
    t_base_T_hand = base_T_hand[0:3,3] 


    #CAMERA CALIBRATION
    hand_T_camera = np.array([[-0.028,  1.   , -0.001,  0.05 ],
                                [-1.   , -0.028, -0.005, -0.008],
                                [-0.005,  0.001,  1.   ,  0.006],
                                [ 0.   ,  0.   ,  0.   ,  1.   ]])
    r_hand_T_camera = hand_T_camera[:3,:3]
    t_hand_T_camera = hand_T_camera[0:3,3]
    print("Translation of hand_T_camera =  " , t_hand_T_camera)
    quat_hand_T_camera = R.from_matrix(r_hand_T_camera).as_quat()
    print("Quaternion of hand_T_camera= " , quat_hand_T_camera)

    #CAMERA_T_'OBJECT'

    camera_T_object = np.zeros((4,4))
    t_camera_T_object = np.array([list(gg_pick[0].translation)])
    r_camera_T_object =np.array([list(gg_pick[0].rotation_matrix)])
    camera_T_object [:3,:3] = r_camera_T_object 
    camera_T_object [0:3,3] = t_camera_T_object 
    camera_T_object [3,3] = 1

    object_T_align = np.array([[-0.00696832 ,-0.21676174 , 0.97616185 , 0.07412773],
                            [-0.1174239,  -0.96887506, -0.21612347, -0.00396809],
                            [ 0.99267646 ,-0.11614281, -0.01851179,  0.00957253],
                            [ 0. ,         0.   ,       0.    ,      1.        ]])

    camera_T_align = camera_T_object @ object_T_align

    base_T_align = base_T_hand @ hand_T_camera @ camera_T_align

    r_base_T_align= base_T_align[:3,:3]
    t_base_T_align = base_T_align[0:3,3]

    # ### Define Experiments
    '''
    FEED THE TRANSLATION POSES TO THE 'cube_positions' TO THE NEXT CELL AND ALSO THE  'locations'
    FOR THE NUMBER OF OBJECTS IN THE SCENE
    '''


    experiments = {
                    'fr3_matrix_exp':
                    {
                        'task': 'Create a plus sign in the working place using the cubes.',
                        'env_information': 'There are 8 cubes in the working place: red, blue, black, green, orange, brown, yellow, gray.'\
                        +'Working place can be divided into 3 columns: A, B, C and 3 rows: 1, 2, 3. For example, location 1_A represent A th column and 1st row of the working place.',
                        'cube_positions': {'<red_cube>': t_base_T_align,
                                        },
                        'locations': {'<location_1_A>':[0.43,-0.3,0.0],
                                    }
                    },
                    }


    # ### Define object instances and choose the experiment from the experiments dictionary

    llm_model = 'gpt-4'
    llm_key = 'your_llm_key'
    brain_llm_temperature = 0
    brain_llm_init_prompt_path = '/home/crrl-franka/FR3Py/autonomous-grasping/prompts/brain_llm_master_prompt.txt'
    brain_llm_feedback_prompt_path = '/home/crrl-franka/FR3Py/autonomous-grasping/prompts/brain_llm_feedback_prompt.txt'
    body_llm_prompt_path = '/home/crrl-franka/FR3Py/autonomous-grasping/prompts/body_llm_prompt.txt'
    experiment = experiments['fr3_matrix_exp']
    brain_llm = BrainLLM(model=llm_model,
                        temperature=brain_llm_temperature,
                        llm_key=llm_key,
                        initial_prompt_path=brain_llm_init_prompt_path,
                        feedback_prompt_path=brain_llm_feedback_prompt_path)
    body_llm = BodyLLM(model=llm_model,
                    temperature=0.0,
                    llm_key=llm_key,
                    prompt_path=body_llm_prompt_path,
                    cube_positions=experiment['cube_positions'],
                    locations=experiment['locations'],
                    controller=language_controller)
    actuator = Actuator()



    # ### Generate Initial Plan


    object_list = list(experiment['cube_positions'].keys())
    env_information = experiment['env_information']
    task = experiment['task']
    feedback_step = 0
    plan = brain_llm(task=task,
                            object_list=object_list,
                            env_information=env_information,
                            feedback_mode=False)
    steps_to_be_done = plan.copy()
    print(plan)


    # ### Robot Actions
    '''
    APPEND THE BASE_T_ALIGN ROT MATRICES TO THE Rtarget which contains Pick up and Place locations
    '''


    #Pick Up
    Rtarget.append(np.array([[-0.50809752, -0.84545902,  0.16441121],
                        [-0.7249874 ,  0.52286648 , 0.44837278],
                        [-0.46488651,  0.10848097, -0.87868246]]))

    #Place location (Delivery)
    Rtarget.append(np.array([[-0.69296176, -0.70428022,  0.15425102],
                            [-0.70864763,  0.70473615 , 0.03413935],
                            [-0.13274994 ,-0.08565235, -0.98744171]]))
    while len(steps_to_be_done) !=0:
        step = steps_to_be_done.pop(0)
        actions = body_llm(task=step, object_list=object_list)
        print("i= ",i )
        print(actions)
        if isinstance(actions, str):
            if actions.lower().strip() == 'pass':
                pass
            else:
                #Feedback
                updated_plan = brain_llm(task=task,
                                        object_list=object_list,
                                        env_information=env_information,
                                        feedback_mode=True,
                                        feedback=actions,
                                        feedback_step=feedback_step,
                                        original_plan=plan)
                plan = plan[:feedback_step] + updated_plan
                steps_to_be_done = updated_plan.copy()
        elif isinstance(actions,list):
            for action in actions:
                actuator(Rtarget[i],ptarget=action[0],
                        duration=action[1],
                        gripper_command=action[2])
                # time.sleep(1.5)
            feedback_step = feedback_step + 1
            i+= 1
        else:
            raise Exception('Error in Body LLM output.')
        
    object_len=-1


