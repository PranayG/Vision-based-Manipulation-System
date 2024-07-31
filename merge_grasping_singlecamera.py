import os
import argparse
import torch
import numpy as np
import open3d as o3d
import time 
import cv2
import sys
import pickle

from camera import RealSenseCamera
from PIL import Image



from gsnet import AnyGrasp
from graspnetAPI import GraspGroup


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.045, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()


# sys.path.append('/usr/local/lib/python3.9/site-packages')
# pose= []
def demo(data_dir):

    # robot = FR3Real()
    # model = PinocchioModel()
    time.sleep(1)

    object_len = 0

    camera1 = RealSenseCamera(camera_serial_no=str(318122303427), VGA=True)
    # camera2 = RealSenseCamera(camera_serial_no=str(318122302882), VGA=True)

    #Camera1 = Left to base frame
    base_T_camera1 = np.array([[ 0.40889382,  0.43091743, -0.80443521,  0.77993618],
                            [ 0.90798509, -0.28047074,  0.31128643, -0.00220407],
                            [-0.09148179, -0.85769828, -0.50594935,  0.31299386],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])


    #Camera2 to = Right base frame
    # base_T_camera2 = np.array([[-0.46315684,  0.51939884, -0.71812992,  0.78216485],
    #                             [ 0.86486408,  0.08783968, -0.49426138,  1.06991923],
    #                             [-0.19363849, -0.85000531, -0.48989295,  0.32733943],
    #                             [ 0.        ,  0.        ,  0.        ,  1.        ]])


    # get camera intrinsics

    fx = 380.296875 
    fy = 380.02658081
    cx = 317.82562256
    cy = 242.7481842 

    scale = 1000.0

    #Workspace of the whiteboard
    # xmin, xmax = -0.5, 0.2  #width of table  xmin is the left side of the camera  , xmax = towards the right side of the camera
    # ymin, ymax = -0.24, 0.03   # height of the table -- >  for heighted objects on the table increase ymin ie. go further up from -0.1 to -0.2 ...
    #                         #  in order to decrease the height below the table , decrease ymax
    # zmin, zmax = 0.1,0.8 

    # xmin, xmax = -0.5, 0.2  #width of table  xmin is the left side of the camera  , xmax = towards the right side of the camera
    # ymin, ymax = -0.4, 0.1   # height of the table -- >  for heighted objects on the table increase ymin ie. go further up from -0.1 to -0.2 ...
    #                         #  in order to decrease the height below the table , decrease ymax
    # zmin, zmax = 0.3,0.8


    xmin, xmax = 0.1, 0.4  #width of table  xmin is the left side of the camera  , xmax = towards the right side of the camera
    ymin, ymax = -0.35, 0.03   # height of the table -- >  for heighted objects on the table increase ymin ie. go further up from -0.1 to -0.2 ...
                            #  in order to decrease the height below the table , decrease ymax
    zmin, zmax = 0.3,0.8

    lims = [xmin, xmax, ymin, ymax, zmin, zmax]


    # while(object_len <= 4):

    #Take a picture 
    # start = time.time()

    color1 , depth1 = camera1.saveAlignedDepth()

    cv2.imwrite("example_data/camera1RGB_test.png", color1)
    cv2.imwrite("example_data/camera1Depth_test.png", depth1.astype(np.uint16))

    camera1.close()

    colors1 = np.array(Image.open(os.path.join(data_dir, 'camera1RGB_test.png')), dtype=np.float32) / 255.0
    depths1 = np.array(Image.open(os.path.join(data_dir, 'camera1Depth_test.png')))


    # color2 , depth2 = camera2.saveAlignedDepth()

    # cv2.imwrite("example_data/camera2RGB_test.png", color2)
    # cv2.imwrite("example_data/camera2Depth_test.png", depth2.astype(np.uint16))

    # camera2.close() #Close camera object

    # # get data
    # colors2 = np.array(Image.open(os.path.join(data_dir, 'camera2RGB_test.png')), dtype=np.float32) / 255.0
    # depths2 = np.array(Image.open(os.path.join(data_dir, 'camera2Depth_test.png')))


    # xmin, xmax = -0.5,0.1
    # ymin, ymax = -0.15,0.15
    # zmin, zmax = 0.475, 1
    

    # get point cloud
    # 720 x 1280
    # print(depths.shape[0])

    '''
    Camera 1 = LEFT CAMERA
    '''
    xmap1, ymap1 = np.arange(depths1.shape[1]), np.arange(depths1.shape[0])
    xmap1, ymap1 = np.meshgrid(xmap1, ymap1)
    points1_z = depths1 / scale
    points1_x = (xmap1 - cx) / fx * points1_z
    points1_y = (ymap1 - cy) / fy * points1_z

    # remove outlier
    mask1 = (points1_z > 0) & (points1_z < 1)
    points_1 = np.stack([points1_x, points1_y, points1_z], axis=-1)
    points_1 = points_1[mask1].astype(np.float32)
    colors_1 = colors1[mask1].astype(np.float32)

 

    # '''
    # Camera 2 = RIGHT CAMERA
    # '''
    # xmap2, ymap2 = np.arange(depths2.shape[1]), np.arange(depths2.shape[0])
    # xmap2, ymap2 = np.meshgrid(xmap2, ymap2)
    # points2_z = depths2 / scale
    # points2_x = (xmap2 - cx) / fx * points2_z
    # points2_y = (ymap2 - cy) / fy * points2_z

    # # remove outlier
    # mask2 = (points2_z > 0) & (points2_z < 1)
    # points_2 = np.stack([points2_x, points2_y, points2_z], axis=-1)
    # points_2 = points_2[mask2].astype(np.float32)
    # colors_2 = colors2[mask2].astype(np.float32)

    '''
    Visualize PCD of workspace
    '''

    mask_x_1 = points_1[:,0] > xmin
    mask_x_2 = points_1[:,0] < xmax

    # Y Axis
    mask_y_1 = points_1[:,1] > ymin
    mask_y_2 = points_1[:,1] < ymax

    # Z Axis
    mask_z_1 = points_1[:,2] < zmin  # Closer to floor     
    mask_z_2 = points_1[:,2] > zmax # Clooser to ceiling

    mask_x = np.logical_and(mask_x_1, mask_x_2) # Along table's wide
    mask_y = np.logical_and(mask_y_1, mask_y_2) # Along table's longitude
    mask_z = np.logical_and(mask_z_1, mask_z_2) # Along table's height
    mask = np.logical_and(mask_x, mask_y, mask_z)

    pcd = o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(points_1[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors_1[mask])

    o3d.visualization.draw_geometries([pcd])

    # points_1 = points_1[mask].astype(np.float32)
    # colors_1 = colors_1[mask].astype(np.float32)

    start = time.time()
    gg, cloud = anygrasp.get_grasp(points_1, colors_1, lims)
    print("Time Taken", time.time() - start)
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    #Anygrasp gives output in the camera1 frame ( since all the points given to the model are in the camera1 frame)        

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:35]
    # trans_y = []
    
    # for i in range(len(gg_pick)):
    #     trans_y.append(gg_pick[i].translation[1])

    # trans_y_sorted = trans_y.copy()

    # trans_y_sorted.sort()

    # ind = trans_y.index(trans_y_sorted[0])
    # print(ind)
    # print(trans_y_sorted)

    # breakpoint()
    
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)
    print('grasp translation:', gg_pick[0].translation)
    print('grasp rotation:', gg_pick[0].rotation_matrix)

    # Visualize the PointCloud object
    # pcd = o3d.geometry.PointCloud()
    # pcd.points= o3d.utility.Vector3dVector(points_2)
    # pcd.colors = o3d.utility.Vector3dVector(colors_2)
    # o3d.visualization.draw_geometries([pcd])


    pcd = o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(points_1)
    pcd.colors = o3d.utility.Vector3dVector(colors_1)
    # o3d.visualization.draw_geometries([pcd])

    # camera2_T_Pose = np.zeros((4,4))
    # t_camera2_T_Pose = np.array(list(gg_pick[0].translation))
    # r_camera2_T_Pose =  np.array(list(gg_pick[0].rotation_matrix))
    # camera2_T_Pose [:3,:3] = r_camera2_T_Pose
    # camera2_T_Pose [0:3,3] = t_camera2_T_Pose
    # camera2_T_Pose [3,3] = 1



    camera1_T_Pose = np.zeros((4,4))
    t_camera1_T_Pose = np.array(list(gg_pick[0].translation))
    r_camera1_T_Pose =  np.array(list(gg_pick[0].rotation_matrix))
    camera1_T_Pose [:3,:3] = r_camera1_T_Pose
    camera1_T_Pose [0:3,3] = t_camera1_T_Pose
    camera1_T_Pose [3,3] = 1


    # print(camera2_T_Pose)
    

    # base_T_Pose = base_T_camera2 @ camera2_T_Pose
    base_T_Pose = base_T_camera1 @ camera1_T_Pose

    r_base_T_Pose= base_T_Pose[:3,:3]
    t_base_T_Pose = base_T_Pose[0:3,3]
    # print(t_base_T_Pose)

    print(base_T_Pose)

    # pose.append(base_T_Pose)

    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin = [0,0,0])
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud, coordinate_frame])
        o3d.visualization.draw_geometries([grippers[0], cloud , coordinate_frame])

    # base_T_hand = getHandPose(robot)

    return base_T_Pose


    # object_len=-1





if __name__ == '__main__':
    
    pose = demo('./example_data/')

    print("pose", pose)

    with open('pose.pkl', 'wb') as file:
        pickle.dump(pose, file)

    print(f"Data saved to {'pose.pkl'}")






    

    


