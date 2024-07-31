import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
# parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
# parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--gripper_height', type=float, default=0.045, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
# cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'test2/color_3body_test2.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'test2/depth_3body_test2.png')))

    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0

    # fx,fy = 659.9139240696829, 663.6905028432622
    # cx,cy = 646.3796860837847, 360.59616797397973
    

    fx = 380.296875 
    fy = 380.02658081
    cx = 317.82562256
    cy = 242.7481842 

    scale = 1000.0

    # # set workspace
    # xmin, xmax = -0.19, 0.12
    # ymin, ymax = 0.02, 0.15
    # zmin, zmax = 0.0, 1.0

    xmin, xmax = -0.5,0.1
    ymin, ymax = -0.15,0.15
    zmin, zmax = 0.475, 1


    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    # 720 x 1280
    # print(depths.shape[0])
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
    print('grasp translation:', gg_pick[0].translation)
    print('grasp rotation:', gg_pick[0].rotation_matrix)


    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin = [0,0,0])
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud, coordinate_frame])
        o3d.visualization.draw_geometries([grippers[0], cloud , coordinate_frame])



if __name__ == '__main__':
    
    demo('./example_data/')
