from scipy.spatial.transform import Rotation as R
import pickle
import numpy as np

with open('hand_pose.pkl', 'rb') as file: 
    
    camera1_T_Pose = pickle.load(file) 

print(camera1_T_Pose)

# r_base_T_Pose= base_T_Pose[:3,:3]
# t_base_T_Pose = base_T_Pose[0:3,3]

# quat_r_base_T_Pose = R.from_matrix(r_base_T_Pose).as_quat()
# print(t_base_T_Pose)
# print("Quaternion of base_T_Pose= " , quat_r_base_T_Pose)

# base_T_align = np.zeros((4,4))
# quat_base_T_align = np.array([[0.953, 0.301, 0.007, -0.008]])
# t_base_T_align = np.array([[0.392, 0.422, 0.021]])
# r_base_T_align = R.from_quat(quat_base_T_align).as_matrix()

# base_T_align[:3,:3] = r_base_T_align
# base_T_align[0:3, 3] = t_base_T_align
# base_T_align[3,3] = 1

# print("Desired base_T_align= ", base_T_align)


# Pose_T_aligned = np.linalg.inv(base_T_Pose) @ base_T_align
# print("Pose_T_aligned = ",Pose_T_aligned)

# quat_base_T_return = np.array([[1, 0.0, 0.00, -0.00]])
# r_base_T_return = R.from_quat(quat_base_T_return).as_matrix()

# print("ReturnRotation Matrix= ", r_base_T_return)
# base_T_aligned = base_T_Pose @ Pose_T_aligned

# # print(base_T_aligned)

# Pose_T_aligned = np.array([[ 0.12168533,  0.0056451 ,  0.99255269 , 0.03976486],
#                             [-0.00560859 , 0.99997177, -0.0049997 ,  0.01692713],
#                             [-0.99255292 ,-0.00495843 , 0.12171357 , 0.01268308],
#                             [ 0. ,         0. ,         0.,          1.        ]])

# base_T_aligned = base_T_Pose @ Pose_T_aligned

# r_base_T_aligned= base_T_aligned[:3,:3]
# t_base_T_aligned = base_T_aligned[0:3,3]

# quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
# print("translation base_T_aligned= " , t_base_T_aligned)
# print("Quaternion of quat_r_base_T_aligned= " , quat_r_base_T_aligned)



# r_base_T_aligned= base_T_aligned[:3,:3]
# t_base_T_aligned = base_T_aligned[0:3,3]

# quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
# print("translation base_T_aligned= " , t_base_T_aligned)
# print("Quaternion of quat_r_base_T_aligned= " , quat_r_base_T_aligned)


# quat_base_T_return = np.array([[0.981, 0.013, 0.192, -0.023]])
# r_base_T_return = R.from_quat(quat_base_T_return).as_matrix()
# t_base_T_return = np.array([[0.020, 0.417, 0.523]])
# print("Rotation of quat_base_T_return= " , r_base_T_return)
# print("translation t_base_T_return= " , t_base_T_return)

# hand_T_camera = np.array([[-0.   , -1.   , -0.011,  0.052],
#                         [ 1.   , -0.   , -0.005, -0.032],
#                         [ 0.005, -0.011,  1.   , -0.001],
#                         [ 0.   ,  0.   ,  0.   ,  1.   ]])

hand_T_camera = np.array([[-0.003, -1.   , -0.014,  0.057],
                        [ 1.   , -0.003,  0.001, -0.043],
                        [-0.001, -0.014,  1.   , -0.005],
                        [ 0.   ,  0.   ,  0.   ,  1.   ]])

base_T_hand = np.array([[ 0.92651204 , 0.03800109 , 0.37434124,  0.00903795],
                        [ 0.02080557 ,-0.99853891 , 0.0498715  , 0.41212371],
                        [ 0.37568946 ,-0.03841816 ,-0.92594896 , 0.56214211],
                        [ 0.     ,     0.     ,     0.  ,        1.        ]])

Pose_T_aligned = np.array([[ 1, 0 , 0,    0.02808009],
                            [ 0 ,1, 0 ,-0.00558704],
                            [ 0 , 0, 1, 0.00320354],
                            [ 0,0.,0, 1]])

r_camera1_T_Pose= camera1_T_Pose[:3,:3]
t_camera1_T_Pose = camera1_T_Pose[0:3,3]

# quat_r_camera1_T_Pose = R.from_matrix(r_camera1_T_Pose).as_quat()
# print("translation t_camera1_T_Pose= " , t_camera1_T_Pose)
# print("Quaternion of r_camera1_T_Pose= " , quat_r_camera1_T_Pose)

# base_T_Pose = base_T_hand @ hand_T_camera@ camera1_T_Pose
# r_base_T_Pose= base_T_Pose[:3,:3]
# t_base_T_Pose = base_T_Pose[0:3,3]

# print("rotation r_base_T_Pose",r_base_T_Pose)
# quat_r_base_T_Pose = R.from_matrix(r_base_T_Pose).as_quat()
# print("Quaternion of quat_r_base_T_Pose= " , quat_r_base_T_Pose)
# print("translation t_base_T_Pose= " , t_base_T_Pose)



# base_T_desiredPose = np.zeros((4,4))
# base_T_desiredPose[0:3,3]= np.array([[0.379, 0.567, 0.112]])
# quat_base_T_desiredPose = np.array([[0.791, 0.610, -0.039, 0.011]])
# base_T_desiredPose[:3,:3] = R.from_quat(quat_base_T_desiredPose).as_matrix()
# base_T_desiredPose[3,3] = 1

# print(base_T_desiredPose)



# Pose_T_aligned = np.linalg.inv(camera1_T_Pose) @ np.linalg.inv(hand_T_camera) @ np.linalg.inv(base_T_hand) @ base_T_desiredPose

# print(Pose_T_aligned)

# base_T_aligned = base_T_hand @ hand_T_camera @ camera1_T_Pose @ Pose_T_aligned 
# print(base_T_aligned)
# r_base_T_aligned= base_T_aligned[:3,:3]
# t_base_T_aligned = base_T_aligned[0:3,3]

# quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
# print("Quaternion of t_base_T_aligned= " , t_base_T_aligned)
# print("translation t_base_T_aligned= " , t_base_T_aligned)

# quat_base_T_place = np.array([[0.999, -0.034, -0.015, -0.006]])
# r_base_T_place = R.from_quat(quat_base_T_place).as_matrix()
# print(r_base_T_place)


base_T_multiplied = np.array([[ 0.67600607 , 0.26195914 , 0.68890665 , 0.32138468],
                                [-0.273812   ,-0.77856676 , 0.56468372,  0.42364628],
                                [ 0.68428312 ,-0.57036391, -0.45458499  ,0.09424099],
                                [ 0.      ,    0.     ,     0.  ,        1.        ]])
r_base_T_multiplied = base_T_multiplied[:3,:3]
t_base_T_multiplied = base_T_multiplied[0:3,3]
quat_base_T_multiplied = R.from_matrix(r_base_T_multiplied).as_quat()
print("trans:", t_base_T_multiplied)
print("Quat_base_T_multiplied",quat_base_T_multiplied)



print("**********************************************8")


rotation_90_deg_y = R.from_euler('y', -90, degrees=True).as_matrix()

rotY_grasp = np.dot(rotation_90_deg_y, camera1_T_Pose[:3,:3])

camera1_T_Pose[:3,:3] = rotY_grasp

base_T_aligned = base_T_hand @ hand_T_camera @ camera1_T_Pose @ Pose_T_aligned 
print("base_T_aligned computed by post multiplying= ", base_T_aligned)
r_base_T_aligned = base_T_aligned[:3,:3]
t_base_T_aligned = base_T_aligned[0:3,3]
quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
print("Quaternion of t_base_T_aligned= " , quat_r_base_T_aligned)
print("translation t_base_T_aligned= " , t_base_T_aligned)




print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

base_T_aligned_rotated = np.array([[-0.4346455 , 0.42652857,  0.79332373 ,0.37427367],
                                    [-0.08808055, -0.89667606 , 0.43384417 , 0.575898  ],
                                    [ 0.89639236 , 0.11865775 , 0.42731744 , 0.11042284],
                                    [ 0.     ,     0.  ,       0.  ,       1.        ]])

t_base_T_aligned_rotated = base_T_aligned_rotated[0:3,3]
print("t_base_T_aligned_rotated = ", t_base_T_aligned_rotated)

r_base_T_aligned_rotated = base_T_aligned_rotated[:3,:3]

quat_base_T_aliged_rotated = R.from_matrix(r_base_T_aligned_rotated).as_quat()
print("quat_base_T_aliged_rotated= ", quat_base_T_aliged_rotated)


base_T_aligned = np.array([[ 0.64083662 , 0.42652857 ,-0.6384416 ,  0.37427367],
                            [ 0.20691979 ,-0.89667606 ,-0.39136064 , 0.575898  ],
                            [-0.7393897 ,  0.11865775, -0.66288938 , 0.11042284],
                            [ 0.    ,      0.     ,     0.   ,       1.        ]])

r_base_T_aligned = base_T_aligned[:3,:3]
quat_r_base_T_aligned = R.from_matrix(r_base_T_aligned).as_quat()
print("quat_r_base_T_aligned", quat_r_base_T_aligned)

t_base_T_aligned = base_T_aligned[0:3,3]
print("t_base_T_aligned", t_base_T_aligned)


quat_shouldbe_rot = np.array([[0.918, 0.284, 0.251, -0.113]])

print("rot_shouldbe_rot= ", R.from_quat(quat_shouldbe_rot).as_matrix())
rot_angle = R.from_quat(quat_shouldbe_rot)
angle_rad = rot_angle.magnitude()
angle_deg = np.degrees(angle_rad)


quat = R.from_matrix(r_base_T_aligned).as_quat()
angle_rad_base_T_aligned = R.from_quat(quat).magnitude()
print(angle_rad_base_T_aligned)

print(angle_rad)


r = R.from_euler('y', (angle_deg/2), degrees=True)
print(r.as_matrix())
rotateY = np.zeros((4,4))
rotateY[:3,:3]  = r.as_matrix()
rotateY[0:3,3] = [0,0,0]
rotateY[3,3] = 1

base_T_aligned_test_rotated = base_T_aligned @  rotateY
print(base_T_aligned_test_rotated)

quat_base_T_aligned_test_rotated = R.from_matrix(base_T_aligned_test_rotated[:3,:3]).as_quat()
print("quat_base_T_aligned_test_rotated" , quat_base_T_aligned_test_rotated)
t_base_T_aligned_test_rotated = base_T_aligned_test_rotated[0:3,3]
print("t_base_T_aligned_test_rotated", t_base_T_aligned_test_rotated)