'''
get the ground truth transformation from data name
'''
import math
import os
import numpy as np
import open3d as o3d

def initial_transformation(pcd_path1,pcd_path2):
    name1, extension = os.path.splitext(os.path.basename(pcd_path1))
    name2, extension = os.path.splitext(os.path.basename(pcd_path2))
    angle1 = int(name1)/10000
    angle2 = int(name2)/10000
    angle1 = math.radians(angle1)
    angle2 = math.radians(angle2)
    delta_z_1 = -(40 - 40 * math.cos(angle1))
    delta_x_1 = 40 * math.sin(angle1) / 2
    theta_new1 = math.atan2(40 * math.sin(angle1) / 2, 40 * math.cos(angle1))
    matrix_1 = np.asarray([[np.cos(theta_new1), 0, np.sin(theta_new1), delta_x_1],
                           [0, 1, 0, 0],
                           [-np.sin(theta_new1), 0, np.cos(theta_new1), delta_z_1],
                           [0, 0, 0, 1]])
    delta_z_2 = -(40 - 40 * math.cos(angle2))
    delta_x_2 = 40 * math.sin(angle2) / 2
    theta_new2 = math.atan2(40 * math.sin(angle2) / 2, 40 * math.cos(angle2))
    matrix_2 = np.asarray([[np.cos(theta_new2), 0, np.sin(theta_new2), delta_x_2],
                           [0, 1, 0, 0],
                           [-np.sin(theta_new2), 0, np.cos(theta_new2), delta_z_2],
                           [0, 0, 0, 1]])
    ground_truth = np.dot(np.linalg.inv(matrix_1), matrix_2) # T（1-2）= T(1-0)-1 * T(2-0)
    return ground_truth

def random_transform(initial_matrix):
    random_matrix = np.zeros((4,4))
    Q, _ = np.linalg.qr(np.random.rand(3, 3))
    random_matrix[:3,:3] = Q
    random_matrix[0:3,3] = np.random.rand(1,3)
    random_matrix[3,3] = 1
    matrix = np.dot(random_matrix,initial_matrix)
    return random_matrix,matrix

def test():
    source_path = "./val/Aquire_1/100000.pcd"
    target_path = "./val/Aquire_1/1000000.pcd"
    initial_matrix = initial_transformation(source_path,target_path)
    random,matrix = random_transform(initial_matrix)
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    source_new = source.transform(random)
    target_new = target.transform(matrix)
    source.paint_uniform_color([1, 0.706, 0])
    target_new.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_new,target_new])

# test()
