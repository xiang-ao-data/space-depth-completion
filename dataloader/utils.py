"""
This script is used to help load pointcloud data, process image
"""

import numpy as np
import open3d as o3d
import torch
from torchvision import transforms
import random
import cv2

def add_shadow(rgb):
    _, binary = cv2.threshold(rgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow = np.zeros_like(binary, np.uint8)
    size = 100 # 方形阴影的尺寸
    # 找到物体的边界
    y, x = np.where(binary == 255)
    top_left = min(x), min(y)
    bot_right = max(x), max(y)
    poly = []
    for i in range(6):
        x1 = np.random.randint(min(x),max(x))
        y1 = np.random.randint(min(y),max(y))
        poly.append([x1,y1])

    vertices = np.array(poly, dtype=np.int32)
    cv2.fillPoly(shadow, [vertices], (1))


    # 创建掩码，用以将阴影叠加到物体上
    mask = np.zeros_like(binary, np.uint8)
    mask[binary == 255] = 1
    shadow = cv2.bitwise_and(shadow, mask)
    low_img = rgb/10

    # 叠加阴影到原始图像
    img_with_shadow = np.where(shadow==1,low_img,rgb).astype(np.uint8)
    return img_with_shadow



def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def read_point_cloud_from_txt(filename):
    point_cloud = np.loadtxt(filename, delimiter='\t')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd

def dense_depth_hole_fill(dense,max_depth=45,blur_type='bilateral',kernel_size=3,blur_size=5):
    CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

    CROSS_KERNEL_5 = np.asarray(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)

    # 5x5 diamond kernel
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)

    # 7x7 cross kernel
    CROSS_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.uint8)

    # 7x7 diamond kernel
    DIAMOND_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.uint8)

    DIAMOND_KERNEL_9 = np.asarray(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.uint8)

    DIAMOND_KERNEL_13 = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)

    DIAMOND_KERNEL_15 = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)

    valid_pixels = (dense > 0.1)
    dense[valid_pixels] = max_depth - dense[valid_pixels]
    # Dilate
    kernel_dict = {'3':CROSS_KERNEL_3,'5':CROSS_KERNEL_5,'7':CROSS_KERNEL_7,'9':DIAMOND_KERNEL_9,'13':DIAMOND_KERNEL_13,'15':DIAMOND_KERNEL_15}
    kernel = kernel_dict[str(kernel_size)]
    depth_map = cv2.dilate(dense, kernel)
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64
    #
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, blur_size, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (blur_size,blur_size), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    return depth_map


# camera intrinsics matrix
fx = 2100
fy = 2100
cx = 1920/2
cy = 1080/2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

image_height = 1080
image_width = 1920


def point_cloud_to_depth_image(pcd, K=K):
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    for point in np.asarray(pcd.points):
        x = int(-point[0] * K[0, 0] / point[2] + K[0, 2])
        y = int(point[1] * K[1, 1] / point[2] + K[1, 2])
        if (0 <= x < image_width) and (0 <= y < image_height):
            depth_image[y, x] = -point[2]
    return depth_image

def depthmap_2_pointcloud(depthmap):
    pcd = o3d.geometry.PointCloud()
    height = depthmap.shape[0]
    width = depthmap.shape[1]
    if height != 1080 or width != 1920:
        new_depth = np.zeros((1080,1920),dtype=np.float32)
        new_depth[1080//2-height//2:1080//2+height//2,1920//2-width//2:1920//2+width//2] = depthmap
        depthmap = new_depth
        height = 1080
        width = 1920
    points = []
    for i in range(height):
        for j in range(width):
            if depthmap[i,j] !=0:
                z = -depthmap[i,j]
                x = -(j-cx)*z/K[0,0]
                y = (i-cy)*z/K[1,1]
                points.append([x,y,z])
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def crop_img(img,shape=(640,960)):
    h = img.shape[0]
    w = img.shape[1]
    crop = img[h//2-shape[0]//2:h//2+shape[0]//2,w//2-shape[1]//2:w//2+shape[1]//2]
    return crop

def crop_val(img,shape=(640,960)):
    h = img.shape[0]
    w = img.shape[1]
    crop = img[h//2-shape[0]//2:h//2+shape[0]//2,w//2-shape[1]//2:w//2+shape[1]//2]
    return crop

def image_augment(datas,args):
    data_augment = []
    for data in datas:
        if args.val ==False:
            data = crop_img(data)
        else: 
            data = crop_val(data)
        data = transforms.functional.to_tensor(data)
        data_augment.append(data)
    return data_augment

# file_path = "../0_1.png"
# img = cv2.imread(file_path,0)
# shadow = add_shadow(img)
# cv2.imwrite("shadow.png",shadow)