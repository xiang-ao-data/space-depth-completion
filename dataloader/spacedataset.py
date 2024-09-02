import torch.utils.data as data
from dataloader.utils import *
import os
import cv2
import torch
import argparse
import imageio
import random

full_path = '/home/ubuntu/myproject/spacecraft-3d-reconstruction/full'
val_path = '/home/ubuntu/myproject/spacecraft-3d-reconstruction/val'

# train shadowed image
shadow_free_path = '/home/ubuntu/myproject/spacecraft-3d-reconstruction/shadow_dataset/no_shadow/val'
shadow_path = '/home/ubuntu/myproject/spacecraft-3d-reconstruction/shadow_dataset/shadow/val'


class Spacedateset(data.Dataset):
    def __init__(self,args):
        self.args = args
        if args.val:
            self.dataset_path = val_path
        else:
            self.dataset_path = full_path
        self.file_path = os.listdir(self.dataset_path)
        self.all_file = None
        self.get_all_file_name()
        self.prefill = args.prefill
        self.min_depth = 15
    
    def get_all_file_name(self):
        sparse_depth = []
        rgb_image = []
        gt = []
        plane = []
        for path in self.file_path:
            for i in range(0,36):
                sparse_depth.append(os.path.join(self.dataset_path,path,str(i)+'00000.pcd'))
                rgb_image.append(os.path.join(self.dataset_path,path,str(i)+'_1.png'))
                gt.append(os.path.join(self.dataset_path,path,str(i)+'_depth.txt'))
                plane.append(os.path.join(self.dataset_path,path,str(i)+'_refine.tif'))
        self.all_file = {'sparse_depth':sparse_depth,'rgb_image':rgb_image,'gt':gt,'plane':plane}

    def __len__(self):
        return len(self.all_file['sparse_depth'])

    def foreground_extraction(self,rgb,depth):
        gray = rgb
        valid = np.where(depth>0,255,0)
        ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        valid = np.where(th>0,255,valid)
        return valid

    def __getitem__(self, index):
        sparse_depth = self.all_file['sparse_depth'][index]
        rgb_image = self.all_file['rgb_image'][index]
        gt = self.all_file['gt'][index]
        plane_depth = self.all_file['plane'][index]
        plane_depth = imageio.imread(plane_depth)
        sparse_pcd = read_pcd(sparse_depth)
        sparse_depth_image = point_cloud_to_depth_image(sparse_pcd)
        min_depth = np.min(sparse_depth_image[sparse_depth_image>0])
        if self.prefill:
            sparse_depth_image = dense_depth_hole_fill(sparse_depth_image,kernel_size=15)
            valid = sparse_depth_image>min_depth
            sparse_depth_image[~valid] =0
        gt_pcd = read_point_cloud_from_txt(gt)
        gt_depth_image = point_cloud_to_depth_image(gt_pcd)
        gt_depth_image = dense_depth_hole_fill(gt_depth_image,kernel_size=5)
        valid = gt_depth_image>min_depth
        gt_depth_image[~valid] = 0
        rgb_image_gt = cv2.imread(rgb_image,0)
        if self.args.with_shadow:
            # 一半概率添加阴影
            if np.random.rand()>0.5:
                rgb_image = add_shadow(rgb_image_gt)
            else:
                rgb_image = rgb_image_gt
        else:
            rgb_image = rgb_image_gt
        max_depth = np.max(sparse_depth_image)
        plane_depth[plane_depth>max_depth] = 0
        plane_depth[plane_depth<min_depth] = 0
        mask = self.foreground_extraction(rgb_image,sparse_depth_image)
        [sparse_depth,rgb,gt,plane_depth,mask,rgb_image_gt] = image_augment([sparse_depth_image,rgb_image,gt_depth_image,plane_depth,mask,rgb_image_gt],self.args)
        sparse_depth = sparse_depth.float()
        rgb = rgb.float()
        gt = gt.float()
        plane_depth = plane_depth.float()
        items = {'gt': gt, 'rgb': rgb, 'sparse': sparse_depth,'plane':plane_depth,'mask':mask,"rgb_gt":rgb_image_gt}
        return items

class Spaceshadowset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.shadow = shadow_path
        self.shadow_free = shadow_free_path
        self.file_path = os.listdir(self.shadow_free)
        self.all_file = None
        self.get_all_file_name()
        self.prefill = args.prefill
        self.min_depth = 15

    def get_all_file_name(self):
        sparse_depth = []
        rgb_image = []
        gt = []
        plane = []
        for path in self.file_path:
            for j in range(10):
                rgb_path = path+'_'+str(j+1)
                for i in range(0, 36):
                    sparse_depth.append(os.path.join(self.shadow_free, path, str(i) + '_noisy00000.pcd'))
                    rgb_image.append(os.path.join(self.shadow, rgb_path, str(i) + '_1.png'))
                    gt.append(os.path.join(self.shadow_free, path, str(i) + '_depth.txt'))
                    plane.append(os.path.join(self.shadow_free, path, str(i) + '_refine.tif'))
        self.all_file = {'sparse_depth': sparse_depth, 'rgb_image': rgb_image, 'gt': gt, 'plane': plane}

    def __len__(self):
        return len(self.all_file['sparse_depth'])

    def foreground_extraction(self, rgb, depth):
        gray = rgb
        valid = np.where(depth > 0, 255, 0)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        valid = np.where(th > 0, 255, valid)
        return valid

    def __getitem__(self, index):
        sparse_depth = self.all_file['sparse_depth'][index]
        rgb_image = self.all_file['rgb_image'][index]
        gt = self.all_file['gt'][index]
        plane_depth = self.all_file['plane'][index]
        plane_depth = imageio.imread(plane_depth)
        sparse_pcd = read_pcd(sparse_depth)
        sparse_depth_image = point_cloud_to_depth_image(sparse_pcd)
        min_depth = np.min(sparse_depth_image[sparse_depth_image > 0])
        if self.prefill:
            sparse_depth_image = dense_depth_hole_fill(sparse_depth_image, kernel_size=15)
            valid = sparse_depth_image > min_depth
            sparse_depth_image[~valid] = 0
        gt_pcd = read_point_cloud_from_txt(gt)
        gt_depth_image = point_cloud_to_depth_image(gt_pcd)
        gt_depth_image = dense_depth_hole_fill(gt_depth_image, kernel_size=5)
        valid = gt_depth_image > min_depth
        gt_depth_image[~valid] = 0
        rgb_image = cv2.imread(rgb_image, 0)
        max_depth = np.max(sparse_depth_image)
        plane_depth[plane_depth > max_depth] = 0
        plane_depth[plane_depth < min_depth] = 0
        mask = self.foreground_extraction(rgb_image, sparse_depth_image)
        [sparse_depth, rgb, gt, plane_depth, mask] = image_augment(
            [sparse_depth_image, rgb_image, gt_depth_image, plane_depth, mask], self.args)
        sparse_depth = sparse_depth.float()
        rgb = rgb.float()
        gt = gt.float()
        plane_depth = plane_depth.float()
        items = {'gt': gt, 'rgb': rgb, 'sparse': sparse_depth, 'plane': plane_depth, 'mask': mask}
        return items




