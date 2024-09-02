"""
use plane constraint to refine the depth map
1. detect plane from coarse depth and normal
2. get plane model
3. calculate final depth map
"""
import open3d as o3d
import numpy as np
import torchvision.transforms.functional
from loss.Loss import MaskL1loss
from dataloader.utils import *

class refine_module():
    def __init__(self) -> None:
        self.normal = None
        self.depth = None
        self.fx = 2100
        self.fy = 2100
        self.cx = 1920/2
        self.cy = 1080/2
        self.K = np.array([[self.fx, 0, self.cx],
              [0, self.fy, self.cy],
              [0, 0, 1]])
        self.points = None
        self.image_shape = [1080,1920]
        self.init_c_matrix(self.image_shape[0],self.image_shape[1])
        self.mask_list = []

    def forward(self,normal,depth):
        self.normal = normal
        self.depth = depth
        self.getmask(visual=False)
        init_depth = torchvision.transforms.functional.to_tensor(self.depth)
        refine_depth = torch.zeros_like(init_depth)
        for mask in self.mask_list:
            depth = self._d_to_depthmap(mask)
            refine_depth = refine_depth + depth
        valid = refine_depth >0
        refine_depth[~valid] = 0
        return refine_depth

    def _depthimg_2_point(self,depthmap):
        pcd = o3d.geometry.PointCloud()
        height = depthmap.shape[0]
        width = depthmap.shape[1]
        points = []
        for i in range(height):
            for j in range(width):
                if depthmap[i,j] !=0:
                    z = -depthmap[i,j]
                    x = -(j-self.cx)*z/self.K[0,0]
                    y = (i-self.cy)*z/self.K[1,1]
                    points.append([x,y,z])
        pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = pcd
        return pcd
    
    def _point_to_depthimg(self,pcd):
        depthmap = np.zeros((1080, 1920), dtype=np.float32)
        for point in np.asarray(pcd.points):
            x = int(-point[0] * self.K[0,0] / point[2] + self.K[0,2])
            y = int(point[1] * self.K[1,1] / point[2] + self.K[1,2])
            if (0 <= x < 1080) and (0 <= y < 1920):
                depthmap[y, x] = -point[2]
        self.depth_img = depthmap
        return depthmap
    
    def _plane_detect(self,pcd,removal=False):
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
        inliers_pcd = pcd.select_by_index(inliers)
        if removal:
            pcd = pcd.select_by_index(inliers, invert=True)
        return plane_model,inliers_pcd,pcd

    def _cloud_to_mask(self,plane_model,pcd):
        [a,b,c,d]=plane_model
        mask = np.zeros((self.image_shape[0],self.image_shape[1],5), dtype=np.float32)
        for point in np.asarray(pcd.points):
            x = int(-point[0] * self.K[0,0] / point[2] + self.K[0,2])
            y = int(point[1] * self.K[1,1] / point[2] + self.K[1,2])
            mask[y,x,0] = d
            mask[y,x,1:4] = [a,b,c]
            # k = point[0]*a + point[1]*b + point[2]*c +d
        mask = torchvision.transforms.functional.to_tensor(mask)
        return mask

    def _d_to_depthmap(self,mask):
        d = mask[0,:,:]
        normal = mask[1:4,:,:]
        points = self.points
        m = torch.sum(torch.mul(points, normal), dim=0, keepdim=True)
        m = torch.where(m==0,torch.ones_like(m),m)
        depth_new = d / m
        return depth_new


    def init_c_matrix(self,height,width): # initial the camera matrix for faster calculation
        points = torch.zeros([3,height,width])
        for i in range(points.shape[1]):
            for j in range(points.shape[2]):
                points[0, i, j] = -(j - self.K[0, 2]) / self.K[0, 0]
                points[1, i, j] = (i - self.K[1, 2]) / self.K[1, 1]
                points[2, i, j] = 1
        self.points = points
        return points

    def getmask(self,visual=False):
        pcd = self._depthimg_2_point(self.depth)
        pcd_backup = pcd
        plane_pcd=[]
        mask_list = []
        while len(pcd.points) > 0.2 * len(pcd_backup.points):
            plane_model,inliers_pcd,pcd = self._plane_detect(pcd,removal=True)
            inliers_pcd.paint_uniform_color([np.random.rand(),np.random.rand(),np.random.rand()])
            plane_pcd.append(inliers_pcd)
            mask = self._cloud_to_mask(plane_model,inliers_pcd)
            mask_list.append(mask)
        if visual ==True:
            o3d.visualization.draw_geometries(plane_pcd)
        self.mask_list = mask_list

