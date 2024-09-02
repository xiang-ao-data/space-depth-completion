"""
predict dense depth from a rgb image and a depth map
"""
from refine import refine_module
from dataloader.utils import *
import cv2
import torch
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
from train import get_model,parse_args,predict
import time


depth_refine = refine_module()

img_path = './val/Clement_1/5_1.png'
gt_path = './val/Clement_1/5_depth.txt'
sparse_path = './val/Clement_1/500000.pcd'
plane_path = './val/Clement_1/5_refine.tif'

def load_checkpoint():
    args,net,backbone = get_model()
    ckeckpoint_b = torch.load('./checkpoint/backbone_best.pth',map_location=torch.device('cpu'))
    net.backbone.load_state_dict(ckeckpoint_b)
    ckeckpoint = torch.load("./checkpoint/cspn_best.pth",map_location=torch.device('cpu'))
    net.load_state_dict(ckeckpoint)
    device = torch.device("cuda")
    net.to(device)
    return net,args


def dense_predict(img,sparse_depth,net,plane_depth=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    [sparse_depth, img, plane_depth] = image_augment([sparse_depth,img,plane_depth],args=None)
    img = img.float()
    sparse_depth = sparse_depth.float()
    img = img.unsqueeze(0)
    sparse_depth = sparse_depth.unsqueeze(0)
    img= img.to(device)
    plane_depth = plane_depth.float().unsqueeze(0).to(device)
    valid = img[:, 0:1, :, :] > 0
    sparse_depth = sparse_depth.to(device)
    batch = {'rgb': img, 'd': sparse_depth,'plane':plane_depth}
    start = time.time()
    out = net(batch)
    depth = out['pred']
    depth[~valid] = 0
    end = time.time()
    print('time:',end-start)
    out = depth.detach().squeeze(0).squeeze(0).cpu().numpy()
    valid = out > 10
    out[~valid] = 0
    return out

def visual_pcd(depth_list):
    pcd_list = []
    for depth in depth_list:
        pcd = depthmap_2_pointcloud(depth)
        pcd = pcd.voxel_down_sample(0.01)
        pcd,ind = pcd.remove_radius_outlier(nb_points =15,radius=0.1)
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)

def color_depth(depth):
    max_depth = np.max(depth)
    min_depth = np.min(depth[depth>0])
    depth[depth>0] = (depth[depth>0]-min_depth)/(max_depth-min_depth)
    depth = depth*255
    depth = depth.astype(np.uint8)
    im_color = cv2.applyColorMap(depth,cv2.COLORMAP_JET)
    cv2.imwrite('out.png',im_color)
    



