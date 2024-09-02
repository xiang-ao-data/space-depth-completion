"""
To faster the training, the plane depth is pre-calculated and saved as float
"""

from refine import refine_module
from dataloader.utils import *
import os
import imageio.v2 as imageio
from tqdm import tqdm

depth_path = './val'
refine_module = refine_module()
depth_list = []
list = os.listdir(depth_path)
for j in tqdm(range(len(list))):
    path = list[j]
    for i in range(36):
        depth = os.path.join(depth_path,path,str(i) + '00000.pcd')
        depth = read_pcd(depth)
        depth = point_cloud_to_depth_image(depth)
        max_d = np.max(depth)
        min_d = np.min(depth)
        depth = dense_depth_hole_fill(depth,kernel_size=15)
        refine_depth = refine_module.forward(normal=None, depth=depth)
        refine = refine_depth.squeeze(0).detach().numpy()
        refine[refine>max_d] =0
        refine[refine<min_d] =0
        imageio.imsave(os.path.join(depth_path,path,str(i) + '_refine.tif'), refine)


