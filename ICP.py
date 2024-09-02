"""
The RANSAC+ICP registration pipeline
"""
import open3d as o3d
import numpy as np
import copy

class pointcloud_registration():
    def __init__(self) -> None:
        self.pcd_list = []
    
    def set_input(self,pcd):
        self.pcd_list = pcd

    def preprocess_point_cloud(self,pcd, voxel_size):
        if voxel_size ==0:
            pcd_down = pcd
        else:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = 0.2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = 0.4
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def draw_registration_result(self,source, target, transformation):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, -40])
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        target_temp.transform(transformation)
        # source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp,target_temp,coordinate_frame])


    def prepare_dateset(self,source,target,voxel_size):
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source_down,source_fpfh,target_down,target_fpfh

    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                    target_fpfh):
        distance_threshold = 0.05
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def icp_refine(self,source, target,transformation):
        distance_threshold = 0.04
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        return result

    def metrix(self,predict_trans,truth_trans):
        mse = np.mean((predict_trans - truth_trans) ** 2)
        rmse = np.sqrt(mse)
        t1= predict_trans[0:3, 0:3]
        t2 = truth_trans[0:3, 0:3]
        et =np.sum((t1-t2)**2)
        et = np.sqrt(et)
        r1 = predict_trans[:,3]
        r2 = truth_trans[:,3]
        er = np.sum((r1-r2)**2)
        er = np.sqrt(er)
        return rmse,et,er

    def run(self,voxel_size=0.05,color= False):
        assert len(self.pcd_list) > 1
        final_pcd = self.pcd_list[0].voxel_down_sample(voxel_size)
        for i in range(len(self.pcd_list)-1):
            source_down,source_fpfh,target_down,target_fpfh = self.prepare_dateset(self.pcd_list[i+1],final_pcd,voxel_size)
            ransac = self.execute_global_registration(source_down,target_down,source_fpfh,target_fpfh)
            icp = self.icp_refine(self.pcd_list[i+1],final_pcd,ransac.transformation)
            if color:
                self.pcd_list[i+1].paint_uniform_color([np.random.rand(),np.random.rand(),np.random.rand()])
            else:
                print(i)
                final_pcd = self.pcd_list[i+1].transform(icp.transformation) + final_pcd
        final_pcd = final_pcd.voxel_down_sample(voxel_size)
        return final_pcd

    def test(self,pcd1,pcd2,voxel_size=0.01):
        source_down, source_fpfh, target_down, target_fpfh = self.prepare_dateset(pcd1,pcd2,voxel_size)
        ransac = self.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        icp = self.icp_refine(pcd1,pcd2, ransac.transformation)
        transform = icp.transformation
        return transform
    

        
            


