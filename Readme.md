# Code Release for Depth Completion of Spacecraft Sparse Point Clouds

This repository contains the code release for the paper "Spacecraft Depth Completion from Sparse LiDAR Data under Adverse Illumination". 

The code is written in Python and uses PyTorch for training and inference. The code is organized as follows:

- `dataloader`: contains the data loading and preprocessing code.
- `models`: contains the neural network architectures for depth completion. 
- `train.py`: the main training script.
- `Ground_truth_transfomation.py`: the script to get the ground truth transformation from data name.
- `ICP.py`: the script to apply ICP algorithm to align the adjacent point cloud
- `refine.py`: the script to get plane depth map.
- `pre-process-depth`: the script to pre-process the depth map for plane depth.
