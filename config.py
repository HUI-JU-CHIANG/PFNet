"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration
 
"""
import os

backbone_path = 'D:/Olia/結構化機器學習/CVPR2021_PFNet-main/backbone/resnet/resnet50-19c8e357.pth'

datasets_root = 'D:/Olia/DATASET/COD10K'

cod_training_root = os.path.join(datasets_root, 'TrainDataset/TrainDataset')

chameleon_path = os.path.join(datasets_root, 'TestDataset/TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/TestDataset/COD10K')
#nc4k_path = os.path.join(datasets_root, 'test/NC4K')
