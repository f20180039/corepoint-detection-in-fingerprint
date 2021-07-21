from sklearn.model_selection import train_test_split
import cv2
import os
from shutil import copyfile, move
BasePath = input("Enter dataset to split: ") #.....FUT phase1

files = []
imagePath = os.path.join(BasePath, "data_used")
gTruthPath = os.path.join(BasePath, "g_Truth")
MaskPath = os.path.join(BasePath, "Mask_gt")

test_path = os.path.join(BasePath, 'test/data_used')
test_gt_path = os.path.join(BasePath, 'test/g_Truth')
test_mask_path = os.path.join(BasePath, 'test/Mask_gt')

# train_path = os.path.join(BasePath, 'train/data_used')
# train_mask_path = os.path.join(BasePath, 'train/Mask_gt')
# train_gt_path = os.path.join(BasePath, 'train/g_Truth')

files = os.listdir(imagePath)

import random as rd

selection = rd.sample(range(len(files)), int(len(files) * 0.2)) # 20% data for testing
print(selection)
for ele in selection:
    save_f = files[ele]
    save_gt = files[ele][:-4] + '_gt.txt'
    move(os.path.join(imagePath, save_f), test_path)
    move(os.path.join(MaskPath, save_f), test_mask_path)
    move(os.path.join(gTruthPath, save_gt), test_gt_path)
