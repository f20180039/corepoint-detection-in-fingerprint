import cv2
import numpy as np
import os.path
from os import path

file1 = open('gt_all_data.txt')
path1 = input("Enter training data folder: ") #D:/Core_point_detection/Core_point_GT/FUT phase1
files = file1.readlines()
for fil in files:
	fil1 = fil.split(",")
	img = cv2.imread(os.path.join(os.path.join(path1, "data_used"), fil1[0]))
	if path.exists(os.path.join(os.path.join(path1, "data_used"), fil1[0])) is False:
		continue
	mask = np.zeros(img.shape[:2], dtype="uint8")
	cv2.rectangle(mask, (int(fil1[1]), int(fil1[2])), (int(fil1[3]), int(fil1[4])), 255, -1)
	output = os.path.join(os.path.join(path1, "Mask_gt"), fil1[0])
	cv2.imwrite(output, mask)
