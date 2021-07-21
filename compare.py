import os
import os.path
import numpy as np
from matplotlib import pyplot as plt

path1 = input(
    "Enter ground truth text folder: ")  # path of text file #D:\Core_point_detection\Core_point_GT\FUT2\test\g_Truth
path2 = input(
    "Enter predicted ground truth text folder: ")  #D:\Core_point_detection\Core_point_GT\FUT2\Predictions
List_txt1 = os.listdir(path1)
List_txt2 = os.listdir(path2)
tpr5, tpr10, tpr15, tpr20, tpr25, tpr30, tpr35, tpr40, tpr45, tpr50, tpr55 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
fpr5, fpr10, fpr15, fpr20, fpr25, fpr30, fpr35, fpr40, fpr45, fpr50, fpr55 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
class1 = '0'
sum = np.array([])
for file1 in List_txt1:
    fo = open(path1 + '/' + file1, "r+")
    line = fo.readlines()
    xa, ya = (line[0].strip()).split(" ")
    xa = float(xa)
    ya = float(ya)
    fp = open(path2 + '/' + file1, "r+")
    line = fp.readlines()
    xp, yp = (line[0].strip()).split(" ")
    xp = float(xp)
    yp = float(yp)
    point1 = np.array((xa, ya))
    point2 = np.array((xp, yp))
    dist = np.linalg.norm(point1 - point2)
    sum = np.append(sum, dist)
    print(file1, xa, ya, xp, yp)
    if dist <= 5:
        tpr5 = tpr5 + 1
    else:
        fpr5 = fpr5 + 1
    if dist <= 10:
        tpr10 = tpr10 + 1
    else:
        fpr10 = fpr10 + 1
    if dist <= 15:
        tpr15 = tpr15 + 1
    else:
        fpr15 = fpr15 + 1
    if dist <= 20:
        tpr20 = tpr20 + 1
    else:
        fpr20 = fpr20 + 1
    if dist <= 25:
        tpr25 = tpr25 + 1
    else:
        fpr25 = fpr25 + 1
    if dist <= 30:
        tpr30 = tpr30 + 1
    else:
        fpr30 = fpr30 + 1
    if dist <= 35:
        tpr35 = tpr35 + 1
    else:
        fpr35 = fpr35 + 1
    if dist <= 40:
        tpr40 = tpr40 + 1
    else:
        fpr40 = fpr40 + 1
    if dist <= 45:
        tpr45 = tpr45 + 1
    else:
        fpr45 = fpr45 + 1
    if dist <= 50:
        tpr50 = tpr50 + 1
    else:
        fpr50 = fpr50 + 1
    if dist <= 55:
        tpr55 = tpr55 + 1
    else:
        fpr55 = fpr55 + 1
    print(dist)
    fo.close()
    fp.close()
print("5", 100*float(tpr5/(tpr5+fpr5)))
print("10", 100*float(tpr10/(tpr10+fpr10)))
print("15", 100*float(tpr15/(tpr15+fpr15)))
print("20", 100*float(tpr20/(tpr20+fpr20)))
print("25", 100*float(tpr25/(tpr25+fpr25)))
print("30", 100*float(tpr30/(tpr30+fpr30)))
print("35", 100*float(tpr35/(tpr35+fpr35)))
print("40", 100*float(tpr40/(tpr40+fpr40)))
print("45", 100*float(tpr45/(tpr45+fpr45)))
print("50", 100*float(tpr50/(tpr50+fpr50)))
print("55", 100*float(tpr55/(tpr55+fpr55)))
bins = np.arange(-100, 100, 5)  # fixed bin size

data = np.random.normal(0, 20, 1000)

plt.xlim([-5, 100])

plt.hist(sum, bins=bins, alpha=0.5)
plt.title('Eucledean distance between prediction and ground truth')
plt.xlabel('deviation in pixels (bin size = 5)')
plt.ylabel('count')

plt.savefig('D:\Core_point_detection\Code_SPNet')
