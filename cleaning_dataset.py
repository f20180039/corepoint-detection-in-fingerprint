import os

BasePath = input("Enter dataset to split: ")
datapath = os.path.join(BasePath, "data_used")
GTpath = os.path.join(BasePath, "g_Truth")
#Mpath = os.path.join(BasePath, "Mask_gt")

# Delete excess files from datapath
files = os.listdir(datapath)

count = [0, 0]
for ele in sorted(files):
    if os.path.exists(os.path.join(datapath, ele)) and os.path.exists(
            os.path.join(GTpath, ele[:-4] + "_gt.txt")):
        count[1] += 1
        continue
    else:
        try:
            os.remove(os.path.join(datapath, ele))
            count[0] += 1
            print('deleted', os.path.join(datapath, ele), count[0])
        except:
            os.remove(os.path.join(GTpath, ele[:-4] + "_gt.txt"))
            count[1] += 1
            print('deleted', os.path.join(GTpath, ele[:-4] + "_gt.txt"), count[1])

print(count)
# Delete excess files from GTpath
files = os.listdir(GTpath)

count = [0, 0]

for ele in sorted(files):
    if os.path.exists(os.path.join(datapath, ele[:-7] + '.bmp')) and os.path.exists(
            os.path.join(GTpath, ele)):  # and os.path.exists(os.path.join(Mpath, ele[:-7]+'.bmp')):
        count[0] += 1
        continue
    else:
        os.remove(os.path.join(GTpath, ele))
        count[1] += 1
        print('deleted', os.path.join(GTpath, ele), count[1])
print(count)
