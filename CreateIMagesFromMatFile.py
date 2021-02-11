import os
import shutil
import glob
import random
import numpy as np
import matplotlib
import h5py
import cv2
filepath = r"D:\MSC\Sem2\IVC\ClassificationTask\catdog\brainTumorData\1.mat"
import ntpath
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


class Patient(object):
    PID = ""
    image = ""
    label = ""
    tumorBorder = ""
    tumorMask = ""

    def __init__(self, PID, image, label, tumorBorder, tumorMask, mergedimage):
        self.PID = PID
        self.image = image
        self.label = label
        self.tumorBorder = tumorBorder
        self.tumorMask = tumorMask
        self.mergedImage = mergedimage


source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"Train"
TEST_PATH = r"Test"
VALID_PATH = r"Validation"
Data_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\BrainTumor\BrainTumor"
Label1_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label1"
Label2_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label2"
Label3_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label3"


#
# for file in os.listdir(Data_PATH):
#     ActualPath = Data_PATH + "\\" + file
#     path, dirs, files = next(os.walk(ActualPath))
#     to_be_moved = random.sample(glob.glob(ActualPath + "/*.mat"), len(files))
#
#     patients =[]
#     for f in enumerate(to_be_moved, 1):
#         matfile = h5py.File(f[1],'a')
#         p = Patient('', '', '', '', '','')
#         p.image = np.mat(matfile['cjdata/image'])
#         p.PID = np.array(matfile['cjdata/PID'])
#         p.label = np.array(matfile['/cjdata/label'])
#         p.tumorBorder = np.array(matfile['/cjdata/tumorBorder'])
#         p.tumorMask = np.array(matfile['/cjdata/tumorMask'])
#         sns.heatmap(p.image)
#         sns.heatmap(p.tumorMask)


if os.path.exists(Label1_PATH):
    shutil.rmtree(Label1_PATH)
if os.path.exists(Label2_PATH):
    shutil.rmtree(Label2_PATH)
if os.path.exists(Label3_PATH):
    shutil.rmtree(Label3_PATH)
for file in os.listdir(Data_PATH):
    ActualPath = Data_PATH + "\\" + file
    path, dirs, files = next(os.walk(ActualPath))
    to_be_moved = random.sample(glob.glob(ActualPath + "/*.mat"), len(files))

    if not os.path.exists(Label1_PATH):
        os.makedirs(Label1_PATH)
    if not os.path.exists(Label2_PATH):
        os.makedirs(Label2_PATH)
    if not os.path.exists(Label3_PATH):
        os.makedirs(Label3_PATH)

    for f in enumerate(to_be_moved, 1):
        matfile = h5py.File(f[1],'r')
        data = matfile.get('cjdata/label')
        data = np.array(data)
        value = int(data[0])
        image = matfile.get('cjdata/image')
        image = np.array(image)
        dest = Label1_PATH
        if value == 2:
            dest = Label2_PATH
        elif value == 3:
            dest = Label3_PATH
        fileFullPath = os.path.splitext(f[1])
        img = np.array(image, dtype=np.float32)
        filename = os.path.join(dest + "\\" + os.path.basename(fileFullPath[0]) + ".png")
        plt.axis('off')
        plt.imsave(filename, img, cmap='gray')
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # filename = os.path.join(dest + "\\" +os.path.basename(fileFullPath[0]) + ".png")
        # cv2.imwrite(filename, gray_image)

