import os
import shutil
import glob
import random
source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"\\Train"
TEST_PATH = r"\\Test"
VALID_PATH = r"\\Validation"
data_dir = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData"

import os, shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

## setPath1 is the path of SETA, SETB, or SET C
## according to setPath1, setPath2 is the path of SETB, SETC, or SET D
## split define after combining two set passed how much part you want as a validation
# Set we have kept it as 0.25
def CreateTrainValidationSliptFolder(setPath1, setPath2, split):
    if os.path.exists(data_dir + TRAIN_PATH):
        shutil.rmtree(data_dir + TRAIN_PATH)
    if os.path.exists(data_dir +TEST_PATH):
        shutil.rmtree(data_dir +TEST_PATH)
    if os.path.exists(data_dir +VALID_PATH):
        shutil.rmtree(data_dir +VALID_PATH)
    path, dirs, files = next(os.walk(setPath1))
    for directory in enumerate(dirs):
        path, direc, files1 = next(os.walk(setPath1 + '\\' + directory[1]))
        file_count1 = len(files1)
        to_be_moved = random.sample(glob.glob(setPath1 + '\\' + directory[1] + "/*.png"), file_count1)
        path, direc, files2 = next(os.walk(setPath2 + '\\' + directory[1]))
        file_count2 = len(files2)
        to_be_moved = to_be_moved + random.sample(glob.glob(setPath2 + '\\' + directory[1] + "/*.png"), file_count2)
        n = 1
        v = 1
        for f in enumerate(to_be_moved, 1):
            if n > file_count1 * split:
                dest = os.path.join(data_dir +TRAIN_PATH +"\\" + directory[1])
                if not os.path.exists(dest):
                    os.makedirs(dest)
                shutil.copy(f[1], dest)
            else:
                dest = os.path.join(data_dir +VALID_PATH+ "\\" + directory[1])
                if not os.path.exists(dest):
                    os.makedirs(dest)
                shutil.copy(f[1], dest)
            n = n + 1
    print("Done")

# According to Setpath CreateTrainValidationSliptFolder testPath a path of Set C, Set A, Set B
def CreateTestFolder(testPath):
    if os.path.exists(data_dir +TEST_PATH):
        shutil.rmtree(data_dir +TEST_PATH)
    copytree(testPath, data_dir + TEST_PATH)
    print("Done")









