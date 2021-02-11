import os
import shutil
import glob
import random
source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"Train"
TEST_PATH = r"Test"
VALID_PATH = r"Validation"
Label1_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label1"
Label2_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label2"
Label3_PATH = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\Label3"

path, dirs, files = next(os.walk(Label1_PATH))
file_count = len(files)
to_be_moved = random.sample(glob.glob(Label1_PATH + "/*.png"), file_count)
if os.path.exists(TRAIN_PATH):
    shutil.rmtree(TRAIN_PATH)
if os.path.exists(TEST_PATH):
    shutil.rmtree(TEST_PATH)
if os.path.exists(VALID_PATH):
    shutil.rmtree(VALID_PATH)
n = 1
v = 1
print(file_count * 0.1)
print((file_count - file_count * 0.1) * 0.1)
for f in enumerate(to_be_moved, 1):
    if n > file_count * 0.1:
        if v > (file_count - file_count * 0.1) * 0.1:
            dest = os.path.join(TRAIN_PATH +"\\label1")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        else:
            dest = os.path.join(VALID_PATH+"\\label1")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        v = v + 1
    else:
        dest = os.path.join(TEST_PATH+"\\label1")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)
    n = n + 1

path, dirs, files = next(os.walk(Label2_PATH))
file_count = len(files)
n = 1
v = 1
print(file_count * 0.1)
print((file_count - file_count * 0.1) * 0.1)
to_be_moved = random.sample(glob.glob(Label2_PATH + "/*.png"), file_count)
for f in enumerate(to_be_moved, 1):
    if n > file_count * 0.1:
        if v > (file_count - file_count * 0.1) * 0.1:
            dest = os.path.join(TRAIN_PATH+"\\label2")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        else:
            dest = os.path.join(VALID_PATH+"\\label2")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        v = v+1
    else:
        dest = os.path.join(TEST_PATH+"\\label2")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)
    n = n + 1

path, dirs, files = next(os.walk(Label3_PATH))
file_count = len(files)
n = 1
v = 1
print(file_count * 0.1)
print((file_count - file_count * 0.1) * 0.1)
to_be_moved = random.sample(glob.glob(Label3_PATH + "/*.png"), file_count)
for f in enumerate(to_be_moved, 1):
    if n > file_count * 0.1:
        if v > (file_count - file_count * 0.1) * 0.1:
            dest = os.path.join(TRAIN_PATH+"\\label3")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        else:
            dest = os.path.join(VALID_PATH+"\\label3")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        v = v+1
    else:
        dest = os.path.join(TEST_PATH+"\\label3")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)
    n = n + 1

