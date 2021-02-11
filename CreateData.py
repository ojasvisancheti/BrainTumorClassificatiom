import os
import shutil
import glob
import random
source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"Train"
TEST_PATH = r"Test"
VALID_PATH = r"Validation"
Label1_PATH = r"D:\MSC\Sem2\IVC\ClassificationTask\catdog\catdog\DOGS"
CAT_PATH = r"D:\MSC\Sem2\IVC\ClassificationTask\catdog\catdog\CATS"

path, dirs, files = next(os.walk(DOG_PATH))
file_count = len(files)
to_be_moved = random.sample(glob.glob(DOG_PATH + "/*.png"), file_count)
if os.path.exists(TRAIN_PATH):
    shutil.rmtree(TRAIN_PATH)
if os.path.exists(TEST_PATH):
    shutil.rmtree(TEST_PATH)
if os.path.exists(VALID_PATH):
    shutil.rmtree(VALID_PATH)
n = 1
v = 1
for f in enumerate(to_be_moved, 1):
    if n > 100:
        if v > 100:
            dest = os.path.join(TRAIN_PATH +"\\dog")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        else:
            dest = os.path.join(VALID_PATH+"\\dog")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        v = v + 1
    else:
        dest = os.path.join(TEST_PATH+"\\dog")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)
    n = n + 1

path, dirs, files = next(os.walk(CAT_PATH))
file_count = len(files)
n = 1
v = 1
to_be_moved = random.sample(glob.glob(CAT_PATH + "/*.png"), file_count)
for f in enumerate(to_be_moved, 1):
    if n > 100:
        if v > 100:
            dest = os.path.join(TRAIN_PATH+"\\cat")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        else:
            dest = os.path.join(VALID_PATH+"\\cat")
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)
        v = v+1
    else:
        dest = os.path.join(TEST_PATH+"\\cat")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)
    n = n + 1

