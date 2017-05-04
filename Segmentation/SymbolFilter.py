# This python file separate images of symbols from images of equations by their names.
# Currently only support png files.

import os
import shutil
from glob import glob

path = "./annotated"
path_list = glob(path + "/*.png")
dir1 = path+"/eqs"
dir2 = path+"/syms"

# check if pic is a symbol
def isEq(pic):
    index = pic.rfind('_')
    return pic[index+1:index+3] == 'eq'

if not os.path.exists(dir1):
    os.makedirs(dir1)

if not os.path.exists(dir2):
    os.makedirs(dir2)


for pic in path_list:
    # print(pic)

    if isEq(pic):
        shutil.copy2(pic, dir1)
    else:
        shutil.copy2(pic, dir2)
