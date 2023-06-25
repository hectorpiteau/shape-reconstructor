from os import listdir
from os.path import isfile, join
import cv2


src_path = "../data/nerf/train/"

dst_path = "../data/nerf100/train/"


files = [f for f in listdir(src_path) if isfile(join(src_path, f))]

for file in files:
    img=cv2.imread(join(src_path, file))
    img = cv2.resize(img, (100, 100))
    cv2.imwrite(join(dst_path,file), img)
