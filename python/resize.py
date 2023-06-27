from os import listdir
from os.path import isfile, join
import cv2


src_path = "../data/nerf/train/"

dst_path100 = "../data/nerf100/train/"
dst_path200 = "../data/nerf200/train/"
dst_path400 = "../data/nerf400/train/"


files = [f for f in listdir(src_path) if isfile(join(src_path, f))]

for file in files:
    img=cv2.imread(join(src_path, file), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (100, 100))
    cv2.imwrite(join(dst_path100,file), img)

for file in files:
    img=cv2.imread(join(src_path, file), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (200, 200))
    cv2.imwrite(join(dst_path200,file), img)

for file in files:
    img=cv2.imread(join(src_path, file), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (400, 400))
    cv2.imwrite(join(dst_path400,file), img)
