from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import cv2


src_path = "../data/nerf/train/r_0.png"

img=cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
image_array = np.array(img)

width = 800
height = 800
tile_size = 8

nb_tiles_x = int(np.ceil(width / tile_size))
nb_tiles_y = int(np.ceil(height / tile_size))

nb_tiles_tot = nb_tiles_x * nb_tiles_y 
nb_tiles_used = 0

for x in range(0, nb_tiles_x):
    for y in range(0, nb_tiles_y):
        # loop in tile #
        is_tile_used = False
        for i in range(0, tile_size):
            for j in range(0, tile_size):
                x_index = x * tile_size + i
                y_index = y * tile_size + j
                if (image_array[x_index, y_index, 3] != 0):
                    is_tile_used = True
                    nb_tiles_used += 1 
                    # image_array[x, y] = [255, 0, 0, 255]
                    break
            if is_tile_used : 
                break
                
print("nb_tiles_x: ", nb_tiles_x)
print("nb_tiles_y: ", nb_tiles_y)
print("nb_tiles_tot: ", nb_tiles_tot)
print("nb_tiles_used: ", nb_tiles_used)
# cv2.imwrite(join(dst_path100,file), img)

plt.imshow(image_array)
plt.show()