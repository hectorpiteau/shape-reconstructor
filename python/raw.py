import rawpy
import imageio
import numpy as np
import matplotlib.pyplot as plt

path = '/home/hpiteau/data/iphone/flower1/IMG_3254.DNG'
raw = rawpy.imread(path)
print(raw.raw_image)
print(raw.raw_image.shape)

data = np.delete(raw.raw_image, -1, axis=2)
ndata = (data - np.min(data)) / (np.max(data) - np.min(data))
ndata = 10 * np.exp(-ndata)  
print(ndata)
print(ndata.shape)
print("min: ", np.min(ndata))
print("max: ", np.max(ndata))
# rgb = raw.postprocess()
# imageio.imsave('rawtest.png', rgb)

plt.imshow(ndata)
plt.show()