from rawpy import *
import imageio
import numpy as np
import matplotlib.pyplot as plt

path = '/home/hpiteau/data/iphone/flower1/IMG_3254.DNG'
raw = rawpy.imread(path)

d = rawpy.RawPy()
d.open_file(path)
d.unpack()
print(raw.raw_image)
bloub = raw.raw_image[::2,::2]
print(raw.raw_image.shape)


# plt.plot(list(range(0, 65536)), raw.tone_curve)
# plt.show()

data = np.delete(raw.raw_image, -1, axis=2)
# ndata = (data - np.min(data)) / (np.max(data) - np.min(data))
# ndata = ndata/16.0 +  ndata * np.exp(ndata * -1)  + 0.5
ndata = data/(data + 16000) 


# for res in range(20,100):

#     ndata = data/(10**(res* 0.1) + data)
#     print(res)
#     # print(ndata)
#     # print(ndata.shape)
#     # print("min: ", np.min(ndata))
#     # print("max: ", np.max(ndata))
#     # rgb = raw.postprocess()
#     # imageio.imsave('rawtest.png', rgb)
      
print(ndata)
print(ndata.shape)
print("min: ", np.min(ndata))
print("max: ", np.max(ndata))
print("original min: ", np.min(data))
print("original max: ", np.max(data))
plt.imshow(ndata)
plt.show()