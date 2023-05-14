from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

images = []
for index in range(0, 20):
    path = "data/bg/"+ "img-{:06d}.png".format(index)
    print("load image: ", path)
    image = Image.open(path)

    # Convert the image to a Numpy array
    image_array = np.array(image)
    images.append(image_array)

    # Print the shape of the array
    print(image_array.shape)


    # plt.imshow(image_array)
    # plt.show()


images = np.array(images)

mean = np.mean(images, axis=0)
print("images shape: ", images.shape)
print("mean shape: ", mean.shape)

plt.imshow(mean)
plt.show()