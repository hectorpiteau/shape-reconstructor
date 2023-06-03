from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

images = []
for index in range(0, 20):
    path = "data/bg/" + "img-{:06d}.png".format(index)
    # path = "data/masked/"+ "img-{:06d}.png".format(90+index)
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
images_b = images[:, 0::2, 1::2]
images_r = images[:, 1::2, 0::2]
images_g1 = images[:, 0::2, 0::2]
images_g2 = images[:, 1::2, 1::2]


# print("images_b shape: ", images_b.shape)
# img_rgb = np.stack([images_r, (images_g2), images_b], axis=-1)
# plt.imsave("rgb.png", img_rgb)
# plt.imshow(img_rgb)
# plt.show()

mean = np.mean(images, axis=0)
print("images shape: ", images.shape)
print("mean shape: ", mean.shape)

# plt.imshow(mean)
# plt.show()


# fig = plt.figure()
# # ax = fig.add_axes([0,0,1,1])
# xs = list(range(0,256))
# ys = np.array([0 for i in xs])

# for index in range(0, 20):
#     ys[images[index][0][0]] += 1

# ys = ys / 20
# plt.yscale("log")
# plt.bar(xs, ys)
# plt.show()

# Compute standard deviation per color
print("images_b shape: ", images_b.shape)
print("images_g1 shape: ", images_g1.shape)
print("images_r shape: ", images_r.shape)

# images_b_mean = np.mean(images_b, axis=0)
# images_r_mean = np.mean(images_r, axis=0)
# images_g1_mean = np.mean(images_g1, axis=0)
# images_g2_mean = np.mean(images_g2, axis=0)

# images_b_flat = np.stack([
#     images_b[i, :, :] - images_b_mean for i in range(0,20)
# ], axis=-1)
# images_g1_flat = np.stack([
#     images_g1[i, :, :] - images_g1_mean for i in range(0,20)
# ], axis=-1)

# plt.imshow((images_g1_flat[:,:,0]))
# plt.show()

# print("images_b_flat shape: ", images_b_flat.shape)

# images_b_std = np.std(images_b, axis=0)
# images_r_std = np.std(images_r, axis=0)
# images_g1_std = np.std(images_g1, axis=0)
# images_g2_std = np.std(images_g2, axis=0)

# print("images_b_std shape: ", images_b_std.shape)
# print("images_r_std shape: ", images_r_std.shape)
# print("images_g1_std shape: ", images_g1_std.shape)
# print("images_g2_std shape: ", images_g2_std.shape)
# plt.imshow(images_g1_std)
# plt.show()


def compute_stds(images, filename="stds", plot=True, color="blue", save=True):
    print("compute standard deviation with respect to pixel intensity.")
    print("images shape: ", images.shape)

    xs = np.array(range(0, 256))
    ys = [[] for x in xs]
    counts = [0 for x in xs]

    images_mean = np.mean(images, axis=0)
    if (plot):
        plt.imshow(images_mean)
        plt.show()

    for i in tqdm(range(0, images.shape[0])):
        for y in tqdm(range(0, images.shape[1])):
            for x in range(0, images.shape[2]):
                ys[int(round(images_mean[y, x]))].append(images[i, y, x] / images_mean[y, x])
                counts[int(round(images_mean[y, x]))] += 1

    yys = np.array([0 for x in xs], dtype=float)
    yys_div_x = np.array([0 for x in xs], dtype=float)

    for i in tqdm(range(0, 256)):
        if len(ys[i]) == 0:
            yys[i] = 0
            continue
        yys[i] = np.std(np.array(ys[i], dtype=float))

    for i in tqdm(range(1, 256)):
        yys_div_x = yys / float(i)

    print("=== saving: ", str(save), " === : ")
    if (save):
        np.save(filename+"_xs", xs)
        np.save(filename+"_ys", yys)
        np.save(filename+"_ysdx", yys_div_x)
        np.save(filename+"_cs", counts)

    if (plot):
        plt.plot(xs, yys, color=color)
        plt.show()
        plt.plot(xs, yys_div_x, color=color)
        plt.show()
    return yys, yys_div_x


compute = True
if (compute):
    stdb, stdbd = compute_stds(
        images_b, filename="exp1/blue_stds", color="blue")
    stdr, stdrd = compute_stds(images_r, filename="exp1/red_stds", color="red")
    stdg1, stdg1d = compute_stds(
        images_g1, filename="exp1/green1_stds", color="green")
    stdg2, stdg2d = compute_stds(
        images_g2, filename="exp1/green2_stds", color="green")
else:
    stdb = np.load("exp1/blue_stds_ys.npy")
    stdbd = np.load("exp1/blue_stds_ysdx.npy")
    stdr = np.load("exp1/red_stds_ys.npy")
    stdrd = np.load("exp1/red_stds_ysdx.npy")
    stdg1 = np.load("exp1/green1_stds_ys.npy")
    stdg1d = np.load("exp1/green1_stds_ysdx.npy")
    stdg2 = np.load("exp1/green2_stds_ys.npy")
    stdg2d = np.load("exp1/green2_stds_ysdx.npy")

xs = np.array(range(0, 256))
xsbar = np.arange(0, 256/4)

# plt.bar(xsbar, stdb.reshape(-1, 4).mean(axis=1))
# plt.xticks(xsbar, [f"{i*4}" for i in xsbar])
# plt.show()
# plt.plot(xs, stdb, color="blue")
# plt.plot(xs, stdr, color="red")
# plt.plot(xs, stdg1, color="green")
# plt.plot(xs, stdg2, color="green")

# plt.show()
plt.plot(xs, stdbd, color="blue")
plt.plot(xs, stdrd, color="red")
plt.plot(xs, stdg1d, color="green")
plt.plot(xs, stdg2d, color="green")
plt.show()
# Compute standard deviation per pixel


# create some example data
# xs = np.arange(0, 256, 1)
# y = np.random.normal(0, 10, size=256)
# plt.plot(xs, y)
# plt.show()
# # reshape the array into a 2D array with 5 columns, and sum along the columns
# y_grouped = y.reshape(-1, 8).mean(axis=1)

# # create x values for the bar chart
# x = np.arange(0, len(y), 8)

# # plot the bar chart
# plt.bar(x, y_grouped)

# # set the x-axis labels
# plt.xticks(x, [f"{i}-{i+7}" for i in x])

# # set the axis labels and title
# plt.xlabel("Value ranges")
# plt.ylabel("Sum of values")
# plt.title("Sum of values grouped by 8")

# # show the plot
# plt.show()

exit(0)


stds = np.zeros((2048, 2048))


def proc_std(i):
    global stds
    global images
    for j in range(0, 2048):
        xs = list(range(0, 256))
        ys = np.array([0 for i in xs])
        # count pixel occurrences for pixel (i,j)
        for index in range(0, 20):
            ys[images[index][i][j]] += 1
        # normalize ys
        ys = ys / 20.0
        # fit gaussian
        mu, std = norm.fit(ys)
        stds[i][j] = std
    print("== finished i:", i, " ==")

# for i in tqdm(range(0, 2048)):
#     for j in tqdm(range(0, 2048)):
#         proc_std(i,j)

# pool_obj = multiprocessing.Pool()
# ans = pool_obj.map(proc_std,range(0,2048))
# pool_obj.close()


stds = np.std(images, axis=0)

np.save("standards", stds)

print(stds)
# print(stds)
ax = plt.subplot()
im = plt.imshow(stds, cmap="plasma")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
plt.show()
