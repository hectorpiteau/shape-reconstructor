from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import cv2
import numpy as np
import torch


def noisify(image, noise_typ = "gauss"):
    print(type(image))
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 255*2
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        print("noise: ", gauss[400][400])
        noisy = (image + gauss)
        
        noisy[noisy<0] = 0
        noisy[noisy>255] = 255
        noisy[image<=0] = 0

        return np.uint8(noisy)
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

paths = ["../data/nerf/train/r_0.png", 
         "../data/nerf/train/r_1.png",
         "../data/nerf/train/r_2.png",
         "../data/nerf/train/r_3.png"]

for path in tqdm(paths):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # cv2.imshow('Original Image', image)
    # cv2.waitKey(0)
    
    # Gaussian Blur
    Gaussian = cv2.GaussianBlur(image, (11, 11), 0)
    # cv2.imshow('Gaussian Blurring', Gaussian)
    # cv2.waitKey(0)

    Noisy = noisify(Gaussian)
    # cv2.imshow('Gaussian Blurring + Noisy', Noisy)
    # cv2.waitKey(0)
    # Noisy = cv2.resize(Noisy, (100,100))
    Noisy = cv2.resize(image, (200,200))
    cv2.imwrite(path+".200.png", Noisy)


