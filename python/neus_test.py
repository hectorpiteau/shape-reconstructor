import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from math import *

draw_sdf = True
draw_sig = False
draw_phi = True
draw_ali = True
step = 0.05
xs = np.arange(0, 50, step)

def sdf(x):
    # surfaces = [15, 30]
    if(x < 15):
        return 10 - x
    if(x < 25):
        return x - 20
    if(x < 35):
        return 30 - x

    return x - 40

def sigmoid(x, s):
    return 1.0/(1 + exp(-s * x))

# logistic density distribution
def phi(x, s):
    return (s * exp(- s * x)) / (1 + exp(-s * x))**2


def alpha_i(x, s):
    return  max(
        (sigmoid(sdf(x), s) - sigmoid(sdf(x + step), s)) / (sigmoid(sdf(x), s)), 
        0
    )
    # return  sigmoid(sdf(x), s) * ((-phi(sdf(x), s)) / (sigmoid(sdf(x), s)))

def trans_i(x, s):
    res = 1
    for i in np.arange(0, x-step, step):
        res = res * (1 - alpha_i(i,s))
    # return sigmoid(sdf(x), s)
    return res



if(draw_sdf):
    y = np.array([sdf(x) for x in tqdm(xs)])
    plt.plot(xs, y/500, label="sdf")
    # plt.axvline(x = 10, color = 'b',  label = 'surface', linestyle='dashed')
    # plt.axvline(x = 30, color = 'b', linestyle='dashed')
 

if(draw_sig):
    ys = np.array([sigmoid(x, 1.0) for x in tqdm(xs)])
    plt.plot(xs, ys, label="sigmoid")

if(draw_phi):
    yp = np.array([phi(x, 1.0) for x in tqdm(xs)])
    # plt.plot(xs, yp, label="sigmoid'")

if(draw_ali):
    ya = np.array([(trans_i(x, 0.5) * alpha_i(x, 0.5)) for x in tqdm(xs)])
    plt.plot(xs, ya, label="weight")
    

plt.plot([0,50], [0,0], color="black")
plt.xlabel("t")

plt.legend()
plt.show()