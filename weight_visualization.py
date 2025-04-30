# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import argparse
import os

def vis1():
    # visualize the two linear layers of MLP
    model = nn.models.Model_MLP()
    model.load_model("./saved_models/test1.pickle")

    """
    The first layer: 784 * 600
    Visualize it as an image of resolution 784 * 600
    """
    plt.matshow(model.layers[0].params['W'])
    plt.axis('off')
    plt.colorbar()
    plt.savefig("./visualization/vis1_1.png")
    plt.close()
    """
    The second layer: 600 * 10
    Visualize it as an image of resolution 600 * 10
    """
    plt.matshow(model.layers[2].params['W'])
    plt.axis('off')
    plt.colorbar()
    plt.savefig("./visualization/vis1_2.png")
    plt.close()

    print("Visualization done! Check ./visualization/vis1_1.png and ./visualization/vis1_2.png")

def vis2():
    # visualize the two convolutional layers of CNN
    model = nn.models.Model_CNN()
    model.load_model("./saved_models/test7.pickle")

    # kernels = []
    # kernels.append(model.layers[0].params['W'])
    # kernels.append(model.layers[3].params['W'])

    """
    The first kernel: 16 * 1 * 5 * 5
    For each out_channel from 1 to 16, 
    visualize it as an image of resolution 5 * 5
    """
    kernel1 = model.layers[0].params['W']
    kernel1 = kernel1.squeeze()
    # plt.matshow(kernel1[0])
    # plt.show()
    # print(kernel1.shape)
    fig = plt.figure(figsize=(6,6))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        im = ax.matshow(kernel1[i])
        ax.axis('off')
        plt.colorbar(im)
    plt.savefig("./visualization/vis2_1.png")
    plt.close()

    """
    The second kernel: 32 * 16 * 5 * 5
    Compute the mean along the second axis (input channel)
    For each out_channel from 1 to 32, 
    visualize it as an image of resolution 5 * 5
    """
    kernel2 = model.layers[3].params['W']
    kernel2 = np.mean(kernel2, axis=1)
    kernel1 = kernel1.squeeze()
    fig = plt.figure(figsize=(12, 6))
    for i in range(32):
        ax = fig.add_subplot(4, 8, i + 1)
        im = ax.matshow(kernel2[i])
        ax.axis('off')
        plt.colorbar(im)
    plt.savefig("./visualization/vis2_2.png")
    plt.close()

    print("Visualization done! Check ./visualization/vis2_1.png and ./visualization/vis2_2.png")

save_path = "./visualization"

if not os.path.exists(save_path):
    os.mkdir(save_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "label", 
    choices=["1", "2"], 
    help="""
        The label of visualization
        - label 1: the visualization of the linear layer of MLP of test1
        - lable 2: the visualization of the convolutional layer of CNN of test 7
        """
)
args = parser.parse_args()

fun_dict = {
    "1": vis1, 
    "2": vis2
}

# run the visualization
fun_dict[args.label]()