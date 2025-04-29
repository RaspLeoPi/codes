# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="1", help="The index of the model you want to visualize")
parser.add_argument("--model", default="linear", choices=["linear", "conv"], 
                    help="specify the kind of model to visualize")

args = parser.parse_args()

model = nn.models.Model_MLP()
model.load_model(f'.\\saved_models\\test{args.test}.pickle')

# TODO: complete the visualization for linear layer and conv layer

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])

plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])
plt.show()