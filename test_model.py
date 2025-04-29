import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="9", help="Choose the model from test")
parser.add_argument("--type", default="CNN", choices=["MLP", "CNN"], help="MLP or CNN")

args = parser.parse_args()

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

if args.type == "MLP":
        model = nn.models.Model_MLP()
else:
        model = nn.models.Model_CNN()
        test_imgs = test_imgs.reshape(num, 1, 28, 28)              # reshape test images
model.load_model(f'.\\saved_models\\test{args.test}.pickle')

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))