import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import argparse

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

def test():
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", default="9", help="Choose the model from test")
        parser.add_argument("--type", default="CNN", choices=["MLP", "CNN"], help="MLP or CNN")

        args = parser.parse_args()

        if args.type == "MLP":
                model = nn.models.Model_MLP()
        else:
                model = nn.models.Model_CNN()
                test_imgs = test_imgs.reshape(num, 1, 28, 28)              # reshape test images
        model.load_model(f'.\\saved_models\\test{args.test}.pickle')

        logits = model(test_imgs)
        print(f"The accuracy of the model in test{args.test}: {nn.metric.accuracy(logits, test_labs)}")

def best_test():
        global test_imgs
        # initialization
        accuracy_dict = {}
        # four MLP models
        for i in range(4):
                name = "test" + str(i + 1)
                model = nn.models.Model_MLP()
                model.load_model(f'.\\saved_models\\{name}.pickle')
                
                logits = model(test_imgs)
                accuracy_dict[name] = nn.metric.accuracy(logits, test_labs)
                print(f"The accuracy of the model in {name}: {accuracy_dict[name]}")
        # five CNN models
        for i in range(4, 12):
                name = "test" + str(i + 1)
                model = nn.models.Model_CNN()
                model.load_model(f'.\\saved_models\\{name}.pickle')
                test_imgs = test_imgs.reshape(num, 1, 28, 28)              # reshape test images

                logits = model(test_imgs)
                accuracy_dict[name] = nn.metric.accuracy(logits, test_labs)
                print(f"The accuracy of the model in {name}: {accuracy_dict[name]}")
        
        best_test = max(accuracy_dict, key=accuracy_dict.get)
        print(f"The test that produces the best model: {best_test}")
        print(f"The accuracy of the best model: {accuracy_dict[best_test]}")

        # save the best model: copy and rename
        with open(f'.\\saved_models\\{best_test}.pickle', 'rb') as src:
                data = src.read()
                with open(f'.\\best_models\\best_model.pickle', 'wb') as dest: 
                        dest.write(data)

def test_best_model():
        global test_imgs
        model = nn.models.Model_MLP()
        try: 
                model.load_model(f'.\\best_models\\best_model.pickle')
        except:
                # use CNN instead
                model = nn.models.Model_CNN()
                model.load_model(f'.\\best_models\\best_model.pickle')
                test_imgs = test_imgs.reshape(num, 1, 28, 28)
        finally: 
                logits = model(test_imgs)
                print(f"The accuracy of the best model: {nn.metric.accuracy(logits, test_labs)}")

parser = argparse.ArgumentParser()
parser.add_argument("type", choices=["single", "multiple", "best"])
parser.add_argument("--test", default="9", help="Choose the model from test")
parser.add_argument("--type", default="CNN", choices=["MLP", "CNN"], help="MLP or CNN")
args = parser.parse_args()

if args.type == "single": 
        if args.type == "MLP":
                model = nn.models.Model_MLP()
        else:
                model = nn.models.Model_CNN()
                test_imgs = test_imgs.reshape(num, 1, 28, 28)              # reshape test images
        model.load_model(f'.\\saved_models\\test{args.test}.pickle')

        logits = model(test_imgs)
        print(f"The accuracy of the model in test{args.test}: {nn.metric.accuracy(logits, test_labs)}")
elif args.type == "multiple":
        best_test()
else:
        test_best_model()