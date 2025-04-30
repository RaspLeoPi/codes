# Self-implemented deep learning with MNIST

## Description

This small homework project implements neural network structure with MNIST from scratch. The special layers are linear layers and convolutional layers. 

## TODO

- [x] Complete the implementation of weight_visualization.py
- [x] Write the report
- [x] Implement new tests of CNN with fewer epoches
- [x] Check the load_model and save_model method of Model_CNN
- [x] Add usage of py files

## Structure

### dataset/MNIST

The dataset, including the training set and validation set. 

### draw_tools

Tools for plotting convergence history and visualizing parameters of neural networks. 

### training_history

The plottings of convergence history of the tests. 

### visualization

Visualization results of the trained model weights. 

### mynn

Core module of self-implemented neural network. Including: 
1. \_\_init\_\_.py: the initializer
2. op.py: defines layers of the network, including **linear layer**, **ReLU activation layer**, **Cross Entropy layer**, **convolutional layer**, **maxpooling layer** and a **flattening layer**. 
3. lr_scheduler.py: defines the learning rate scheduler of the training structure. 
4. metric.py
5. augment.py: does data augmentation for CNN tests. 
6. optimizer.py: defines SGD and MomentGD. 
7. runner.py: defines the trainer. 

### saved_models

The directory of saving trained models, saved as pickle files that can be loaded for testing. 

### best_models

The directory of saving trained models, saved as pickle files that can be loaded for testing. 

### test_models.py

The testing place for existing models. 

usage: test_model.py [-h] [--test TEST] [--type {MLP,CNN}] {single,multiple,best}

### test_train.py

The training place for models. Containing several tests, each with different network structure. 

usage: test_train.py [-h] [--test TEST]

options:
  -h, --help   show this help message and exit
  --test TEST  The index of the test function you want to run

### weight_visualization.py

The visualizer of weights of the trained models.

usage: weight_visualization.py [-h] {1,2}

positional arguments:
  {1,2}       The label of visualization - label 1: the visualization of the linear layer of MLP of test1 -
              lable 2: the visualization of the convolutional layer of CNN of test 7

### hyperparameter_search.py

Not implemented. 

---

### Start Up

First look into the `dataset_explore.ipynb` and get familiar with the data.

### Codes need your implementation

1. `op.py` 
   Implement the forward and backward function of `class Linear`
   Implement the `MultiCrossEntropyLoss`. Note that the `Softmax` layer could be included in the `MultiCrossEntropyLoss`.
   Try to implement `conv2D`, do not worry about the efficiency.
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py` You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py` You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.


### Train the model.

Open test_train.py, modify parameters and run it.

If you want to train the model on your own dataset, just change the values of variable *train_images_path* and *train_labels_path*

### Test the model.

Open test_model.py, specify the saved model's path and the test dataset's path, then run the script, the script will output the accuracy on the test dataset.
