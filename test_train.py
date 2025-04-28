# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import argparse

# fixed seed for experiment
np.random.seed(300)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()


# best accuracy performance: 0.93590
def test1():
        """
        layers: linear->ReLU->linear->Softmax->CrossEntropy
        dimension: 784->600->10
        optimizer: SGD
        ...

        """
        print("=== test 1 is on the way ===")
        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train(
                [train_imgs, train_labs], 
                [valid_imgs, valid_labs], 
                num_epochs=5, 
                log_iters=100, 
                save_dir=r'./best_models', 
                model_name='test1'
        )

        print(f"Final best score: {runner.best_score}")

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

# best accuracy performance: 0.93840
def test2():
        """
        layers: linear->ReLU->linear->Softmax->CrossEntropy
        dimension: 784->600->10
        optimizer: MomentGD
        ...

        """
        print("=== test 2 is on the way ===")
        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train(
                [train_imgs, train_labs], 
                [valid_imgs, valid_labs], 
                num_epochs=5, 
                log_iters=100, 
                save_dir=r'./best_models', 
                model_name='test2'
        )

        print(f"Final best score: {runner.best_score}")

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

# best accuracy performance: 0.9348
def test3():
        """
        change the number of neurons to 480
        layers: linear->ReLU->linear->Softmax->CrossEntropy
        dimension: 784->480->10
        optimizer: SGD
        ...

        """
        print("=== test 3 is on the way ===")
        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 480, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train(
                [train_imgs, train_labs], 
                [valid_imgs, valid_labs], 
                num_epochs=5, 
                log_iters=100, 
                save_dir=r'./best_models', 
                model_name='test3'
        )

        print(f"Final best score: {runner.best_score}")

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()        
        pass

# best accuracy performance: 
def test4():
        """
        use l2 regularization (weight decay)
        layers: linear->ReLU->linear->Softmax->CrossEntropy
        dimension: 784->600->10
        optimizer: SGD
        ...

        """
        print("=== test 4 is on the way ===")
        linear_model = nn.models.Model_MLP(
                [train_imgs.shape[-1], 600, 10], 
                'ReLU', 
                [1e-4, 1e-4], 
                weight_decay=True, 
                weight_decay_lambda=1e-7        # original value: 1e-8 with no improvement from the original example
        )
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train(
                [train_imgs, train_labs], 
                [valid_imgs, valid_labs], 
                num_epochs=5, 
                log_iters=100, 
                save_dir=r'./best_models', 
                model_name='test4'
        )

        print(f"Final best score: {runner.best_score}")

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()
        pass

def test5():
        """
        implementation of CNN
        layers: conv1->ReLU->maxpool->conv2->ReLU->maxpool->Flatten->linear->Softmax->CrossEntropy
        dimension: 28*28*1->26*26*32->13*13*32->11*11*64->6*6*64->10
        optimizer: SGD
        ...
        """
        print("=== test 5 is on the way ===")
        # Reshape input from (batch, 784) to (batch, 1, 28, 28)
        train_imgs_cnn = train_imgs.reshape(-1, 1, 28, 28)
        valid_imgs_cnn = valid_imgs.reshape(-1, 1, 28, 28)
        
        # Define CNN architecture with pooling
        conv_params = [
            {
                'in_channels': 1,
                'out_channels': 16,
                'kernel_size': 3,
                'pool_params': {'pool_size': 2, 'stride': 2}
            },
            {
                'in_channels': 16,
                'out_channels': 32, 
                'kernel_size': 3,
                'pool_params': {'pool_size': 2, 'stride': 2}
            }
        ]
        
        cnn_model = nn.models.Model_CNN(
            conv_params=conv_params,
            num_classes=10,
            act_func='ReLU'
        )
        
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[800, 2400, 4000],
            gamma=0.5
        )
        loss_fn = nn.op.MultiCrossEntropyLoss(
            model=cnn_model,
            max_classes=train_labs.max()+1
        )

        runner = nn.runner.RunnerM(
            cnn_model,
            optimizer,
            nn.metric.accuracy,
            loss_fn,
            scheduler=scheduler
        )

        runner.train(
            [train_imgs_cnn, train_labs],
            [valid_imgs_cnn, valid_labs],
            num_epochs=5,
            log_iters=300,              # dev evaluation for every 300 iterations
            save_dir=r'./best_models',
            model_name='test5', 
            detail_eval=False
        )

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

def test6():
        """
        implementation of CNN with data augmentation
        layers: conv1->maxpool->conv2->maxpool->linear->Softmax->CrossEntropy
        dimension: 28*28*1->26*26*32->13*13*32->11*11*64->6*6*64->10
        optimizer: SGD
        ...

        """
        print("=== test 6 is on the way ===")
        # data augmentation first
        aug_train_imgs = np.array([nn.augment.augment_image(train_imgs[i,]) for i in range(train_imgs.shape[0])])
        # aug_train_imgs = np.concatenate((train_imgs, aug_train_imgs), axis=0)
        # train_labs = np.concatenate((train_labs, train_labs), axis=0)
        # Reshape input from (batch, 784) to (batch, 1, 28, 28)
        train_imgs_cnn = aug_train_imgs.reshape(-1, 1, 28, 28)
        valid_imgs_cnn = valid_imgs.reshape(-1, 1, 28, 28)

        # Define CNN architecture with pooling
        conv_params = [
            {
                'in_channels': 1,
                'out_channels': 16,
                'kernel_size': 3,
                'pool_params': {'pool_size': 2, 'stride': 2}
            },
            {
                'in_channels': 16,
                'out_channels': 32, 
                'kernel_size': 3,
                'pool_params': {'pool_size': 2, 'stride': 2}
            }
        ]
        
        cnn_model = nn.models.Model_CNN(
            conv_params=conv_params,
            num_classes=10,
            act_func='ReLU'
        )
        
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[800, 2400, 4000],
            gamma=0.5
        )
        loss_fn = nn.op.MultiCrossEntropyLoss(
            model=cnn_model,
            max_classes=train_labs.max()+1
        )

        runner = nn.runner.RunnerM(
            cnn_model,
            optimizer,
            nn.metric.accuracy,
            loss_fn,
            scheduler=scheduler
        )

        runner.train(
            [train_imgs_cnn, train_labs],
            [valid_imgs_cnn, valid_labs],
            num_epochs=5,
            log_iters=300,              # dev evaluation for every 300 iterations
            save_dir=r'./best_models',
            model_name='test6'
        )

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

test_mapping = {
        "test1": test1, 
        "test2": test2, 
        "test3": test3, 
        "test4": test4,
        "test5": test5, 
        "test6": test6
}

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="test1", help="The name of the test function you want to run")

args = parser.parse_args()

if args.test in test_mapping:
        test_mapping[args.test]()
else:
        print(f"Function '{args.test}' not found.")