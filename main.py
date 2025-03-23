import math
import os
import sys

import torchmetrics.classification
import torchmetrics.functional.classification
import torchmetrics.functional.classification.precision_recall
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2

import matplotlib.pyplot as plot

# import torch_directml
import torch
import torch.nn as nn

import torchmetrics

import datetime

from sbls import SBLS

# os.system('clear')

torch.set_printoptions(profile="full")

# dml = torch_directml.device()

ITERATIONS = 10

def name(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]

def show_img(img):
    img = np.array(img, dtype='float')
    # pixels = img.reshape((28, 28))
    plot.imshow(img, cmap='gray', aspect="auto", interpolation='none')
    plot.show()

def load_mnist(batch_size_train: int, batch_size_test: int) -> tuple[DataLoader, DataLoader]:
    data_path = '/tmp/data/mnist'

    transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return DataLoader(mnist_train, batch_size=batch_size_train, shuffle=True, drop_last=True), DataLoader(mnist_test, batch_size=batch_size_test, shuffle=True, drop_last=True)

def test(test_loader, sbls, metric_list: list[torchmetrics.Metric]):
    test_input, test_target = next(iter(test_loader))

    test_prediction = sbls(test_input)

    for metric in metric_list:
        metric.update(test_prediction, test_target)

# bcause I don't feel like researching how to implement a custom torch.Dataloader
def test2(test_data: tuple[torch.Tensor, torch.Tensor], sbls, metric_list: list[torchmetrics.Metric]):
    test_input, test_target = test_data

    test_prediction = sbls(test_input)

    # print("PREDICTION")
    # print(test_prediction[:10])

    for metric in metric_list:
        metric.update(test_prediction, test_target)

# this experiment shows the influence of non-trained networks vs those trained with different types of parameters
def experiment_1(feature_size: int, enhancement_windows: int):
    accuracy_pre = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    accuracy_post = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    confusion_matrix_pre = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true")
    confusion_matrix_post = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true")

    
    for i in range(ITERATIONS):
        print(f"\niteration {i}")
        train_loader, test_loader = load_mnist(60000, 10000)
        sbls = SBLS(input_size=28*28, output_size=10, simulation_steps=30, initital_feature_size=feature_size, enhancement_nodes_per_window=100, initial_enhancement_window_num=enhancement_windows, output_is_one_hot=True)

        test(test_loader, sbls, [accuracy_pre, confusion_matrix_pre])

        print("Training network...")
        data = next(iter(train_loader))
        sbls.add_new_data(data)
        print(f"Training complete!")

        test(test_loader, sbls, [accuracy_post, confusion_matrix_post])

    print(f"Results of experiment 1 with params feature_size: {feature_size}, enhancement_windows: {enhancement_windows}")

    print(f"accuracy_pre: {accuracy_pre.compute()}")
    print(f"accuracy_post: {accuracy_post.compute()}")
    fig_pre, _ = confusion_matrix_pre.plot()
    fig_post, _ = confusion_matrix_post.plot()
    fig_pre.set_size_inches(7.0, 7.0)
    fig_post.set_size_inches(7.0, 7.0)

    plot.show()

# this experiment shows the influence of how much trainig data is available
def experiment_2(train_size: int):
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true")

    for i in range(ITERATIONS):
        print(f"iteration {i}")
        train_loader, test_loader = load_mnist(train_size, 10000)
        sbls = SBLS(input_size=28*28, output_size=10, simulation_steps=30, initital_feature_size=100, enhancement_nodes_per_window=100, initial_enhancement_window_num=20, output_is_one_hot=True)

        print("Training network...")
        data = next(iter(train_loader))
        sbls.add_new_data(data)
        print(f"Training complete!")

        test(test_loader, sbls, [accuracy, confusion_matrix])

    print(f"Results of experiment 2 with params train_size: {train_size}")
    print(f"accuracy: {accuracy.compute()}")

    fig, _ = confusion_matrix.plot()
    fig.set_size_inches(7.0, 7.0)

    plot.savefig(f"images/ex2_{train_size}.png")


# this experiments shows the effect of incrementally adding enhancement nodes
def experiment_3(initial_windows: int, windows_to_add: list[int]):
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true")
    accuracy_add_list = []
    confusion_matrix_add_list = []
    for elem in windows_to_add:
        accuracy_add_list.append(torchmetrics.classification.MulticlassAccuracy(num_classes=10))
        confusion_matrix_add_list.append(torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true"))

    for i in range(ITERATIONS):
        print(f"iteration {i}")
        train_loader, test_loader = load_mnist(60000, 10000)
        sbls = SBLS(input_size=28*28, output_size=10, simulation_steps=30, initital_feature_size=100, enhancement_nodes_per_window=100, initial_enhancement_window_num=initial_windows, output_is_one_hot=True)

        print("Training network...")
        data = next(iter(train_loader))
        sbls.add_new_data(data)
        print(f"Training complete!")

        test(test_loader, sbls, [accuracy, confusion_matrix])

        for window_add, accuracy_add, confusion_matrix_add in zip(windows_to_add, accuracy_add_list, confusion_matrix_add_list):
            sbls.add_enhancement_windows(window_add)

            test(test_loader, sbls, [accuracy_add, confusion_matrix_add])


    print(f"Results of experiment 3 with params initial_windows: {initial_windows}, windows_to_add: {window_add}")
    print(f"accuracy_init: {accuracy.compute()}")
    fig, _ = confusion_matrix.plot()
    fig.set_size_inches(7.0, 7.0)
    plot.savefig(f"images/ex3_{initial_windows}_{window_add}.png")

    for i, (accuracy_add, confusion_matrix_add) in enumerate(zip(accuracy_add_list, confusion_matrix_add_list)):
        print(f"accuracy_step_{i+1}: {accuracy_add.compute()}")
        fig, _ = confusion_matrix_add.plot()
        fig.set_size_inches(7.0, 7.0)
        plot.savefig(f"images/ex3_{initial_windows}_{window_add}_step_{i+1}.png")


# this experiment shows the effect of incrementally adding learning data
def experiment_4(data_increment_size: int):
    data_iterations = (int)(60000 / data_increment_size)

    accuracy_add_list = []
    confusion_matrix_add_list = []
    for i in range(data_iterations):
        accuracy_add_list.append(torchmetrics.classification.MulticlassAccuracy(num_classes=10))
        confusion_matrix_add_list.append(torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10, normalize="true"))

    for i in range(ITERATIONS):
        print(f"iteration {i}")
        train_loader, test_loader = load_mnist(data_increment_size, 10000)
        sbls = SBLS(input_size=28*28, output_size=10, simulation_steps=30, initital_feature_size=100, enhancement_nodes_per_window=100, initial_enhancement_window_num=20, output_is_one_hot=True)

        for i, (data, accuracy_add, confusion_matrix_add) in enumerate(zip(iter(train_loader), accuracy_add_list, confusion_matrix_add_list)):
            print(f"Training network iteration {i}")
            sbls.add_new_data(data)
            print(f"Training complete!")

            test(test_loader, sbls, [accuracy_add, confusion_matrix_add])


    print(f"Results of experiment 4 with params data_increment_size: {data_increment_size}")
    for i, (accuracy_add, confusion_matrix_add) in enumerate(zip(accuracy_add_list, confusion_matrix_add_list)):
        print(f"accuracy_step_{i}: {accuracy_add.compute()}")
        fig, _ = confusion_matrix_add.plot()
        fig.set_size_inches(7.0, 7.0)
        plot.savefig(f"images/ex4_{data_increment_size}_step_{i}.png")


def experiment_5():
    # set mse metric
    mse = torchmetrics.MeanSquaredError(num_outputs=2)

    # load input data
    train_folder = "training_set"
    test_folder = "test_set"

    training_input_filenames = sorted(os.listdir(os.path.join(".", train_folder, "input")))
    test_input_filenames = sorted(os.listdir(os.path.join(".", test_folder, "input")))

    training_images = [cv2.imread(os.path.join(".", train_folder, "input", file), 0) for file in training_input_filenames]
    test_images = [cv2.imread(os.path.join(".", test_folder, "input", file), 0) for file in test_input_filenames]

    # training_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in training_images], 0)
    # test_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in test_images], 0)
    
    training_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in training_images], 0)
    test_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in test_images], 0)

    # load target data
    training_steer_target = torch.tensor(np.load(os.path.join(".", train_folder, "steer_out.npy")), dtype=torch.float)
    training_throttle_target = torch.tensor(np.load(os.path.join(".", train_folder, "throttle_out.npy")), dtype=torch.float)
    training_target = torch.tensor(list(zip(training_steer_target, training_throttle_target)))

    test_steer_target = torch.tensor(np.load(os.path.join(".", test_folder, "steer_out.npy")), dtype=torch.float)
    test_throttle_target = torch.tensor(np.load(os.path.join(".", test_folder, "throttle_out.npy")), dtype=torch.float)
    test_target = torch.tensor(list(zip(test_steer_target, test_throttle_target)))


    # instantiate network
    sbls = SBLS(input_size=50*50, output_size=2, simulation_steps=30, initital_feature_size=500, enhancement_nodes_per_window=100, initial_enhancement_window_num=20, output_is_one_hot=False)

    print("TARGET")
    print(test_target[:10])

    # show_img(test_input[0])

    # test before training
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

    # feed training data into network
    sbls.add_new_data([training_input, training_target])

    # test after training
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

    for i in range(5):
        sbls.add_enhancement_windows(2)

        test2((test_input, test_target), sbls, [mse])
        print(mse.compute())


def experiment_6():
    # set mse metric
    mse = torchmetrics.MeanSquaredError(num_outputs=2)

    # load input data
    train_folder = "training_set"
    test_folder = "test_set"

    training_input_filenames = sorted(os.listdir(os.path.join(".", train_folder, "input")))
    test_input_filenames = sorted(os.listdir(os.path.join(".", test_folder, "input")))

    training_images = [cv2.imread(os.path.join(".", train_folder, "input", file), 0) for file in training_input_filenames]
    test_images = [cv2.imread(os.path.join(".", test_folder, "input", file), 0) for file in test_input_filenames]

    # training_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in training_images], 0)
    # test_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in test_images], 0)
    
    training_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in training_images], 0)
    test_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in test_images], 0)

    # load target data
    training_steer_target = torch.tensor(np.load(os.path.join(".", train_folder, "steer_out.npy")), dtype=torch.float)
    training_throttle_target = torch.tensor(np.load(os.path.join(".", train_folder, "throttle_out.npy")), dtype=torch.float)
    training_target = torch.tensor(list(zip(training_steer_target, training_throttle_target)))

    test_steer_target = torch.tensor(np.load(os.path.join(".", test_folder, "steer_out.npy")), dtype=torch.float)
    test_throttle_target = torch.tensor(np.load(os.path.join(".", test_folder, "throttle_out.npy")), dtype=torch.float)
    test_target = torch.tensor(list(zip(test_steer_target, test_throttle_target)))

    for i in range(6):
        print(f"iteration{i}")
        sbls = SBLS(input_size=50*50, output_size=2, initital_feature_size=500, enhancement_nodes_per_window=100, initial_enhancement_window_num=20+2*i, simulation_steps=30, output_is_one_hot=False)
        
        test2((test_input, test_target), sbls, [mse])
        print(mse.compute())

        sbls.add_new_data([training_input, training_target])

        test2((test_input, test_target), sbls, [mse])
        print(mse.compute())


def experiment_7():
    # set mse metric
    mse = torchmetrics.MeanSquaredError(num_outputs=2)

    # load input data
    train_folder = "training_set"
    test_folder = "test_set"

    training_input_filenames = sorted(os.listdir(os.path.join(".", train_folder, "input")))
    test_input_filenames = sorted(os.listdir(os.path.join(".", test_folder, "input")))

    training_images = [cv2.imread(os.path.join(".", train_folder, "input", file), 0) for file in training_input_filenames]
    test_images = [cv2.imread(os.path.join(".", test_folder, "input", file), 0) for file in test_input_filenames]

    # training_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in training_images], 0)
    # test_input = torch.stack([torch.tensor(cv2.GaussianBlur(image, (3, 3), 0), dtype=torch.float) / 255 for image in test_images], 0)
    
    training_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in training_images], 0)
    test_input = torch.stack([torch.tensor(image, dtype=torch.float) / 255 for image in test_images], 0)

    # load target data
    training_steer_target = torch.tensor(np.load(os.path.join(".", train_folder, "steer_out.npy")), dtype=torch.float)
    training_throttle_target = torch.tensor(np.load(os.path.join(".", train_folder, "throttle_out.npy")), dtype=torch.float)
    training_target = torch.tensor(list(zip(training_steer_target, training_throttle_target)))

    test_steer_target = torch.tensor(np.load(os.path.join(".", test_folder, "steer_out.npy")), dtype=torch.float)
    test_throttle_target = torch.tensor(np.load(os.path.join(".", test_folder, "throttle_out.npy")), dtype=torch.float)
    test_target = torch.tensor(list(zip(test_steer_target, test_throttle_target)))

    # instantiate network
    sbls = SBLS(input_size=50*50, output_size=2, initital_feature_size=500, enhancement_nodes_per_window=100, initial_enhancement_window_num=20, simulation_steps=30, output_is_one_hot=False)

    # test untrained network
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

    # feed first 5000 datapoints
    sbls.add_new_data([training_input[:5000], training_target[:5000]])

    # test again
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

    # feed 1000 datapoints more
    sbls.add_new_data([training_input[5000:6000], training_target[5000:6000]])

    # test again
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

    # feed remaining 683 datapoints
    sbls.add_new_data([training_input[6000:], training_target[6000:]])

    # test again
    test2((test_input, test_target), sbls, [mse])
    print(mse.compute())

experiment_7()





# experiment_1(50, 10)
# experiment_1(50, 20)
# experiment_1(100, 10)
# experiment_1(200, 20)
# experiment_1(500, 20)
# experiment_1(500, 40)


# experiment_2(30000)
# experiment_2(10000)
# experiment_2(5000)
    
# experiment_3(20, [10])
# experiment_3(20, [5, 5])
# experiment_3(20, [2, 2, 2, 2, 2])

# experiment_4(30000)
# experiment_4(20000)
# experiment_4(10000)
# experiment_4(5000)