import os.path
import random
from PIL import Image
import numpy as np


classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def load_data(path):
    dataset = []
    for name, label in classes.items():
        folder_path = os.path.join(path, name)
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                image = Image.open(image_path).resize((64, 64))
                image_arr = np.array(image) / 255.0
                dataset.append((image_arr, label))
    random.shuffle(dataset)

    return dataset


class FullConnectedLayer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def calculate_grads(self):
        pass


class MaxPoolLayer:
    def __init__(self):
        pass

    def forward(self):
        pass


class ConvolutionLayer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def calculate_grads(self):
        pass


class NeuralNetwork:
    def __init__(self):
        self.WxTh = 0

    def forward(self):
        pass

    def backward_prop(self):
        pass

    def train(self):
        pass
