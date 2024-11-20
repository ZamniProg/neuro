import os.path
import random
from PIL import Image
import numpy as np


classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def relu(x):
    return np.max(0, x)


def load_data(path):
    learn_data = []
    train_data = []
    for name, label in classes.items():
        folder_path = os.path.join(path, name)
        batch_of_files = os.listdir(folder_path)
        for_train = len(batch_of_files) / 10
        for idx, file in enumerate(batch_of_files):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                image = Image.open(image_path).resize((64, 64))
                image_arr = np.array(image) / 255.0
                if idx <= for_train:
                    train_data.append((image_arr, label))
                else:
                    learn_data.append((image_arr, label))
    random.shuffle(learn_data)
    random.shuffle(train_data)

    return learn_data, train_data


class FullConnectedLayer:
    """Полносвязный слой для нелинейного преобразования всех предыдущих входных данных"""
    def __init__(self, input_size, output_size, activation=relu):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input_data = None

    def forward(self, input_data):
        h_s, w_s, channels = input_data.shape

        input_x = input_data.reshape(input_data.shape[0], -1)
        self.input_data = input_x

        z = np.dot(input_x, self.weights) + self.biases

        output = self.activation(z) if self.activation else z  # функция активации
        return output

    def calculate_grads(self, d_out, learning_rate=0.01):
        d_weights = np.dot(self.input_data.T, d_out)
        d_biases = np.sum(d_out, axis=0, keepdims=True)

        d_input = np.dot(d_out, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


class MaxPoolLayer:
    """Слой для уменьшения размеров изображения (оставляем только значимые веса)"""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_data = None

    def forward(self, input_data):
        h, w, c = input_data.shape

        self.input_data = input_data

        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        out = np.zeros_like((h_out, w_out, c))

        for f in range(c):
            for k1 in range(0, h_out):
                for k2 in range(0, w_out):
                    h_start = k1 * self.stride
                    h_end = h_start + self.pool_size

                    w_start = k2 * self.stride
                    w_end = w_start * self.pool_size

                    out[k1, k2, f] = np.max(input_data[h_start:h_end, w_start:w_end, f])

        return out

    def backward_prop(self, d_out):
        h_s, w_s, channels = self.input_data.shape
        h_e, w_e, channels = d_out.shape

        d_input = np.zeros_like(self.input_data)

        for c in range(channels):
            for h in range(h_e):
                for w in range(w_e):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size

                    w_start = w * self.stride
                    w_end = w_start + self.pool_size

                    region = self.input_data[h_start:h_end, w_start:w_end, c]
                    max_val = np.max(region)

                    d_input[h_start:h_end, w_start:w_end, c] += (region == max_val) * d_out[h, w, c]

        return d_input


class ConvolutionLayer:     # maybe ready
    """Слой для свертки изображения"""
    def __init__(self, filter_size, num_filters, num_channels, stride=2, learning_rate=0.01, activation=relu):
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.stride = stride
        self.filter_size = filter_size
        self.filters = np.random.randn(filter_size, filter_size, num_channels, num_filters) * 0.01  # rework
        self.biases = np.zeros((num_filters, ))
        self.learning_rate = learning_rate
        self.activation = activation
        self.image = None

    def forward(self, image):
        h, w, _, channels = image.shape
        self.image = image
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1

        out = np.zeros((h_out, w_out, self.num_channels))

        for f in range(self.num_filters):
            for k1 in range(self.filter_size):
                for k2 in range(self.filter_size):
                    h_start = k1 * self.stride
                    h_end = h_start + self.filter_size
                    w_start = k2 * self.stride
                    w_end = w_start + self.filter_size

                    region = image[h_start:h_end, w_start:w_end, :]

                    out[k1, k2, f] = np.sum(region * self.filters[:, :, :, f]) + self.biases[f]

        out = self.activation(out) if self.activation else out

        return out

    def backward_prop(self, grad_out):
        h_image, w_image, c_image = self.image.shape
        h_out, w_out, num_filters = grad_out.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.image)

        for f in range(self.num_filters):
            for k1 in range(h_out):
                for k2 in range(w_out):
                    h_start = k1 * self.stride
                    h_end = h_start + self.filter_size
                    w_start = k2 * self.stride
                    w_end = w_start + self.filter_size

                    region = self.image[h_start:h_end, w_start:w_end, :]

                    d_filters[:, :, :, f] += region * grad_out[k1, k2, f]

                    d_input[h_start:h_end, w_start:w_end, :] += self.filters[:, :, :, f] * grad_out[k1, k2, f]

            d_biases[f] += np.sum(grad_out[:, :, f])

        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases

        return d_input, d_filters, d_biases


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, filters, filters_size, learning_rate=0.01):
        pass

    def forward(self):
        pass

    def backward_prop(self):
        pass

    def train(self):
        pass


learn, _ = load_data(r"flower_photos")
print(learn[0][0].shape)
