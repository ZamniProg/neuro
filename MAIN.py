import os.path
import random
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def relu(x):
    return np.maximum(0, x)


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, grads, t):
        """
        Обновляет веса с использованием алгоритма Adam.
        :param weights: текущие веса
        :param grads: градиенты
        :param t: текущий шаг обучения
        :return: обновленные веса
        """
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights

    def get(self):
        return {
            'm': self.m.tolist(),
            'v': self.v.tolist(),
            't': self.t,
            'beta1': self.beta1,
            'beta2': self.beta2
        }


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def invert_image(image_path):
    image = Image.open(image_path).resize((128, 128))
    image_arr = np.array(image) / 255.0
    return image_arr


def load_data(path):
    learn_data = []
    train_data = []

    for name, label in classes.items():
        folder_path = os.path.join(path, name)
        batch_of_files = os.listdir(folder_path)

        for_train = len(batch_of_files) // 10

        for idx, file in enumerate(batch_of_files):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                image = Image.open(image_path).resize((128, 128))
                image_arr = np.array(image) / 255.0

                if idx < for_train:
                    train_data.append((image_arr, label))
                else:
                    learn_data.append((image_arr, label))

    random.shuffle(learn_data)
    random.shuffle(train_data)

    learn_images = np.array([item[0] for item in learn_data])
    learn_labels = np.array([item[1] for item in learn_data])

    train_images = np.array([item[0] for item in train_data])
    train_labels = np.array([item[1] for item in train_data])

    return (learn_images, learn_labels), (train_images, train_labels)


class FullConnectedLayer:
    def __init__(self, input_size, output_size, history_file, activation=relu, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.history_file = history_file
        self.learning_rate = learning_rate
        self.grads = {}
        self.input_data = None
        self.d_weights = None
        self.d_biases = None
        self.d_input = None

        self.optimizer = AdamOptimizer(learning_rate=learning_rate)

    def calculate_grads(self, d_out, learning_rate=0.01):
        self.d_weights = np.dot(self.input_data.T, d_out)
        self.d_biases = np.sum(d_out, axis=0, keepdims=True)

        self.d_input = np.dot(d_out, self.weights.T)

        count = self.input_data.shape[0]

        self.d_input /= count
        self.d_weights /= count
        self.d_biases /= count

        # Обновляем веса с помощью Adam
        self.weights = self.optimizer.update(self.weights, self.d_weights, self.optimizer.t)
        self.biases -= self.learning_rate * self.d_biases  # Для biases можно оставить обычное обновление

        self.grads = {
            'input': self.d_input,
            'weights': self.d_weights,
            'biases': self.d_biases
        }

        return self.d_input

    def get_weights(self):
        return self.weights

    def forward(self, input_data):
        input_x = input_data.reshape(input_data.shape[0], -1)
        self.input_data = input_x

        z = np.dot(input_x, self.weights) + self.biases

        output = self.activation(z) if self.activation else z  # функция активации
        return output

    def save_history(self, history_file):
        weights_history = []
        biases_history = []
        grads_history = []

        weights_history.append(self.weights.copy())
        biases_history.append(self.biases.copy())

        grads_history.append({
            'input': self.grads['input'].tolist(),
            'weights': self.grads['weights'].tolist(),
            'biases': self.grads['biases'].tolist()
        })

        np.savez(history_file,
                 weights=weights_history,
                 biases=biases_history,
                 grads=grads_history)

        print(f"Сохранена информация о weights, biases, grads в файл: {history_file}")


class MaxPoolLayer:
    """Слой для уменьшения изображения и выявления наиболее полезных параметров"""
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input_shape = None
        self.forward_cache = None

    def forward(self, input_data):
        self.input_shape = input_data.shape
        (batch_size, height, width, channels) = self.input_shape
        try:
            pool_height, pool_width = self.pool_size
        except:
            pool_height = self.pool_size
            pool_width = self.pool_size
        stride = self.stride

        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        self.forward_cache = input_data
        output = np.zeros((batch_size, out_height, out_width, channels))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i, start_j = i * stride, j * stride
                        end_i, end_j = start_i + pool_height, start_j + pool_width
                        output[b, i, j, c] = np.max(input_data[b, start_i:end_i, start_j:end_j, c])

        return output

    def backward_prop(self, d_out):
        (batch_size, height, width, channels) = self.input_shape
        pool_height, pool_width = self.pool_size
        stride = self.stride
        d_input = np.zeros(self.input_shape)

        h_e = (height - pool_height) // self.stride + 1
        w_e = (width - pool_width) // self.stride + 1

        d_out_reshaped = d_out.reshape(batch_size, h_e, w_e, channels)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(h_e):
                    for j in range(w_e):
                        start_i, start_j = i * stride, j * stride
                        end_i, end_j = start_i + pool_height, start_j + pool_width

                        # Получаем текущий регион из входных данных
                        input_region = self.forward_cache[b, start_i:end_i, start_j:end_j, c]
                        # Маска для элементов, которые были максимальными
                        mask = (input_region == np.max(input_region))
                        # Градиент распространяется только на максимальные элементы
                        d_input[b, start_i:end_i, start_j:end_j, c] += mask * d_out_reshaped[b, i, j, c]

        return d_input


class ConvolutionLayer:  # maybe ready
    """Слой для свертки изображения"""

    def __init__(self, filter_size, num_filters, num_channels=3, stride=2, learning_rate=0.01, activation=relu):
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.stride = stride
        self.filter_size = filter_size
        self.filters = np.random.randn(filter_size, filter_size, num_channels, num_filters) * np.sqrt(2 / filter_size)
        self.biases = np.zeros((num_filters,))
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.image = None
        self.grads = {}

    def forward(self, image):
        count, h, w, channels = image.shape
        self.image = image
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1
        out = np.zeros((count, h_out, w_out, self.num_filters))

        for b in range(count):
            for f in range(self.num_filters):
                for i in range(0, h_out):
                    for j in range(0, w_out):
                        h_start, w_start = i * self.stride, j * self.stride
                        h_end, w_end = h_start + self.filter_size, w_start + self.filter_size
                        region = image[b, h_start:h_end, w_start:w_end, :]
                        out[b, i, j, f] = np.sum(region * self.filters[:, :, :, f]) + self.biases[f]

        return self.activation(out)

    def backward_prop(self, grad_out):
        count, h_image, w_image, c_image = self.image.shape
        _, h_out, w_out, num_filters = grad_out.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.image)

        for i in range(count):
            for f in range(num_filters):
                for k1 in range(h_out):
                    for k2 in range(w_out):
                        h_start = k1 * self.stride
                        h_end = h_start + self.filter_size
                        w_start = k2 * self.stride
                        w_end = w_start + self.filter_size

                        region = self.image[i, h_start:h_end, w_start:w_end, :]

                        d_filters[:, :, :, f] += region * grad_out[i, k1, k2, f]

                        d_input[i, h_start:h_end, w_start:w_end, :] += self.filters[:, :, :, f] * grad_out[i, k1, k2, f]

                d_biases[f] += np.sum(grad_out[i, :, :, f])

        d_biases /= count
        d_filters /= count
        d_input /= count

        self.grads = {
            'input': d_input,
            'filters': d_filters,
            'biases': d_biases
        }

        self.filters = self.optimizer.update(self.filters, d_filters, self.optimizer.t)
        self.biases -= self.optimizer.learning_rate * d_biases

        return d_input

    def save_history(self, history_file):
        weights_history = []
        biases_history = []
        grads_history = []

        weights_history.append(self.filters.copy())
        biases_history.append(self.biases.copy())

        grads_history.append({
            'input': self.grads['input'].tolist(),
            'filters': self.grads['filters'].tolist(),
            'biases': self.grads['biases'].tolist()
        })

        # Сохраняем данные в файл
        np.savez(history_file,
                 weights=weights_history,
                 biases=biases_history,
                 grads=grads_history)

        print(f"Сохранена информация о filters, biases, grads в файл: {history_file}")


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 filter_size, num_filters,
                 pool_size, stride,
                 learning_rate=0.01, activation=relu):
        self.conv1 = ConvolutionLayer(filter_size, num_filters // 2, 3, stride, learning_rate)
        self.mpl1 = MaxPoolLayer((pool_size, pool_size), stride)
        self.conv2 = ConvolutionLayer(filter_size, num_filters, num_filters // 2, stride, learning_rate)
        self.mpl2 = MaxPoolLayer((pool_size, pool_size), stride)
        self.fcl = FullConnectedLayer(hidden_size, 128, "history_fcl1.npz", activation, learning_rate)
        self.fcl2 = FullConnectedLayer(128, output_size, "history_fcl2.npz", learning_rate=learning_rate)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.mpl1.forward(x)
        x = self.conv2.forward(x)
        x = self.mpl2.forward(x)
        x = self.fcl.forward(x)
        x = self.fcl2.forward(x)
        x = softmax(x)
        return x

    def backward_prop(self, d):
        d = self.fcl2.calculate_grads(d)
        d = self.fcl.calculate_grads(d)
        d = self.mpl2.backward_prop(d)
        d = self.conv2.backward_prop(d)
        d = self.mpl1.backward_prop(d)
        d = self.conv1.backward_prop(d)
        return d

    def train(self, images, labels, save, epochs=10, batch_size=29):
        num_samples = images.shape[0]
        losses_e = []
        losses_b = []
        accuracies_b = []
        accuracies_e = []
        weights_b = []
        weights_e = []
        save_dir = "saves"
        query = input("Загрузить уже готовые веса для модели? (y/n) - ")
        if query == "y":
            self.load_model(f"{save_dir}/full_model.npz")

        with open("Accuracy.txt", "w") as f:
            for epoch in range(epochs):
                time_s = time.time()
                total_loss = 0
                correct_predictions = 0
                epoch_weights = []
                epoch_losses = []

                # Перемешивание данных в начале каждой эпохи
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                images = images[indices]
                labels = labels[indices]

                for batch_start in tqdm(range(0, num_samples, batch_size)):
                    acc = []
                    loss_b = []
                    batch_weights = []

                    time_s_b = time.time()
                    batch_end = min(batch_start + batch_size, num_samples)
                    image_batch = images[batch_start:batch_end]
                    label_batch = labels[batch_start:batch_end]

                    predictions = self.forward(image_batch)

                    one_hot_labels = np.zeros_like(predictions)
                    one_hot_labels[np.arange(label_batch.size), label_batch] = 1

                    loss = -np.sum(one_hot_labels * np.log(predictions + 1e-7)) / batch_size
                    total_loss += loss
                    loss_b.append(loss)

                    now_correct_predictions = np.sum(np.argmax(predictions, axis=1) == label_batch)
                    correct_predictions += now_correct_predictions

                    grad_loss = predictions - one_hot_labels

                    self.backward_prop(grad_loss)

                    batch_weights.append(self.fcl.get_weights())

                    time_e_b = time.time()

                    weights_b.append(batch_weights)
                    losses_b.append(loss_b)

                time_e = time.time()

                average_loss = total_loss / (num_samples // batch_size)
                accuracy = correct_predictions / num_samples

                epoch_weights.append(self.fcl.get_weights())
                weights_e.append(epoch_weights)
                epoch_losses.append(average_loss)
                losses_e.append(epoch_losses)

                self.save_model(save)
                self.fcl.save_history(f"{save_dir}/fcl1_history_{epoch + 1}.npz")
                self.fcl2.save_history(f"{save_dir}/fcl2_history_{epoch + 1}.npz")
                self.conv1.save_history(f"{save_dir}/conv1_history_{epoch + 1}.npz")
                self.conv2.save_history(f"{save_dir}/conv2_history_{epoch + 1}.npz")

                print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                      f"Time: {time_e - time_s} sec.")
                f.write(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                        f"Time: {time_e - time_s} sec.")
        print("Learn of model is ready! Enjoy!")

    def save_model(self, file_path):
        """
        Сохраняет параметры модели в файл.
        :param file_path: путь к файлу для сохранения
        """
        model_data = {
            "conv1_filters": self.conv1.filters,
            "conv1_biases": self.conv1.biases,

            "mpl1_pool_size": self.mpl1.pool_size,
            "mpl1_stride": self.mpl1.stride,

            "conv2_filters": self.conv2.filters,
            "conv2_biases": self.conv2.biases,

            "mpl2_pool_size": self.mpl2.pool_size,
            "mpl2_stride": self.mpl2.stride,

            "fcl_weights": self.fcl.weights,
            "fcl_biases": self.fcl.biases,

            "fcl2_weights": self.fcl2.weights,
            "fcl2_biases": self.fcl2.biases,
        }
        np.savez(file_path, **model_data)
        print(f"Модель сохранена в файл: {file_path}")

    def load_model(self, file_path):
        """
        Загружает параметры модели из файла.
        :param file_path: путь к файлу для загрузки
        """
        model_data = np.load(file_path, allow_pickle=True)
        self.conv1.filters = model_data["conv1_filters"]
        self.conv1.biases = model_data["conv1_biases"]
        self.conv1.activation = relu
        self.mpl1.pool_size = model_data["mpl1_pool_size"]
        self.mpl1.stride = model_data["mpl1_stride"]
        self.conv2.filters = model_data["conv2_filters"]
        self.conv2.biases = model_data["conv2_biases"]
        self.conv2.activation = relu
        self.mpl2.pool_size = model_data["mpl2_pool_size"]
        self.mpl2.stride = model_data["mpl2_stride"]
        self.fcl.weights = model_data["fcl_weights"]
        self.fcl.biases = model_data["fcl_biases"]
        self.fcl.activation = relu
        self.fcl2.weights = model_data["fcl2_weights"]
        self.fcl2.biases = model_data["fcl2_biases"]

        print(f"Модель загружена из файла: {file_path}")

    def use(self, image_path):
        image = invert_image(image_path)
        image = np.expand_dims(image, axis=0)
        true_image = Image.open(image_path)
        prediction = self.forward(image)
        result = np.argmax(prediction, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(true_image)
        axes[0].axis('off')
        axes[0].set_title("True Image")

        axes[1].imshow(true_image)
        axes[1].axis('off')
        axes[1].set_title(f"Prediction: {list(classes.keys())[result[0]]}")

        plt.tight_layout()
        plt.show()


def test_model(model, test_images, test_labels):
    """
    Функция для тестирования обученной модели на тестовом наборе данных.

    :param model: обученная модель (экземпляр NeuralNetwork)
    :param test_images: изображения для тестирования (матрица данных)
    :param test_labels: метки классов для тестовых изображений
    :return: точность на тестовом наборе данных
    """
    num_samples = test_images.shape[0]
    correct_predictions = 0

    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Testing: {i}/{num_samples} samples processed")

        prediction = model.forward(test_images[i:i+1, :, :, :])

        predicted_class = np.argmax(prediction, axis=1)

        if predicted_class == test_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / num_samples
    print(f"Test Accuracy: {accuracy:.4%}")

    return accuracy


def main():
    # Загрузка данных
    learn, train = load_data("flower_photos")
    learn_images, learn_labels = learn
    train_images, train_labels = train

    real_model = "saves/full_model.npz"

    output_size = len(classes)
    input_size = 128*128
    filter_size = 2
    num_filters = 64
    hidden_size = 4096
    pool_size = 2
    stride = 2

    model = NeuralNetwork(input_size, hidden_size, output_size,
                          filter_size, num_filters, pool_size, stride)

    # model.train(learn_images, learn_labels, real_model)

    model.load_model("saves/full_model.npz")
    # model.use("/home/zamni/PycharmProjects/neuro/flower_photos/tulips/65347450_53658c63bd_n.jpg")
    test_model(model, train_images, train_labels)


if __name__ == "__main__":
    main()
