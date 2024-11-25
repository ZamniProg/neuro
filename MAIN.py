import os.path
import random
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def load_data(path):
    learn_data = []
    train_data = []

    for name, label in classes.items():
        folder_path = os.path.join(path, name)
        batch_of_files = os.listdir(folder_path)

        # Определяем количество данных для тренировочного набора (10%)
        for_train = len(batch_of_files) // 10  # Количество файлов, которые пойдут в тренировочные данные

        for idx, file in enumerate(batch_of_files):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                image = Image.open(image_path).resize((128, 128))  # Меняем размер изображения на 64x64
                image_arr = np.array(image) / 255.0  # Нормализуем пиксели (0-1)

                # Добавляем изображение в соответствующий список (train или learn)
                if idx < for_train:
                    train_data.append((image_arr, label))
                else:
                    learn_data.append((image_arr, label))

    # Перемешиваем данные
    random.shuffle(learn_data)
    random.shuffle(train_data)

    # Извлекаем изображения и метки из данных
    learn_images = np.array([item[0] for item in learn_data])
    learn_labels = np.array([item[1] for item in learn_data])

    train_images = np.array([item[0] for item in train_data])
    train_labels = np.array([item[1] for item in train_data])

    # Возвращаем данные в виде кортежей: изображения и метки для learn и train
    return (learn_images, learn_labels), (train_images, train_labels)


class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_prop(self, input_data, training=True):
        if training:
            self.mask = (np.random.rand(*input_data.shape) > self.dropout_rate).astype(float)
            return input_data * self.mask / (1.0 - self.dropout_rate)
        else:
            return input_data

    def backward_prop(self, d_out):
        return d_out * self.mask / (1.0 - self.dropout_rate)


class FullConnectedLayer:
    """Полносвязный слой для нелинейного преобразования всех предыдущих входных данных"""

    def __init__(self, input_size, output_size, history_file, activation=relu, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.learning_rate = learning_rate
        self.history_file = history_file
        self.input_data = None
        self.d_weights = None
        self.d_biases = None
        self.d_input = None

    def get_weights(self):
        return self.weights

    def forward(self, input_data):
        input_x = input_data.reshape(input_data.shape[0], -1)
        self.input_data = input_x

        z = np.dot(input_x, self.weights) + self.biases

        output = self.activation(z) if self.activation else z  # функция активации
        return output

    def calculate_grads(self, d_out, learning_rate=0.01):
        self.d_weights = np.dot(self.input_data.T, d_out)
        self.d_biases = np.sum(d_out, axis=0, keepdims=True)

        self.d_input = np.dot(d_out, self.weights.T)

        count = self.input_data.shape[0]

        self.d_input /= count
        self.d_weights /= count
        self.d_biases /= count

        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases

        return self.d_input

    def save_history(self, losses):
        """Сохраняет текущие данные о весах, смещениях и градиентах"""
        try:
            history = np.load(self.history_file, allow_pickle=True)
            weights_history = history['weights'].tolist()
            biases_history = history['biases'].tolist()
            grads_history = history['grads'].tolist()
            losses_history = history['losses'].tolist()
        except FileNotFoundError:
            weights_history = []
            biases_history = []
            grads_history = []
            losses_history = []

        weights_history.append(self.weights.copy())
        biases_history.append(self.biases.copy())
        grads_history.append({
            "d_weights": self.d_weights.copy(),
            "d_biases": self.d_biases.copy(),
            "d_input": self.d_input.copy()
        })
        losses_history.append(losses.copy())

        np.savez(self.history_file,
                 weights=weights_history,
                 biases=biases_history,
                 grads=grads_history,
                 losses=losses_history)


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
        pool_height, pool_width = self.pool_size
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
        self.image = None

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

        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases

        return d_input


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 filter_size, num_filters,
                 pool_size, stride,
                 learning_rate=0.01, activation=relu):
        self.conv1 = ConvolutionLayer(filter_size, num_filters, 3, stride, learning_rate)
        self.mpl1 = MaxPoolLayer((pool_size // 2, pool_size // 2), stride // 2)
        self.conv2 = ConvolutionLayer(filter_size, num_filters, num_filters, stride, learning_rate)
        self.mpl2 = MaxPoolLayer((pool_size, pool_size), stride)
        self.fcl = FullConnectedLayer(hidden_size, hidden_size, "history_fcl1.npz", activation, learning_rate)
        self.fcl2 = FullConnectedLayer(hidden_size, output_size, "history_fcl2.npz", activation, learning_rate)

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
        query = input("Загрузить уже готовые веса для модели? (y/n) - ")
        if query == "y":
            self.load_model(save)

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

                # Разделение данных на мини-батчи
                for batch_start in range(0, num_samples, batch_size):
                    acc = []
                    loss_b = []
                    batch_weights = []

                    time_s_b = time.time()
                    batch_end = min(batch_start + batch_size, num_samples)
                    image_batch = images[batch_start:batch_end]
                    label_batch = labels[batch_start:batch_end]

                    # Прямое распространение (forward pass)
                    predictions = self.forward(image_batch)

                    # Преобразуем метки в one-hot формат для кросс-энтропии
                    one_hot_labels = np.zeros_like(predictions)
                    one_hot_labels[np.arange(label_batch.size), label_batch] = 1

                    # Вычисляем функцию ошибки (Cross-Entropy Loss)
                    loss = -np.sum(one_hot_labels * np.log(predictions + 1e-7)) / batch_size
                    total_loss += loss
                    loss_b.append(loss)

                    # Оценка корректных предсказаний
                    now_correct_predictions = np.sum(np.argmax(predictions, axis=1) == label_batch)
                    correct_predictions += now_correct_predictions

                    # Градиент ошибки
                    grad_loss = predictions - one_hot_labels  # Градиент CrossEntropyLoss

                    self.backward_prop(grad_loss)  # Запускаем обратное распространение

                    # Сохранение весов после обратного распространения
                    batch_weights.append(self.fcl.get_weights())

                    time_e_b = time.time()
                    print(f"Loss: {loss:.4f}, Accuracy: {now_correct_predictions / batch_size:.4%}\n"
                          f"Time: {time_e_b - time_s_b} sec.")

                    # Сохраняем результаты для текущего батча
                    weights_b.append(batch_weights)
                    losses_b.append(loss_b)

                time_e = time.time()

                # Средняя ошибка и точность за эпоху
                average_loss = total_loss / (num_samples // batch_size)
                accuracy = correct_predictions / num_samples

                # Сохранение весов и потерь для текущей эпохи
                epoch_weights.append(self.fcl.get_weights())
                weights_e.append(epoch_weights)
                epoch_losses.append(average_loss)
                losses_e.append(epoch_losses)

                self.save_model(save)
                self.fcl.save_history(average_loss)
                self.fcl2.save_history(average_loss)

                # Вывод результатов текущей эпохи
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                      f"Time: {time_e - time_s} sec.")
                f.write(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                        f"Time: {time_e - time_s} sec.")

        # Сохранение в файлы
        with open("Batch_Losses.txt", "w") as f_b_loss:
            f_b_loss.write(str(losses_b))

        with open("Epoch_Losses.txt", "w") as f_e_loss:
            f_e_loss.write(str(losses_e))

        with open("Batch_Weights.txt", "w") as f_b_weights:
            f_b_weights.write(str(weights_b))

        with open("Epoch_Weights.txt", "w") as f_e_weights:
            f_e_weights.write(str(weights_e))

    def save_model(self, file_path):
        """
        Сохраняет параметры модели в файл.
        :param file_path: путь к файлу для сохранения
        """
        model_data = {
            "conv1_filters": self.conv1.filters,
            "conv1_biases": self.conv1.biases,
            "conv1_activation": self.conv1.activation,
            "mpl1_pool_size": self.mpl1.pool_size,
            "mpl1_stride": self.mpl1.stride,
            "conv2_filters": self.conv2.filters,
            "conv2_biases": self.conv2.biases,
            "mpl2_pool_size": self.mpl2.pool_size,
            "mpl2_stride": self.mpl2.stride,
            "conv2_activation": self.conv1.activation,
            "fcl_weights": self.fcl.weights,
            "fcl_biases": self.fcl.biases,
            "fcl_activation": self.fcl.activation,
            "fcl2_weights": self.fcl2.weights,
            "fcl2_biases": self.fcl2.biases,
            "fcl2_activation": self.fcl2.activation
        }
        np.savez(file_path, **model_data)
        print(f"Модель сохранена в файл: {file_path}")

    def load_model(self, file_path):
        """
        Загружает параметры модели из файла.
        :param file_path: путь к файлу для загрузки
        """
        model_data = np.load(file_path)
        self.conv1.filters = model_data["conv1_filters"]
        self.conv1.biases = model_data["conv1_biases"]
        self.conv1.activation = model_data["conv1_activation"]
        self.mpl1.pool_size = model_data["mpl1_pool_size"]
        self.mpl1.stride = model_data["mpl1_stride"]
        self.conv2.filters = model_data["conv2_filters"]
        self.conv2.biases = model_data["conv2_biases"]
        self.conv2.activation = model_data["conv2_activation"]
        self.mpl2.pool_size = model_data["mpl2_pool_size"]
        self.mpl2.stride = model_data["mpl2_stride"]
        self.fcl.weights = model_data["fcl_weights"]
        self.fcl.biases = model_data["fcl_biases"]
        self.fcl.activation = model_data["fcl_activation"]
        self.fcl2.weights = model_data["fcl2_weights"]
        self.fcl2.biases = model_data["fcl2_biases"]
        self.fcl2.activation = model_data["fcl2_activation"]
        print(f"Модель загружена из файла: {file_path}")


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

    # Прямое распространение и оценка точности
    for i in range(num_samples):
        # Выводим прогресс каждые 100 шагов
        if i % 100 == 0:
            print(f"Testing: {i}/{num_samples} samples processed")

        # Получаем прогноз для текущего изображения
        prediction = model.forward(test_images[i:i + 1])  # Передаем одно изображение (batch size = 1)

        # Получаем класс с наибольшей вероятностью
        predicted_class = np.argmax(prediction, axis=1)

        # Проверяем, совпадает ли предсказанный класс с реальной меткой
        if predicted_class == test_labels[i]:
            correct_predictions += 1

    # Вычисляем точность
    accuracy = correct_predictions / num_samples
    print(f"Test Accuracy: {accuracy:.4%}")

    return accuracy


def main():
    # Загрузка данных
    learn, train = load_data("flower_photos")
    learn_images, learn_labels = learn
    train_images, train_labels = train

    real_model = "saved_model"

    output_size = len(classes)  # рассчитано автоматически
    input_size = 128 * 128  # рассчитать (вроде обычный входной слой)
    filter_size = 2  # рассчитать (1 - [128x128x3] -> [64x64xN]; 2 - [64x64xM] -> [32x32xM]), 1 - (2), 2 - (2)
    num_filters = 32  # рассчитать (не уверен, но вроде наплевать)
    hidden_size = 16 * 16 * num_filters  # рассчитать (из конца в выходной)
    pool_size = 2  # рассчитать (1 - [64x64xN] -> [64x64xN]; 2 - [32x32xM] -> [16x16xM]), 1 - (2), 2 - (2)
    stride = 2  # рассчитать (2)

    gray_image = learn_images[0].mean(axis=-1)
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"Label: {learn_labels[0]}")
    plt.axis('off')
    plt.show()

    model = NeuralNetwork(input_size, hidden_size, output_size,
                          filter_size, num_filters, pool_size, stride)

    model.train(learn_images, learn_labels, real_model)

    test_accuracy = test_model(model, train_images, train_labels)
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
