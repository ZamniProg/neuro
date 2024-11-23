import os.path
import random
from PIL import Image
import numpy as np
import time

classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Численная стабильность
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


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

    def __init__(self, input_size, output_size, activation=relu, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.learning_rate = learning_rate
        self.input_data = None

    def get_weights(self):
        return self.weights

    def forward(self, input_data):
        input_x = input_data.reshape(input_data.shape[0], -1)
        self.input_data = input_x

        z = np.dot(input_x, self.weights) + self.biases

        output = self.activation(z) if self.activation else z  # функция активации
        return output

    def calculate_grads(self, d_out, learning_rate=0.01):
        d_weights = np.dot(self.input_data.T, d_out)
        d_biases = np.sum(d_out, axis=0, keepdims=True)

        d_input = np.dot(d_out, self.weights.T)

        self.weights -= self.learning_rate * d_weights
        self.biases -= self.learning_rate * d_biases

        return d_input


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
        self.filters = np.random.randn(filter_size, filter_size, num_channels, num_filters) * 0.01  # rework
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

        for i in range(count):
            for f in range(self.num_filters):
                for k1 in range(self.filter_size):
                    for k2 in range(self.filter_size):
                        h_start = k1 * self.stride
                        h_end = h_start + self.filter_size
                        w_start = k2 * self.stride
                        w_end = w_start + self.filter_size

                        region = image[i, h_start:h_end, w_start:w_end, :]

                        out[i, k1, k2, f] = np.sum(region * self.filters[:, :, :, f]) + self.biases[f]

        out = self.activation(out) if self.activation else out

        return out

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

        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases

        return d_input


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 filter_size, num_filters,
                 pool_size, stride,
                 learning_rate=0.001, activation=relu):
        self.conv1 = ConvolutionLayer(filter_size, num_filters, 3, stride, learning_rate)
        self.mpl1 = MaxPoolLayer((pool_size // 2, pool_size // 2), stride // 2)
        self.conv2 = ConvolutionLayer(filter_size, num_filters, num_filters, stride, learning_rate)
        self.mpl2 = MaxPoolLayer((pool_size, pool_size), stride)
        self.fcl = FullConnectedLayer(hidden_size, hidden_size, activation, learning_rate)
        self.fcl2 = FullConnectedLayer(hidden_size, output_size, activation, learning_rate)

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

    def train(self, images, labels, epochs=10, batch_size=29 * 2):
        num_samples = images.shape[0]
        losses_e = []
        losses_b = []
        accuracies_b = []
        accuracies_e = []
        weights_b = []
        weights_e = []

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

                    # Обратное распространение (backward pass)
                    self.backward_prop(grad_loss)

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
    # Note: надо короче переделать по размерам(что бы в общем у нас было не 3x3 в конце, а хотя бы 6x6)
    learn, train = load_data("flower_photos")
    learn_images, learn_labels = learn
    train_images, train_labels = train

    output_size = len(classes)  # рассчитано автоматически
    input_size = 128 * 128  # рассчитать (вроде обычный входной слой)
    filter_size = 4  # рассчитать (1 - [128x128x3] -> [64x64xN]; 2 - [63x63xM] -> [31x31xM]), 1 - (2), 2 - (2)
    num_filters = 40  # рассчитать (не уверен, но вроде наплевать)
    hidden_size = 15 * 15 * num_filters  # рассчитать (из конца в выходной)
    pool_size = 2  # рассчитать (1 - [64x64xN] -> [63x63xN]; 2 - [31x31xM] -> [14x14xM]), 1 - (2), 2 - (2)
    stride = 2  # рассчитать (2)

    model = NeuralNetwork(input_size, hidden_size, output_size,
                          filter_size, num_filters, pool_size, stride)

    model.train(learn_images, learn_labels)

    test_accuracy = test_model(model, train_images, train_labels)
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
