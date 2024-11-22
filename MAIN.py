import os.path
import random
from PIL import Image
import numpy as np
import time

classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


def relu(x):
    return np.maximum(0, x)


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
                image = Image.open(image_path).resize((64, 64))  # Меняем размер изображения на 64x64
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
        count, h_s, w_s, channels = input_data.shape

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
        count, h, w, c = input_data.shape

        self.input_data = input_data

        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        out = np.zeros((count, h_out, w_out, c))
        for i in range(count):
            for f in range(c):
                for k1 in range(0, h_out):
                    for k2 in range(0, w_out):
                        h_start = k1 * self.stride
                        h_end = h_start + self.pool_size

                        w_start = k2 * self.stride
                        w_end = w_start + self.pool_size

                        out[i, k1, k2, f] = np.max(input_data[i, h_start:h_end, w_start:w_end, f])

        return out

    def backward_prop(self, d_out):
        count = self.input_data.shape[0]
        h_s, w_s, channels = self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3]
        w_e = (h_s - self.pool_size) // self.stride + 1
        h_e = (w_s - self.pool_size) // self.stride + 1

        d_out_reshaped = d_out.reshape(count, h_e, w_e, 16)

        d_input = np.zeros_like(self.input_data)

        for i in range(count):
            for c in range(channels):
                for h in range(h_e):
                    for w in range(w_e):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size

                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        region = self.input_data[i, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(region)

                        d_input[i, h_start:h_end, w_start:w_end, c] += (region == max_val) * d_out_reshaped[i, h, w, c]

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
                 learning_rate=0.01, activation=relu):
        self.conv1 = ConvolutionLayer(filter_size, num_filters)
        self.mpl1 = MaxPoolLayer(pool_size * 2, stride * 2)
        self.conv2 = ConvolutionLayer(filter_size, num_filters, 16)
        self.mpl2 = MaxPoolLayer(pool_size, 1)
        self.fcl = FullConnectedLayer(hidden_size, output_size)  # вот тут надо подкорректировать input и output

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.mpl1.forward(x)
        x = self.conv2.forward(x)
        x = self.mpl2.forward(x)
        x = self.fcl.forward(x)

        return x

    def backward_prop(self, d):
        d = self.fcl.calculate_grads(d)
        d = self.mpl2.backward_prop(d)
        d = self.conv2.backward_prop(d)
        d = self.mpl1.backward_prop(d)
        d = self.conv1.backward_prop(d)

        return d

    def train(self, images, labels, epochs=10, batch_size=29):
        num_samples = images.shape[0]

        with open("Accuracy.txt", "w") as f:
            for epoch in range(epochs):
                time_s = time.time()
                total_loss = 0
                correct_predictions = 0

                # Перемешивание данных в начале каждой эпохи
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                images = images[indices]
                labels = labels[indices]

                # Разделение данных на мини-батчи
                for batch_start in range(0, num_samples, batch_size):
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

                    # Оценка корректных предсказаний
                    now_correct_predictions = np.sum(np.argmax(predictions, axis=1) == label_batch)
                    correct_predictions += now_correct_predictions
                    if now_correct_predictions > correct_predictions:
                        print("УРА ЧТО-ТО СОВПАЛО!!!")

                    # Градиент ошибки
                    grad_loss = predictions - one_hot_labels  # Градиент CrossEntropyLoss

                    # Обратное распространение (backward pass)
                    self.backward_prop(grad_loss)
                    time_e_b = time.time()

                    print(f"Loss: {loss:.4f}, Accuracy: {now_correct_predictions / batch_size:.4%}\n"
                          f"Time: {time_e_b - time_s_b} sec.")

                time_e = time.time()

                # Средняя ошибка и точность за эпоху
                average_loss = total_loss / (num_samples // batch_size)
                accuracy = correct_predictions / num_samples

                # Вывод результатов текущей эпохи
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                      f"Time: {time_e - time_s} sec.")
                f.write(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}\n"
                        f"Time: {time_e - time_s} sec.")


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
    # Need: надо короче переделать по размерам(что бы в общем у нас было не 3x3 в конце, а хотя бы 6x6)
    learn, train = load_data("flower_photos")
    learn_images, learn_labels = learn
    train_images, train_labels = train

    output_size = len(classes)  # рассчитано автоматически
    input_size = 64 * 64  # рассчитать (вроде обычный входной слой)
    hidden_size = 3 * 3 * 16  # рассчитать (из конца в выходной)
    filter_size = 2  # рассчитать (1 - [64x64x3] -> [32x32x3]; 2 - [16x16x3] -> [8x8x3]), 1 - (2), 2 - (2)
    num_filters = 16  # рассчитать (не уверен, но вроде наплевать)
    pool_size = 2  # рассчитать (1 - [32x32x3] -> [16x16x3]; 2 - [8x8x3] -> [4x4x3]), 1 - (2), 2 - (2)
    stride = 2  # рассчитать (2)

    model = NeuralNetwork(input_size, hidden_size, output_size,
                          filter_size, num_filters, pool_size, stride)

    model.train(learn_images, learn_labels)

    test_accuracy = test_model(model, train_images, train_labels)
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
