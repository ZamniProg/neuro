import os
import random
import numpy as np
from PIL import Image

# Классы цветов
classes = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}


# Функция для загрузки данных
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
                image = Image.open(image_path).resize((128, 128))  # Меняем размер изображения на 128x128
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


# Активирующие функции
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Численно устойчиво
    return exps / np.sum(exps, axis=1, keepdims=True)


# Функция потерь (кросс-энтропия)
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m


# Производная потерь по выходу сети
def cross_entropy_derivative(y_pred, y_true):
    m = y_true.shape[0]
    grad = y_pred
    grad[range(m), y_true] -= 1
    return grad / m


# Инициализация весов
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)  # Для воспроизводимости
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# Прямой проход
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Обратный проход
def backward_pass(X, Y, Z1, A1, A2, W1, W2):
    m = X.shape[0]

    dZ2 = cross_entropy_derivative(A2, Y)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# Обновление параметров
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


# Обучение
def train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        # Прямой проход
        Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

        # Потери
        loss = cross_entropy_loss(A2, Y)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        # Обратный проход
        dW1, db1, dW2, db2 = backward_pass(X, Y, Z1, A1, A2, W1, W2)

        # Обновление весов
        W1, b1, W2, b2 = update_weights(W1, b1, W2, db2, dW1, db1, dW2, db2, learning_rate)
    return W1, b1, W2, b2


# Предсказание
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)


# Загружаем данные
data_path = "flower_photos"  # Путь к папке с изображениями
(learn_images, learn_labels), (train_images, train_labels) = load_data(data_path)

# Преобразуем изображения в формат, подходящий для входа нейросети
input_size = 128 * 128 * 3  # Размер входного слоя: 128x128, 3 канала (RGB)
learn_images = learn_images.reshape(len(learn_images), -1)
train_images = train_images.reshape(len(train_images), -1)

# Гиперпараметры
hidden_size = 16  # Количество нейронов в скрытом слое
output_size = 5   # Количество классов
epochs = 3306
learning_rate = 0.1

# Функции для обучения и тестирования (без изменений)
# Используются функции из предыдущего примера: initialize_weights, train, predict и др.

# Обучение модели
W1, b1, W2, b2 = train(train_images, train_labels, input_size, hidden_size, output_size, epochs, learning_rate)

# Тестирование на learn-наборе
predictions = predict(learn_images, W1, b1, W2, b2)
accuracy = np.mean(predictions == learn_labels) * 100
print(f"Accuracy on learn set: {accuracy:.2f}%")
