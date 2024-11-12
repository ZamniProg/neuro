import json
import zipfile
import time
from PIL import Image
import os
import numpy as np

learn_image_folder = r"Z:\programming\pythonproj\neuro_1\learn\start\learn_base"
test_image_folder = r"Z:\programming\pythonproj\neuro_1\learn\start\test_base"


def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def unzip(zip_file_path, extract_folder):
    print(f"Разархивация файла {zip_file_path}")

    os.makedirs(extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f'Файл {zip_file_path} успешно разархивирован в {extract_folder}')


def encode_text_labels(labels, max_length, grouping_factor=2):
    unique_chars = sorted(set(''.join(labels)))
    char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
    num_labels = []

    for label in labels:
        # Кодируем каждый символ как число
        label_encoded = [char_to_num[char] for char in label]

        # Группируем элементы, чтобы уменьшить размер последовательности
        if grouping_factor > 1:
            label_encoded = [
                sum(label_encoded[i:i + grouping_factor]) // grouping_factor
                for i in range(0, len(label_encoded), grouping_factor)
            ]

        # Делаем padding до max_length
        label_encoded = label_encoded[:max_length] + [0] * max(0, max_length - len(label_encoded))
        num_labels.append(label_encoded)

    return np.array(num_labels), char_to_num, len(unique_chars)


def decode_text(predicted_output, label_map, grouping_factor=2):
    # Определение предсказанного текста с учетом grouping_factor
    predicted_text = []
    for i in range(0, predicted_output.shape[1], grouping_factor):
        # Группируем прогнозируемые символы и выбираем наиболее вероятный
        group = predicted_output[:, i:i + grouping_factor]
        char_idx = np.argmax(np.sum(group, axis=1))  # Находим индекс наибольшей вероятности в группе
        predicted_text.append(list(label_map.keys())[char_idx])

    return ''.join(predicted_text)


def decode_true_text(targets, label_map, max_length=None):
    # Определение истинного текста
    true_text = ''.join(
        [list(label_map.keys())[idx] for idx in targets if idx != 0]
    )
    # Ограничение максимальной длиной, если необходимо
    if max_length:
        true_text = true_text[:max_length]
    return true_text


def preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert("L")

    width, height = image.size
    resized = width / height

    if width > height:
        new_width = target_size[0]
        new_height = int(target_size[0] / resized)
    elif width < height:
        new_width = int(target_size[1] / resized)
        new_height = target_size[1]
    else:
        new_width, new_height = target_size[0], target_size[1]

    image = image.resize((new_width, new_height), Image.LANCZOS)

    new_image = Image.new("L", target_size, 255)
    new_image.paste(image, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    return np.array(new_image) / 255.0


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def back_prop(self, x):
        grad = np.ones_like(x)
        grad[x <= 0] = self.alpha
        return grad


class RNNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, seq_len, max_text_length, label_map):
        self.hidden_size = hidden_layer
        self.label_map = label_map
        self.input_layer = input_layer
        self.seq_len = seq_len
        self.leakyReLU = LeakyReLU()
        self.max_text_length = max_text_length

        self.WxTh = np.random.randn(hidden_layer, input_layer) * 0.01
        self.WhTh = np.random.randn(hidden_layer, hidden_layer) * 0.01
        self.WhTy = np.random.randn(output_layer, hidden_layer) * 0.01

        self.bh = np.zeros((hidden_layer, 1))
        self.by = np.zeros((output_layer, 1))

    def forward(self, inp):
        h = np.zeros((self.hidden_size, 1))
        hidden_states = []
        x_full = inp.reshape(self.input_layer, 1)

        output_sequence = []
        for _ in range(self.max_text_length):
            for i in range(self.seq_len):
                x = x_full.reshape(-1, 1)
                h = np.tanh(np.dot(self.WxTh, x) + np.dot(self.WhTh, h) + self.bh)
                hidden_states.append(h)

            y = np.dot(self.WhTy, h) + self.by
            output_sequence.append(y)

        return np.hstack(output_sequence), hidden_states

    def compute_grad(self, inp, targets, learning_rate=0.01):
        out_sequence, hidden_states = self.forward(inp)

        loss = np.mean((out_sequence - targets) ** 2)
        grad_out_sequence = 2 * (out_sequence - targets) / out_sequence.shape[1]

        d_WxTh = np.zeros_like(self.WxTh)
        d_WhTh = np.zeros_like(self.WhTh)
        d_WhTy = np.zeros_like(self.WhTy)
        d_by = np.zeros_like(self.by)
        d_bh = np.zeros_like(self.bh)

        d_hide = np.dot(self.WhTy.T, grad_out_sequence[:, -1].reshape(-1, 1))

        for t in reversed(range(self.seq_len)):
            dh_raw = (1 - hidden_states[t] ** 2) * d_hide

            d_WxTh += np.dot(dh_raw, inp[t].reshape(-1, 1))
            d_WhTh += np.dot(dh_raw, hidden_states[t - 1].T) if t > 0 else 0
            d_bh += dh_raw
            d_WhTy += np.dot(grad_out_sequence[:, t].reshape(-1, 1), hidden_states[-1].T)
            d_by += grad_out_sequence[:, t].reshape(-1, 1)

            d_hide = np.dot(self.WhTh, dh_raw)

        self.WxTh -= learning_rate * d_WxTh
        self.WhTh -= learning_rate * d_WhTh
        self.WhTy -= learning_rate * d_WhTy
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

        return loss

    @staticmethod
    def save_gradients(gradients, file_path="gradients.json"):
        filtered_gradients = {
            "d_WxTh": gradients["d_WxTh"].tolist(),
            "d_WhTh": gradients["d_WhTh"].tolist(),
            "d_WhTy": gradients["d_WhTy"].tolist()
        }
        with open(file_path, 'a') as f:
            json.dump(filtered_gradients, f, indent=6)
            f.write(",\n")
        print(f"Градиенты сохранены в {file_path}")

    def train(self, images, labels, epochs=10, learning_rate=0.01):
        all_gradients = []
        batch_size = 32

        for epoch in range(epochs):
            time_start = time.time()
            epoch_loss = 0
            batch_loss = 0
            correct_predictions = 0
            total_images = 0

            for batch_s in range(0, len(images), batch_size):
                t1 = time.time()
                batch_i = images[batch_s:batch_s + batch_size]
                batch_l = labels[batch_s:batch_s + batch_size]

                batch_loss = 0
                for i, image in enumerate(batch_i):
                    print(f"Обработка изображения №{i + 1}.")
                    inputs = preprocess_image(image, (64, 64)).flatten()
                    targets = batch_l[i]

                    batch_loss += self.compute_grad(inputs, targets, learning_rate)
                    total_images += 1

                    predicted_output, _ = self.forward(inputs)

                    predicted_text = decode_text(predicted_output, self.label_map, grouping_factor=2)
                    true_text = decode_true_text(targets, self.label_map, self.max_text_length)

                    if predicted_text.strip() == true_text.strip():
                        correct_predictions += 1

                    if i == batch_size - 1:
                        accuracy = (correct_predictions / total_images) * 100
                        print(f"Обработка {i + 1} изображений - Точность: {accuracy:.2f}%, Потеря: {batch_loss:.4f}")

                        gradients = {
                            "epoch": epoch + 1,
                            "d_WxTh": self.WxTh.tolist(),
                            "d_WhTh": self.WhTh.tolist(),
                            "d_WhTy": self.WhTy.tolist()
                        }
                        all_gradients.append(gradients)
                print(f"время обработки одного батча: {time.time() - t1}")

                epoch_loss += batch_loss / batch_size

            accur = (correct_predictions / total_images) * 100
            print(f"Обработка 50000 изображений в {epoch + 1} эпохе - Точность: {accur:.2f}%, Потеря: {batch_loss:.4f}")

            gradients = {
                "epoch": epoch + 1,
                "d_WxTh": self.WxTh.tolist(),
                "d_WhTh": self.WhTh.tolist(),
                "d_WhTy": self.WhTy.tolist()
            }

            all_gradients.append(gradients)

            print(f"Эпоха {epoch + 1} завершена, средняя потеря за эпоху: {epoch_loss / total_images:.4f}")
            self.save_weights("weights.json")
            time_end = time.time()

            print(f"\nСреднее время обработки 1 изображения: {50000 / (time_end - time_start)}.4f секунды\n")
        with open("gradients_over_epochs.json", 'w') as f:
            json.dump(all_gradients, f, indent=4)
        print("Все градиенты сохранены в gradients_over_epochs.json")

    def load_weights(self, file_path="weights.json"):
        try:
            with open(file_path, 'r') as f:
                weights = json.load(f)
                self.WxTh = np.array(weights["WxTh"])
                self.WhTh = np.array(weights["WhTh"])
                self.WhTy = np.array(weights["WhTy"])
                self.bh = np.array(weights["bh"])
                self.by = np.array(weights["by"])
            print(f"Веса загружены из {file_path}")
        except FileNotFoundError:
            print("Файл с весами не найден. Начало обучения с нуля.")

    def save_weights(self, file_path="weights.json"):
        weights = {
            "WxTh": self.WxTh.tolist(),
            "WhTh": self.WhTh.tolist(),
            "WhTy": self.WhTy.tolist(),
            "bh": self.bh.tolist(),
            "by": self.by.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(weights, f)
        print(f"Промежуточные веса сохранены в {file_path}")


def unzip_all():
    zip_file_path_learn = r"Z:\programming\pythonproj\neuro_1\learn\start\learn_base.zip"
    zip_file_path_test = r"Z:\programming\pythonproj\neuro_1\learn\start\test_base.zip"

    # unzip(zip_file_path_learn, learn_image_folder)
    # unzip(zip_file_path_test, test_image_folder)


def main():
    input_size = 64 * 64
    hidden_size = 32
    seq_length = 100

    train_data = load_json_data(r"Z:\programming\pythonproj\neuro_1\learn\start\jsons\learn_images_data.json")
    test_data = load_json_data(r"Z:\programming\pythonproj\neuro_1\learn\start\jsons\test_images_data.json")

    max_text_length = max(len(item["text"]) for item in train_data)
    numeric_labels, label_map, output_size = encode_text_labels([item["text"] for item in train_data], max_text_length)

    print(f"{input_size} : {hidden_size} : {output_size} : {seq_length} : {max_text_length}")

    model = RNNetwork(input_size, hidden_size, output_size, seq_length, max_text_length, label_map)

    model.load_weights("weights.json")

    model.train(
        [f"{learn_image_folder}/{item['image_id']}" for item in train_data],
        numeric_labels,
        epochs=10
    )
    model.save_weights("weights.json")

    correct_predictions = 0
    for item in test_data:
        image = preprocess_image(f"{test_image_folder}/{item['image_id']}")
        predicted_output, _ = model.forward(image.flatten())
        predicted_text = ''.join([list(label_map.keys())[np.argmax(char)] for char in predicted_output.T])

        if predicted_text.strip() == item["text"].strip():
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Точность на тестовом наборе: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
