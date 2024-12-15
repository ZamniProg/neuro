import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Путь к папке с сохраненными .npz файлами
save_dir = "saves"
output_dir = "layer_plots"
os.makedirs(output_dir, exist_ok=True)

# Список файлов в папке
files = [f for f in os.listdir(save_dir) if f.endswith(".npz")]

# Словарь для хранения данных по слоям
data_by_layer = {}

# Считываем данные из файлов
for file in files:
    file_path = os.path.join(save_dir, file)

    # Получаем имя слоя и эпоху
    layer_name = file.split("_history_")[0]
    if len(file.split("_history_")) == 1:
        continue
    epoch = int(file.split("_history_")[1].split(".npz")[0])

    # Загружаем веса и градиенты
    data = np.load(file_path)
    weights = data.get("weights")
    gradients = data.get("gradients")

    if layer_name not in data_by_layer:
        data_by_layer[layer_name] = {"epochs": [], "weight_norms": [], "gradient_norms": []}

    # Норма весов и градиентов
    weight_norm = np.linalg.norm(weights) if weights is not None else 0
    gradient_norm = np.linalg.norm(gradients) if gradients is not None else 0

    # Добавляем данные
    data_by_layer[layer_name]["epochs"].append(epoch)
    data_by_layer[layer_name]["weight_norms"].append(weight_norm)
    data_by_layer[layer_name]["gradient_norms"].append(gradient_norm)

# Параметры слоёв fcl1 и fcl2
layer_1_name = "fcl1"
layer_2_name = "fcl2"

# Данные для построения (выбираем слои fcl1 и fcl2)
weights_fcl1 = np.array(data_by_layer[layer_1_name]["weight_norms"])  # Веса для fcl1
weights_fcl2 = np.array(data_by_layer[layer_2_name]["weight_norms"])  # Веса для fcl2

losses = [1.6142,
1.4600,
1.3326,
1.3497,
1.1928,
1.1378,
1.1843,
1.0816,
1.0882,
1.0805,
0.9811,
1.3311,
0.9381,
0.9936,
0.9859,
0.9477,
1.0729,
0.9761,
0.8326,
0.7500]

loss_values = np.array(losses)
# Генерация сетки для рельефной тепловой карты
X, Y = np.meshgrid(np.linspace(min(weights_fcl1), max(weights_fcl1), 100),
                   np.linspace(min(weights_fcl2), max(weights_fcl2), 100))

# Интерполяция для более гладкой поверхности
Z = griddata((weights_fcl1, weights_fcl2), loss_values, (X, Y), method='cubic')

# Построение 3D модели с тепловым эффектом
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности с улучшением визуализации
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Добавление цветовой шкалы
fig.colorbar(surf, label="Loss Values")

# Настройка подписей осей
ax.set_title("3D Heatmap with Relief for Layers fcl1 and fcl2")
ax.set_xlabel("Weight Norms of fcl1")
ax.set_ylabel("Weight Norms of fcl2")
ax.set_zlabel("Loss Values")

output_file = os.path.join(output_dir, "grad_3Dmodel.png")
# Показ графика
plt.savefig(output_file)
plt.close()

# Построение отдельных графиков для весов и градиентов
for layer_name, metrics in data_by_layer.items():
    # Сортировка данных по эпохам
    sorted_indices = np.argsort(metrics["epochs"])
    epochs = np.array(metrics["epochs"])[sorted_indices]
    weight_norms = np.array(metrics["weight_norms"])[sorted_indices]
    gradient_norms = np.array(metrics["gradient_norms"])[sorted_indices]

    # График норм весов
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, weight_norms, label="Weight Norms", marker="o", color="blue")
    plt.title(f"Weight Norms for Layer: {layer_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Values")
    plt.legend()
    plt.grid()
    output_file = os.path.join(output_dir, f"{layer_name}_weight_norms.png")
    plt.savefig(output_file)
    plt.close()
    print(f"График весов сохранён в {output_file}")

    # График норм градиентов
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gradient_norms, label="Gradient Norms", marker="x", color="red")
    plt.title(f"Gradient Norms for Layer: {layer_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Values")
    plt.legend()
    plt.grid()
    output_file = os.path.join(output_dir, f"{layer_name}_gradient_norms.png")
    plt.savefig(output_file)
    plt.close()
    print(f"График градиентов сохранён в {output_file}")


