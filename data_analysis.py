import os
from PIL import Image
import matplotlib.pyplot as plt
from config import DATA_DIR, PLOTS_DIR


def analyze_dataset():
    class_counts = {}
    image_sizes = []

    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)

            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)

    print(f"Количество классов: {len(class_counts)}")
    print(f"Общее количество изображений: {sum(class_counts.values())}")
    print("Количество изображений на класс:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    unique_sizes = set(image_sizes)
    print(f"Уникальные размеры изображений: {unique_sizes}")

    most_common_size = max(set(image_sizes), key=image_sizes.count)
    print(f"Наиболее часто встречающийся размер: {most_common_size}")

    plot_class_distribution(class_counts)

    return class_counts, most_common_size


def plot_class_distribution(class_counts):
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Distribution of Images per Class')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()


def get_img_size():
    _, most_common_size = analyze_dataset()
    return most_common_size