import os
import pickle
import threading
import queue
from data_analysis import analyze_dataset, get_img_size
from data_preparation import create_data_generators
from models import create_model
from training import train_model
from visualization import plot_history, plot_learning_curves
from config import RESULTS_DIR, MODELS_DIR


def user_input(prompt, timeout):
    user_response = queue.Queue()

    def get_input():
        try:
            user_response.put(input(prompt))
        except EOFError:
            user_response.put(None)

    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()

    try:
        response = user_response.get(timeout=timeout)
    except queue.Empty:
        response = None

    return response


def train_and_save_model(name, size, use_regularization, input_shape, num_classes, train_generator,
                         validation_generator):
    print(f"Creating {name} model...")
    model = create_model(input_shape, num_classes, size, use_regularization)

    print(f"Training {name} model...")
    history = train_model(model, train_generator, validation_generator, f'{name}_model')

    print(f"Saving {name} model history...")
    with open(os.path.join(RESULTS_DIR, f'{name}_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Plotting {name} model learning curves...")
    plot_learning_curves(history, f'{name} Model')

    print(f"{name} model training complete.\n")
    return history


def main():
    # Анализ датасета и определение размера изображений
    class_counts, img_size = analyze_dataset()
    print(f"Используемый размер изображений: {img_size}")

    # Подготовка данных
    train_generator, validation_generator = create_data_generators(img_size)

    # Создание и обучение моделей
    input_shape = img_size + (3,)
    num_classes = len(train_generator.class_indices)

    models = [
        ('small', 'small', True),
        ('medium', 'medium', True),
        ('large', 'large', True),
        ('no_reg', 'medium', False)
    ]

    histories = {}

    for name, size, use_regularization in models:
        history = train_and_save_model(name, size, use_regularization, input_shape, num_classes, train_generator,
                                       validation_generator)
        histories[name] = history.history

        user_response = user_input(
            f"Model {name} training complete. Press Enter to continue with the next model, or type 'q' to quit (Автоматическое продолжение через 30 минут): ",
            timeout=1800)  # 1800 секунд = 30 минут
        if user_response and user_response.lower() == 'q':
            break
        else:
            print("Continuing with the next model...")

    # Сохранение общей истории обучения, если все модели были обучены
    if len(histories) == len(models):
        print("Saving overall training history...")
        with open(os.path.join(RESULTS_DIR, 'training_history.pkl'), 'wb') as f:
            pickle.dump(histories, f)

        # Визуализация результатов всех моделей
        print("Plotting overall training history...")
        plot_history(list(histories.values()), list(histories.keys()))

    print("Training complete. Models saved in:", MODELS_DIR)
    print("Training history saved in:", RESULTS_DIR)
    print("Training plots saved in:", os.path.join(RESULTS_DIR, 'plots'))


if __name__ == "__main__":
    main()