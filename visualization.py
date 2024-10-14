import matplotlib.pyplot as plt
from config import PLOTS_DIR
import os


def plot_history(histories, labels):
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    for history, label in zip(histories, labels):
        plt.plot(history.history['accuracy'], label=f'{label} (train)')
        plt.plot(history.history['val_accuracy'], label=f'{label} (val)')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(122)
    for history, label in zip(histories, labels):
        plt.plot(history.history['loss'], label=f'{label} (train)')
        plt.plot(history.history['val_loss'], label=f'{label} (val)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_history.png'))
    plt.close()


def plot_learning_curves(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_learning_curves.png'))
    plt.close()