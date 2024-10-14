import os

# Основные настройки
DATA_DIR = 'AID'
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Пути для сохранения результатов
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Создаем директории, если они не существуют
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# IMG_SIZE будет определен позже в data_analysis.py
IMG_SIZE = None