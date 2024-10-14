from tensorflow.keras.callbacks import ModelCheckpoint
from config import MODELS_DIR, NUM_EPOCHS
import os


def train_model(model, train_generator, validation_generator, model_name):
    checkpoint_path = os.path.join(MODELS_DIR, f'{model_name}.h5')
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 monitor='val_accuracy',
                                 mode='max',
                                 verbose=1)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )
    return history