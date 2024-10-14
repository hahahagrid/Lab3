from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

import tensorflow as tf
if tf.test.is_built_with_cuda(): from tensorflow.keras.optimizers import Adam
else: from tensorflow.keras.optimizers.legacy import Adam


def create_model(input_shape, num_classes, size='small', use_regularization=True):
    model = Sequential()

    if size == 'small':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
    elif size == 'medium':
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
    elif size == 'large':
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        if use_regularization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if use_regularization:
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model