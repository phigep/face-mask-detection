from keras.applications import EfficientNetV2B0, EfficientNetV2B3
from dataclasses import dataclass
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda,Resizing
from keras.optimizers import AdamW
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataclasses import dataclass

############# Simple function tog et normal cnn model
def get_model(base_model, input_dim=(),resize_dim=(), classes=3, classifier_activation="softmax"):
    inputs = Input(shape=input_dim)
    resized_inputs = Resizing(height=resize_dim[0], width=resize_dim[1])(inputs)
    
    base = base_model(resized_inputs, training=False)

    x = GlobalAveragePooling2D()(base)
    x = Dropout(rate=0.3)(x)
    outputs = Dense(classes, activation=classifier_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

effnetv2b0_base = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,
        )

effnetv2b3_base = EfficientNetV2B3(
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,
        )

import os

def train_model(model, train_data, val_data, checkpoint_dir, epochs=50):
    """
    Trains the given model using provided training and validation data,
    and saves the best model checkpoint.

    Parameters:
      model         : Keras model to be trained.
      train_data    : Training data (e.g., tf.data.Dataset).
      val_data      : Validation data (e.g., tf.data.Dataset).
      checkpoint_dir: Directory where the best model checkpoint will be saved.
      epochs        : Number of epochs for training.
    
    Returns:
      history : Training history object from model.fit()
    """

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}.h5')
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    return model,history