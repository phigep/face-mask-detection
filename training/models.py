from keras.applications import EfficientNetV2B0, EfficientNetV2B3
from dataclasses import dataclass
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda,Resizing
from keras.optimizers import AdamW
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
from dataclasses import dataclass

############# Simple function tog et normal cnn model
def get_model(base_model, input_dim=(),resize_dim=(), classes=3, loss=None, classifier_activation="softmax"):
    inputs = Input(shape=input_dim)
    resized_inputs = Resizing(height=resize_dim[0], width=resize_dim[1])(inputs)
    
    base = base_model(resized_inputs, training=False)

    x = GlobalAveragePooling2D()(base)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=512,activation="relu")(x)
    x = Dropout(rate=0.3)(x)
    outputs = Dense(classes, activation=classifier_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)

    if loss:
        model.compile(optimizer="adamw", loss=loss, metrics=["accuracy"])

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



############ Encoder for SIMCLR

