from keras.applications import EfficientNetV2B0, EfficientNetV2B3
from dataclasses import dataclass
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda,Resizing
from keras.optimizers import AdamW
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
from dataclasses import dataclass


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


if __name__ == "__main__":
    base_model = effnetv2b3_base
    input_dimensions = (32, 32, 3)  # Adjusted for CIFAR-10 dataset
    resized_shape = (224, 224, 3) 
    num_classes = 10  # CIFAR-10 has 10 classes

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images and one-hot encode the labels
    #x_train = x_train.astype("float32") / 255.0
    #x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Get the model
    model = get_model(base_model, input_dim=input_dimensions,resize_dim=resized_shape, classes=num_classes, classifier_activation="softmax", loss="categorical_crossentropy")

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
