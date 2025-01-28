from keras.applications import EfficientNetV2B0, EfficientNetV2B3
from dataclasses import dataclass
from keras.models import Model, Sequential
from keras.layers import Rescaling,BatchNormalization,MaxPooling2D,Add,Conv2D,ReLU,DepthwiseConv2D, Dense, GlobalAveragePooling2D, Input, Dropout, Lambda,Resizing, GlobalMaxPooling2D
from keras.optimizers import AdamW
from keras.datasets import cifar10
from keras.losses import Loss
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dataclasses import dataclass
import tensorflow as tf
import os






class SupervisedContrastiveLoss(Loss):
    def __init__(self, temperature=1.0, name="supervised_contrastive_loss"):
        """
        Args:
            temperature: Temperature scalar that scales the logits.
            name: Optional name for the loss.
        """
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, labels, feature_vectors):
        """
        Computes the supervised contrastive loss.
        
        Args:
            labels: A tensor of shape (batch_size, num_classes) with one-hot encoded labels.
            feature_vectors: A tensor of shape (batch_size, dim) containing embeddings.
        
        Returns:
            Scalar loss.
        """
        # 1. Normalize the feature vectors along the feature axis.
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        
        # 2. Compute the similarity (logits) matrix.
        # Each element [i, j] is the dot-product between normalized vectors i and j, scaled by temperature.
        logits = tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized))
        logits = logits / self.temperature
        
        # 3. Create a mask to remove self-comparisons.
        # We use an identity matrix to zero-out (or later ignore) self-similarities.
        batch_size = tf.shape(logits)[0]
        logits_mask = 1 - tf.eye(batch_size)
        
        # 4. Build the positive mask from the one-hot labels.
        # For one-hot labels, the matrix multiplication gives 1 when two examples share the same class.
        positive_mask = tf.matmul(labels, tf.transpose(labels))
        # Exclude self-comparisons.
        positive_mask = positive_mask - tf.eye(batch_size)
        
        # 5. For numerical stability, subtract the maximum logit from each row.
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - logits_max
        
        # 6. Exponentiate the adjusted logits and apply the logits_mask (to zero out self-similarity terms).
        exp_logits = tf.exp(logits) * logits_mask
        
        # 7. Compute the denominator, which is the sum over all non-self examples.
        denominator = tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8  # avoid division by zero
        
        # 8. Compute the log-probabilities.
        log_prob = logits - tf.math.log(denominator)
        
        # 9. For every anchor, keep only the log-probabilities corresponding to the positives.
        #    Then, average over the number of positives.
        sum_log_prob_pos = tf.reduce_sum(positive_mask * log_prob, axis=1)
        # Count the number of positive examples per anchor.
        positive_count = tf.reduce_sum(positive_mask, axis=1) + 1e-8
        
        # 10. Compute the loss per sample.
        loss_per_sample = - sum_log_prob_pos / positive_count
        
        # 11. Average the loss over the batch.
        loss = tf.reduce_mean(loss_per_sample)
        return loss



def get_supervised_contrastive_loss(temperature=0.05):
    return SupervisedContrastiveLoss(temperature=temperature)


############# Simple function tog et normal cnn model
def get_model(base_model, input_dim=(),resize_dim=(), classes=3, classifier_activation="softmax",training_base=False):
    
    inputs = Input(shape=input_dim)
    base_model.input_tensor = inputs
    resized_inputs = Resizing(height=resize_dim[0], width=resize_dim[1])(inputs)
    
    base = base_model(resized_inputs, training=training_base)
    x = GlobalAveragePooling2D()(base)
    x = Dropout(rate=0.3)(x)
    outputs = Dense(classes, activation=classifier_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_model_(base_model, input_dim=(),resize_dim=(), classes=3, classifier_activation="softmax",training_base=False):
    
    inputs = Input(shape=input_dim)
    base_model.input_tensor = inputs
    resized_inputs = Resizing(height=resize_dim[0], width=resize_dim[1])(inputs)
    
    base = base_model(resized_inputs)
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

def create_simple_base(input_shape=(224, 224, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(1./255),
        Conv2D(16, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same'),
        BatchNormalization(),
        ReLU()
    ])
    return model

def create_minimal_base(input_shape=(224, 224, 3)):
    model = Sequential([
            ReLU(),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same'),
            BatchNormalization(),
            ReLU()
    ])
    return model

simple_base = create_simple_base()


def get_fcn_model(base_model, input_dim=(),resize_dim=(), classes=3, pooling_operation="average"):
    
    inputs = Input(shape=input_dim)
    base_model.input_tensor = inputs
    resized_inputs = Resizing(height=resize_dim[0], width=resize_dim[1])(inputs)
    
    base = base_model(resized_inputs, training=False) # to true if worse
    if pooling_operation=="average":
        x = GlobalAveragePooling2D()(base)
    #elif pooling_operation=="max":
    #    x = GlobalMaxPooling2D()(base)
    x= Dense(128,activation="relu")(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def train_model(model, train_data, val_data, checkpoint_dir, epochs=20,modelname="effnetv2b0"):
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
    checkpoint_path = os.path.join(checkpoint_dir, f'{modelname}.keras')
    
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




def mbconv_block(x, expand_filters, out_filters, kernel_size=3, strides=1):
    """A simplified MBConv (Mobile Inverted Bottleneck) block."""
    inp = x
    x = Conv2D(expand_filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x) 

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                        padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    
    x = Conv2D(out_filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    if strides == 1 and inp.shape[-1] == out_filters:
        x = Add()([inp, x])
    
    return x

def create_efficient_base(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    # Scale inputs to [0, 1]
    x = Rescaling(1.0 / 255)(inputs)
    
    # Stem: a simple conv to reduce spatial size early
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    x = mbconv_block(x, expand_filters=32, out_filters=16, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=96, out_filters=24, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_filters=144, out_filters=24, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=144, out_filters=40, kernel_size=5, strides=2)
    x = mbconv_block(x, expand_filters=240, out_filters=40, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_filters=240, out_filters=80, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_filters=480, out_filters=80, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=480, out_filters=80, kernel_size=3, strides=1)
    x = Conv2D(128, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    model = Model(inputs, x)
    return model



def create_efficient_base_deeper(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    # Scale inputs to [0, 1]
    x = Rescaling(1.0 / 255)(inputs)
    
    # Stem: a simple conv to reduce spatial size early
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    x = mbconv_block(x, expand_filters=32, out_filters=16, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=96, out_filters=24, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_filters=144, out_filters=24, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=144, out_filters=40, kernel_size=5, strides=2)
    x = mbconv_block(x, expand_filters=240, out_filters=40, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_filters=240, out_filters=80, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_filters=480, out_filters=80, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_filters=480, out_filters=80, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_filters=576, out_filters=96, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_filters=576, out_filters=96, kernel_size=5, strides=1)
    x = Conv2D(512, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    
    # This model ends with a 4D tensor (feature map), no classification head.
    model = Model(inputs, x)
    return model



mbconv_base = create_efficient_base()
mbconv_base_deeper = create_efficient_base_deeper()



def combine_batches(features, labels):
    if features.shape[0] != labels.shape[0]:
        raise ValueError("The number of batches in features and labels must be the same.")
    
    num_features = features.shape[-1]
    X = features.reshape(-1, num_features)
    num_classes = labels.shape[-1]
    labels_reshaped = labels.reshape(-1, num_classes)
    y = np.argmax(labels_reshaped, axis=1)
    
    return X, y


from dataclasses import dataclass
from sklearn.base import BaseEstimator
import numpy as np
from typing import Any

@dataclass
class HybridModel():
    fcn: Model
    clf: BaseEstimator
    train_dataset: tf.data.Dataset
    validation_dataset:tf.data.Dataset
    test_dataset:tf.data.Dataset
    X_train: Any = None
    X_val: Any = None
    X_test: Any = None
    y_train: Any = None
    y_val: Any = None
    y_train: Any = None
    def prepare_x_y_pairs(self, dataset: tf.data.Dataset):

        features = []
        labels = []
        
        for batch_inputs, batch_labels in dataset:
            # Extract features using the FCN
            batch_features = self.fcn.predict(batch_inputs)
            features.append(batch_features)
            labels.append(batch_labels.numpy())  # Convert TensorFlow tensors to NumPy arrays

        # Convert lists to NumPy arrays
        features = np.array(features[:-1])
        labels = np.array(labels[:-1])
        
        # Combine batches into X and y using the provided function
        X, y = combine_batches(features, labels)
        
        return X, y
    
    def __post_init__(self):
        """
        Post-initialization processing to prepare training, validation, and test data.
        """
        self.X_train, self.y_train = self.prepare_x_y_pairs(self.train_dataset)
        self.X_val, self.y_val = self.prepare_x_y_pairs(self.validation_dataset)
        self.X_test, self.y_test = self.prepare_x_y_pairs(self.test_dataset)
    