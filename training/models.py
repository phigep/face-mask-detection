from keras.applications import EfficientNetV2B0, EfficientNetV2B3
from dataclasses import dataclass
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda,Resizing, GlobalMaxPooling2D
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