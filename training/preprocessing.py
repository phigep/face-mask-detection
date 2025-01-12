import tensorflow as tf
from keras import layers,Sequential

def random_brightness(x):
    
    return tf.image.random_brightness(x, max_delta=0.1)

data_augmentation = Sequential([
    
    layers.Resizing(224, 224),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomContrast(0.1),
    layers.Lambda(random_brightness)
])

def get_augmented_dataset(
    dataset_dir,
    shuffle=True,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224)
):
    """
    Loads and augments an image dataset from a directory.

    Parameters:
      dataset_dir (str): Path to the dataset directory.
      shuffle (bool): Whether to shuffle the dataset.
      labels (str or list): Labels mode; usually 'inferred'.
      label_mode (str): Mode for labels ('categorical', 'binary', or 'int').
      batch_size (int): Batch size.
      image_size (tuple): Size to resize the images (height, width).

    Returns:
      A tf.data.Dataset with the data augmentation pipeline applied.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        shuffle=shuffle,
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size
    )
    augmented_dataset = dataset.map(
        lambda images, labels: (data_augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    augmented_dataset = augmented_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return augmented_dataset