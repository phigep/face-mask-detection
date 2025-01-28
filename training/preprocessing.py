import tensorflow as tf
from keras import layers,Sequential

def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.3)  # Stronger brightness adjustment


def random_crop(x):
    """Randomly crop the image to simulate occlusion or different view angles."""
    crop_height = tf.random.uniform((), minval=180, maxval=224, dtype=tf.int32)
    crop_width = tf.random.uniform((), minval=180, maxval=224, dtype=tf.int32)
    return tf.image.random_crop(x, size=[crop_height, crop_width, 3])

def random_color_jitter(x):
    """Adjust hue, saturation, contrast, and brightness randomly."""
    x = tf.image.random_hue(x, 0.2)
    x = tf.image.random_saturation(x, 0.5, 1.5)
    return x

data_augmentation = Sequential([
    layers.Resizing(224, 224),
    layers.RandomFlip("horizontal"),  # Strong flip probability
    #layers.RandomFlip("vertical"),  # Add vertical flipping for robustness
    layers.RandomRotation(0.2),  # Stronger rotation to simulate different angles
    layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),  # Increased translation
    layers.RandomContrast(0.4),  # Strong contrast changes
    layers.Lambda(random_brightness),
    #layers.Lambda(lambda x: random_blur(x, kernel_size=7)),  # Stronger blur effect
    #layers.Lambda(random_crop),  # Simulate occlusions by random cropping
    #layers.Lambda(random_color_jitter),  # Simulate lighting changes
    layers.RandomShear(0.2),  # Add shear to simulate perspective distortion
    layers.GaussianNoise(0.3)  # Simulate noise in low-quality images
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