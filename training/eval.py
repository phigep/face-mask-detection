import tensorflow as tf
import keras
def make_gradcam_heatmap(model, img_array,last_conv_layer_name="top_conv", output_index=0):
    def get_grad_model():
        inputs = model.input
        base = model.get_layer(last_conv_layer_name)
        last_conv_output = base(inputs)
        x = last_conv_output
        for idx in range(3, len(model.layers)):
            x = model.layers[idx](x)
        output = x
        return keras.Model(inputs, [last_conv_output, output])
    grad_model=get_grad_model()
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #print(last_conv_layer_output)
        #print(last_conv_layer_output, preds)
       ##print(preds)
        pred_index = tf.argmax(preds[0])
       ##print(pred_index)
        class_channel = preds[:, pred_index]
       ##print(class_channel)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
   ##print(tf.reduce_max(pooled_grads))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
   ##print(tf.shape(last_conv_layer_output))
    last_conv_layer_output = last_conv_layer_output[0]
   ##print(tf.shape(last_conv_layer_output))
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    
    heatmap = tf.squeeze(heatmap)
    #print(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
   ##print(heatmap.numpy())
    return heatmap.numpy()

import numpy as np
import matplotlib as mpl
from IPython.display import Image, display
from plotnine import (
    ggplot, aes, geom_tile, facet_wrap, theme,
    element_blank, coord_fixed
)
def superimpose_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["viridis"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    return superimposed_img
import matplotlib.pyplot as plt
import math
def plot_images_grid(images, figsize=(10, 10)):
    def compute_grid_dims(n):
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return nrows, ncols
    """
    Displays a list of images in a grid using Matplotlib.
    
    Parameters:
      images: list of images. Each image can be either:
              - a NumPy array of shape (H, W, 3), or
              - a PIL Image.
      nrows: number of rows in the grid.
      ncols: number of columns in the grid.
      figsize: figure size passed to plt.subplots.
    """
    nrows, ncols = compute_grid_dims(len(images))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # Flatten the axes array for convenience.
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for ax, img in zip(axes, images):
        # If the image is a PIL Image, convert it to a NumPy array.
        if hasattr(img, 'convert'):
            img = np.array(img)
        ax.imshow(img)
        ax.axis('off')

    # If there are more subplots than images, hide the extra axes.
    for ax in axes[len(images):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()