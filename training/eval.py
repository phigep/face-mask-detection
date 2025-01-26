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
def superimpose_gradcam(img, heatmap, cam_path=None, alpha=0.4):
    """
    Superimposes the Grad-CAM heatmap onto the original image.

    Args:
        img (np.array or PIL.Image): Original image array.
        heatmap (np.array): Grad-CAM heatmap.
        cam_path (str, optional): Path to save the superimposed image. Defaults to "cam.jpg".
        alpha (float, optional): Transparency factor for the heatmap. Defaults to 0.4.

    Returns:
        PIL.Image.Image: Image with heatmap superimposed.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use 'magma' colormap to colorize heatmap
    magma = mpl.colormaps["magma"]

    # Use RGB values of the colormap
    magma_colors = magma(np.arange(256))[:, :3]
    magma_heatmap = magma_colors[heatmap]

    # Create an image with RGB colorized heatmap
    magma_heatmap = keras.utils.array_to_img(magma_heatmap)
    magma_heatmap = magma_heatmap.resize((img.shape[1], img.shape[0]))
    magma_heatmap = keras.utils.img_to_array(magma_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = magma_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Optionally save the image
    if cam_path:
        superimposed_img.save(cam_path)

    return superimposed_img

    return superimposed_img
import matplotlib.pyplot as plt
import math

def plot_images_grid(images, figsize=(10, 10), captions=None, font_size=12):
    """
    Displays a list of images in a grid using Matplotlib, with optional captions beneath each image.
    
    Parameters:
        images (list): List of images. Each image can be either:
                       - a NumPy array of shape (H, W, 3), or
                       - a PIL Image.
        figsize (tuple, optional): Size of the entire figure. Defaults to (10, 10).
        captions (list of str, optional): List of captions for each image. Must be the same length as images.
        font_size (int, optional): Font size for the captions. Defaults to 12.
        
    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the grid of images.
    """
    if captions is not None and len(captions) != len(images):
        raise ValueError("Length of captions must match the number of images.")
    
    def compute_grid_dims(n):
        """
        Computes the number of rows and columns for the image grid based on the number of images.
        
        Args:
            n (int): Number of images.
            
        Returns:
            tuple: Number of rows and columns (nrows, ncols).
        """
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return nrows, ncols

    nrows, ncols = compute_grid_dims(len(images))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten the axes array for easy iteration. If there's only one subplot, make it iterable.
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]
            # If the image is a PIL Image, convert it to a NumPy array.
            if hasattr(img, 'convert'):
                img = np.array(img)
            ax.imshow(img)
            ax.axis('off')
            
            # Add caption beneath the image if captions are provided
            if captions is not None:
                # Adjust the position of the caption
                ax.set_title(captions[idx], fontsize=font_size, pad=10)
        else:
            ax.axis('off')  # Hide axes without images
    
    plt.tight_layout()
    plt.show()
    return fig