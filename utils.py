import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage

def highlight_area(image_data, attention_map):
    """Overlay the attention map on the image data."""
    height, width = attention_map.shape
    highlighted_image = np.array(image_data, copy=True)

    # Create an RGBA overlay with transparency based on attention map
    overlay = np.zeros_like(image_data, dtype=np.uint8)
    overlay[..., :3] = [255, 255, 0]  # Yellow for attention
    overlay[..., 3] = (attention_map * 255).astype(np.uint8)

    # Blend overlay with the image data where attention is non-zero
    highlighted_image = np.where(overlay[..., 3:] > 0, overlay, highlighted_image)
    return highlighted_image

def numpy_to_qimage(numpy_image):
    """Convert a NumPy RGBA image array to QImage for display in PyQt."""
    height, width, channel = numpy_image.shape
    bytes_per_line = channel * width
    return QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
