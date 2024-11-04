import sys
import numpy as np
from scipy.ndimage import gaussian_filter  # For realistic blur effect
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

def generate_realistic_attention_map(height, width, num_hotspots=3, blur_sigma=15):
    """Generate a realistic attention map with concentrated hotspots."""
    attention_map = np.zeros((height, width), dtype=np.float32)

    # Create several hotspots randomly located in the attention map
    for _ in range(num_hotspots):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        attention_map[y, x] = 1  # Set a high value at the hotspot

    # Apply Gaussian blur to make the attention look smoother and more realistic
    attention_map = gaussian_filter(attention_map, sigma=blur_sigma)

    # Normalize the attention map to keep values between 0 and 1
    attention_map = attention_map / attention_map.max()
    return attention_map

def highlight_area(image_data, attention_map):
    """Apply an attention map to the image data to highlight regions of focus."""
    height, width = attention_map.shape
    highlighted_image = np.array(image_data, copy=True)

    # Create an RGBA overlay with yellow color and transparency based on attention map
    overlay = np.zeros_like(image_data, dtype=np.uint8)
    overlay[..., :3] = [255, 255, 0]  # Yellow for attention
    overlay[..., 3] = (attention_map * 255).astype(np.uint8)  # Alpha channel based on attention map

    # Use np.where to blend overlay and image_data where overlay has transparency
    highlighted_image = np.where(overlay[..., 3:] > 0, overlay, highlighted_image)
    return highlighted_image

class AttentionVisualizerApp(QMainWindow):
    def __init__(self, image_data, attention_data):
        super().__init__()
        self.setWindowTitle("AI Attention Visualizer")
        self.time_steps = attention_data.shape[2]
        self.original_image_data = np.array(image_data.convert("RGBA"))  # Store original for resizing
        self.attention_data = attention_data  # Assign attention_data to the instance

        # Set up central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image label for displaying images
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Slider for controlling time steps
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.time_steps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.slider)

        # Display the initial image
        self.update_image(0)

        # Resize event
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        """Handle window resize event and rescale the image accordingly."""
        self.update_image(self.slider.value())
    
    def compute_highlighted_image(self, time_step, new_size):
        """Compute the highlighted image at the given time step and resize it to `new_size`."""
        attention_map = np.array(Image.fromarray((self.attention_data[:, :, time_step] * 255).astype(np.uint8)).resize(new_size)) / 255.0
        resized_image = np.array(Image.fromarray(self.original_image_data).resize(new_size))
        highlighted_image = highlight_area(resized_image, attention_map)

        # Convert numpy array to QImage for QLabel
        height, width, channel = highlighted_image.shape
        bytes_per_line = channel * width
        q_image = QImage(highlighted_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(q_image)

    def update_image(self, time_step):
        """Update displayed image based on the slider position and current window size."""
        # Get the new size based on the QLabel dimensions
        label_size = self.image_label.size()
        new_size = (label_size.width(), label_size.height())
        pixmap = self.compute_highlighted_image(time_step, new_size)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  # Ensures QLabel scales the image to fit

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load an example image (replace this with your own image path)
    image_path = "input/1.jpeg"  # Replace with your image path
    image = Image.open(image_path)

    # Create synthetic attention data for demonstration with realistic attention hotspots
    H, W = image.size[1], image.size[0]
    N = 10  # Number of time steps
    attention_data = np.stack([generate_realistic_attention_map(H, W) for _ in range(N)], axis=2)

    # Initialize and run the application
    window = AttentionVisualizerApp(image, attention_data)
    window.resize(600, 600)  # Set initial window size
    window.show()
    sys.exit(app.exec_())
