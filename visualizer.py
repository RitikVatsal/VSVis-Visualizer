import numpy as np
from PyQt5.QtWidgets import QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from utils import highlight_area, numpy_to_qimage

class AttentionVisualizerApp(QMainWindow):
    def __init__(self, image_data, attention_data):
        super().__init__()
        self.setWindowTitle("AI Attention Visualizer")
        self.time_steps = attention_data.shape[2]
        self.original_image_data = np.array(image_data.convert("RGBA"))
        self.attention_data = attention_data

        # Set up the UI layout
        self.setup_ui()

    def setup_ui(self):
        """Initialize and configure the UI components."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image display label
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Time step slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.time_steps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.slider)

        # Initial display update
        self.update_image(0)
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        """Handle window resizing and update the image display."""
        self.update_image(self.slider.value())
    
    def compute_highlighted_image(self, time_step, new_size):
        """Generate and resize the highlighted image for the specified time step."""
        attention_map = np.array(Image.fromarray((self.attention_data[:, :, time_step] * 255).astype(np.uint8)).resize(new_size)) / 255.0
        resized_image = np.array(Image.fromarray(self.original_image_data).resize(new_size))
        highlighted_image = highlight_area(resized_image, attention_map)

        return QPixmap.fromImage(numpy_to_qimage(highlighted_image))

    def update_image(self, time_step):
        """Update the displayed image based on the slider's time step position."""
        label_size = self.image_label.size()
        new_size = (label_size.width(), label_size.height())
        pixmap = self.compute_highlighted_image(time_step, new_size)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
