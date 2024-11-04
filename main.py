import sys
from PyQt5.QtWidgets import QApplication
from PIL import Image
from attention_generator import AttentionMapGenerator
from visualizer import AttentionVisualizerApp

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load an example image
    image_path = "input/1.jpeg"  # Replace with your own image path
    image = Image.open(image_path)

    # Generate synthetic attention data
    H, W = image.size[1], image.size[0]
    N = 10  # Number of time steps
    attention_generator = AttentionMapGenerator(H, W)
    attention_data = attention_generator.generate_attention_sequence(N)

    # Initialize and run the application
    window = AttentionVisualizerApp(image, attention_data)
    window.resize(600, 600)
    window.show()
    sys.exit(app.exec_())
