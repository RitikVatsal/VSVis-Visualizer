import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Frame, Canvas, Scale


def highlight_area(image_path, highlight_array):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    data = np.array(image)

    # Ensure the highlight_array matches the image dimensions
    height, width = data.shape[0:2]
    assert highlight_array.shape == (height, width), "Highlight array must match image dimensions."

    # Create an empty array for the highlighted image
    highlighted_image = np.zeros_like(data)

    # Apply the highlight
    for y in range(height):
        for x in range(width):
            alpha_value = int(highlight_array[y, x] * 127)  # Scale to 0-127 for transparency
            if alpha_value > 0:  # If there's any highlight
                highlighted_image[y, x] = [255, 255, 0, alpha_value]  # RGBA for yellow
            else:
                highlighted_image[y, x] = data[y, x]  # Keep original pixel

    # Convert back to an image
    highlighted_image = Image.fromarray(highlighted_image, "RGBA")
    return highlighted_image

def generate_highlight_array(image_size):
    # Generate a highlight array based on image size with random values between 0 and 1
    height, width = image_size
    return np.random.rand(height, width)  # Random values between 0 and 1

class HeatmapApp:
    def __init__(self, master, image_path):
        self.master = master
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGBA")
        self.image_size = self.image.size[::-1]  # (height, width)

        self.highlight_arrays = [generate_highlight_array(self.image_size) for _ in range(3)]
        self.current_index = 0

        self.canvas = Canvas(master, width=self.image.size[0], height=self.image.size[1])
        self.canvas.pack()

        self.slider = Scale(master, from_=0, to=2, orient=tk.HORIZONTAL, command=self.update_heatmap)
        self.slider.pack()

        self.update_heatmap(0)  # Initialize with the first heatmap

    def update_heatmap(self, index):
        self.current_index = int(index)
        highlight_array = self.highlight_arrays[self.current_index]
        highlighted_image = highlight_area(self.image_path, highlight_array)

        # Convert to Tkinter PhotoImage and display on canvas
        self.tk_image = ImageTk.PhotoImage(highlighted_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image  # Keep a reference

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Heatmap Visualizer")
    
    image_path = "input/1.jpeg"  # Update with your image path
    app = HeatmapApp(root, image_path)

    root.mainloop()
