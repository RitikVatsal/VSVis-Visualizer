import numpy as np
from scipy.ndimage import gaussian_filter

class AttentionMapGenerator:
    def __init__(self, height, width, num_hotspots=3, blur_sigma=15):
        self.height = height
        self.width = width
        self.num_hotspots = num_hotspots
        self.blur_sigma = blur_sigma

    def generate_attention_map(self):
        """Generate a single attention map with concentrated hotspots."""
        attention_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Create hotspots
        for _ in range(self.num_hotspots):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            attention_map[y, x] = 1

        # Apply Gaussian blur to simulate realistic attention spread
        attention_map = gaussian_filter(attention_map, sigma=self.blur_sigma)
        attention_map = attention_map / attention_map.max()  # Normalize to [0, 1]
        return attention_map

    def generate_attention_sequence(self, num_steps):
        """Generate a sequence of attention maps for each time step."""
        return np.stack([self.generate_attention_map() for _ in range(num_steps)], axis=2)
