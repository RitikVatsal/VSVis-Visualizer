import os
import librosa
import cv2
from PIL import Image
import numpy as np


class VideoAudioSyncManager:
    def __init__(self, fps=30):
        self.fps = fps

    def find_matching_audio(self, image_path):
        """Find matching audio file for the given image."""
        directory, image_file = os.path.split(image_path)
        audio_file = image_file.replace('.png', '.mp3')  # Assuming .png images
        audio_path = os.path.join(directory, audio_file)
        if os.path.exists(audio_path):
            return audio_path
        else:
            raise FileNotFoundError(f"Matching audio file not found for {image_path}")

    def create_video_from_image(self, image_path, audio_path, output_path):
        """Create a video from the image, matching the audio duration."""
        image = Image.open(image_path).convert("RGB")
        image_cv = np.array(image)
        height, width, _ = image_cv.shape

        # Get audio duration
        audio_duration = librosa.get_duration(path=audio_path)
        num_frames = int(audio_duration * self.fps)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        for _ in range(num_frames):
            video.write(image_cv)

        video.release()
        print(f"Video created at {output_path}")
        return output_path
