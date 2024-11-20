from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QWidget, QApplication, QProgressBar
import sys
import time  # For simulation purposes
import torch
import os


class InferenceWorker(QThread):
    finished = pyqtSignal(object, object, object)  # Signal for results (sim_by_head, audio_feats, image_feats)
    error = pyqtSignal(str)  # Signal for error messages

    def __init__(self, inference_manager, video_path, audio_path, save_dir):
        super().__init__()
        self.inference_manager = inference_manager
        self.video_path = video_path
        self.audio_path = audio_path
        self.save_dir = save_dir

    def run(self):
        try:
            print("Model inference started. Please wait...")
            # Run model inference
            sim_by_head, audio_feats, image_feats = self.inference_manager.run_inference(
                self.video_path, self.audio_path, self.save_dir
            )
            torch.cuda.empty_cache()  # Clear CUDA memory after inference
            print("Model inference completed successfully!")
            self.finished.emit(sim_by_head, audio_feats, image_feats)
        except Exception as e:
            self.error.emit(str(e))


class AttentionVisualizerApp(QMainWindow):
    def __init__(self, inference_manager, sync_manager):
        super().__init__()
        self.inference_manager = inference_manager
        self.sync_manager = sync_manager
        self.worker = None  # Placeholder for the worker thread
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Attention Visualizer")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel("Select an image to visualize attention")
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", filter="Image Files (*.png)")
        if image_path:
            try:
                # Find matching audio and create video
                audio_path = self.sync_manager.find_matching_audio(image_path)
                video_path = image_path.replace('.png', '.mp4')
                video_path = self.sync_manager.create_video_from_image(image_path, audio_path, video_path)

                # Set up save directory
                save_dir = os.path.splitext(image_path)[0] + "_output"

                # Show progress bar and loading message
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate mode
                self.label.setText("Model inference running. Please wait...")
                # print("Model inference running. Please wait...")

                # Run inference in a worker thread
                self.worker = InferenceWorker(self.inference_manager, video_path, audio_path, save_dir)
                self.worker.finished.connect(self.inference_complete)
                self.worker.error.connect(self.inference_error)
                self.worker.start()
            except Exception as e:
                self.label.setText(f"Error: {e}")

    def inference_complete(self, sim_by_head, audio_feats, image_feats):
        """Handle the completion of inference."""
        self.progress_bar.setVisible(False)
        self.label.setText("Inference complete! Results saved.")
        print(f"Attention Map Shape: {sim_by_head.shape}")

    def inference_error(self, error_message):
        """Handle errors during inference."""
        self.progress_bar.setVisible(False)
        self.label.setText(f"Error: {error_message}")
        print(f"Error during inference: {error_message}")
