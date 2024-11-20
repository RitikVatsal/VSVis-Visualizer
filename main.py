import sys
from PyQt5.QtWidgets import QApplication
from PIL import Image
from visualizer_app import AttentionVisualizerApp
from inference_manager import ModelInferenceManager
from sync_manager import VideoAudioSyncManager
import torch

if __name__ == "__main__":
    # app = QApplication(sys.argv)

    # image_path = r"C:\Users\Aisha\Desktop\VGSVis\input\example_figure.png"
    # audio_path = r"C:\Users\Aisha\Desktop\VGSVis\input\example_figure_caption.mp3"
    # output_video_path = r"C:\Users\Aisha\Desktop\VGSVis\input\example_figure_video.mp4"

    # Create video
    # video_creator = VideoCreator(image_path, audio_path, output_video_path)
    # video_creator.create_video()

    # Load DenseAV model
    # print("CUDA Available:", torch.cuda.is_available())
    # print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    # model = torch.hub.load('mhamilton723/DenseAV', 'sound_and_language').cuda()
    # print("DenseAV model loaded successfully!")

    # Run model inference
    # inference = ModelInference(model)
    # frames = inference.preprocess_video(output_video_path)
    # audio = inference.preprocess_audio(audio_path)

    # print(f"Video frames tensor shape: {frames.shape}")  # Should be [num_frames, 3, 224, 224]
    # print(f"Running model inference ...")
    # sim_by_head, audio_feats, image_feats = inference.run_inference(frames, audio)
    # # print(f"sim by head: {sim_by_head}")
    # print("Sim by head shape:", sim_by_head.shape)

    # comment model inference if you want to save time and view visualization directly
    # uncomment sample sim_by_head_sample and use it for the visualization 

    # sim_by_head_sample = torch.randn(1348, 2, 14, 14, 249)


    #TODO: visualize sim by head (utlize plotting.py)
    # Initialize and run the application
    # window = AttentionVisualizerApp(image, attention_data)
    # window.resize(600, 600)
    # window.show()
    # sys.exit(app.exec_())
    # Initialize PyTorch model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.hub.load('mhamilton723/DenseAV', 'sound_and_language').to(device)
    
    # Initialize managers
    inference_manager = ModelInferenceManager(model, device)
    sync_manager = VideoAudioSyncManager(fps=30)  # Default FPS is 30
    
    # Initialize and run the visualization application
    app = QApplication(sys.argv)
    window = AttentionVisualizerApp(inference_manager, sync_manager)
    window.show()
    sys.exit(app.exec_())