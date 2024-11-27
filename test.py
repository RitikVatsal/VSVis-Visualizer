import torch
import torchaudio
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import librosa
from denseav.plotting import plot_attention_video, plot_2head_attention_video, plot_feature_video, display_video_in_notebook
from denseav.shared import norm, crop_to_divisor, blur_dim

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# Load DenseAV model
model = torch.hub.load('mhamilton723/DenseAV', 'sound_and_language').cpu()
print("DenseAV model loaded successfully!")

# Paths
image_path = r"/Users/aishaeldeeb/Desktop/VGSVis/input/example_figure.png"
audio_path = r"/Users/aishaeldeeb/Desktop/VGSVis/input/example_figure.mp3"
output_video_path = r"/Users/aishaeldeeb/Desktop/VGSVis/input/example_figure_video.mp4"

# Create a video matching the audio length
image = Image.open(image_path).convert("RGB")
image_cv = np.array(image)
height, width, _ = image_cv.shape

audio_duration = librosa.get_duration(path=audio_path)
fps = 30  # Frames per second
num_frames = int(audio_duration * fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for _ in range(num_frames):
    video.write(image_cv)

video.release()
print(f"Video created successfully at {output_video_path}")

# Extract audio and resample if necessary
waveform, sample_rate = torchaudio.load(audio_path)
target_sample_rate = 16000

if sample_rate != target_sample_rate:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

# Prepare audio input
audio_input = {"audio": waveform.cpu()}

# Load video and extract frames
cap = cv2.VideoCapture(output_video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frames.append(frame_pil)

cap.release()

# Transform frames for model
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: x.cpu()),
])

frames_tensor = torch.stack([transform(frame) for frame in frames])
print(f"Video frames tensor shape: {frames_tensor.shape}")  # Should be [num_frames, 3, 224, 224]

# Model Inference
with torch.no_grad():
    print("Running model inference")
    audio_feats = model.forward_audio(audio_input)
    audio_feats = {k: v.cpu() for k,v in audio_feats.items()}
    image_feats = model.forward_image({"frames": frames_tensor.unsqueeze(0)}, max_batch_size=2)
    image_feats = {k: v.cpu() for k,v in image_feats.items()}

    sim_by_head = model.sim_agg.get_pairwise_sims(
        {**image_feats, **audio_feats},
        raw=False,
        agg_sim=False,
        agg_heads=False
    ).mean(dim=-2).cpu()

    sim_by_head = blur_dim(sim_by_head, window=3, dim=-1)
    
    print("Audio features:", {k: v.shape for k, v in audio_feats.items()})
    print("Image features:", {k: v.shape for k, v in image_feats.items()})
    print("Sim by head:", {sim_by_head.shape})
