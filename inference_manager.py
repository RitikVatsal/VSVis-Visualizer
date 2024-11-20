import torchaudio
import torchvision.transforms as T
from torchvision.io import read_video
import torch
from PIL import Image
from denseav.shared import norm, crop_to_divisor, blur_dim

class ModelInferenceManager:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def run_inference(self, video_path, audio_path, save_dir=None):
        try:
            # Load audio
            audio, audio_sample_rate = torchaudio.load(audio_path)
            sample_rate = 16000
            if audio_sample_rate != sample_rate:
                audio = torchaudio.transforms.Resample(orig_freq=audio_sample_rate, new_freq=sample_rate)(audio)
            audio = audio[0].unsqueeze(0).to(self.device)

            # Load video frames
            original_frames, _, info = read_video(video_path, pts_unit="sec")
            video_fps = int(info["video_fps"])

            # Transform frames
            img_transform = T.Compose([
                T.Resize((224, 224), Image.BILINEAR),
                lambda x: x.to(torch.float32) / 255,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            frames = torch.cat([img_transform(f.permute(2, 0, 1)).unsqueeze(0) for f in original_frames], axis=0)

            # Ensure frames are on the correct device and batch process
            frames = frames.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                # Audio and image features
                audio_feats = self.model.forward_audio({"audio": audio})
                audio_feats = {k: v.cpu() for k, v in audio_feats.items()}  # Move results to CPU
                
                image_feats = self.model.forward_image({"frames": frames}, max_batch_size=2)
                image_feats = {k: v.cpu() for k, v in image_feats.items()}  # Move results to CPU
                
                # Similarity by head
                sim_by_head = self.model.sim_agg.get_pairwise_sims(
                    {**image_feats, **audio_feats},
                    raw=False,
                    agg_sim=False,
                    agg_heads=False
                ).mean(dim=-2).cpu()

                # Apply blur for smooth results
                sim_by_head = blur_dim(sim_by_head, window=3, dim=-1)

            return sim_by_head, audio_feats, image_feats, video_fps

        except Exception as e:
            raise RuntimeError(f"Error during inference: {e}")
