from denseav.shared import blur_dim
from collections import defaultdict
import customtkinter as ctk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import pygame
import time
import librosa
import numpy as np
import os
import torch
import torchaudio
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from evaluation import get_alignment_score_object, get_glancing_score_object, \
    get_alignment_score_word, get_glancing_score_word


# Initialize customtkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class VSVisUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VSVis Visualizer")
        self.geometry("800x750")

        # Initialize variables
        self.folder_path = None
        self.cap = None
        self.playing = False
        self.cap_rep = 0
        self.duration = 0
        self.total_frames = 0
        self.current_frame = 0
        self.clock = None
        self.overlay = None
        self.range_slider = None
        self.range_btn = None
        self.ranger = 0
        self.bbox_btn = None
        self.drawing_enabled = False
        self.is_drawing = False
        self.tensor = None

        # Variables for bounding box
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.bbox_coordinates = None

        self.mode = 0 # 0 for default, 1 for no_infer, 2 for inference

        # Initialize pygame for audio playback
        pygame.mixer.init()

        self.title = ctk.CTkLabel(self, text="VSVis Visualizer\n\n\nSelect Mode")
        self.title.pack(pady=10)

        self.start_menu_frame = ctk.CTkFrame(self)
        self.start_menu_frame.pack(pady=10)
        
        # If No Inference (Overlay file needed)
        self.only_vis = ctk.CTkButton(self.start_menu_frame, text="Only Visualization", command=self.no_infer_init)
        self.only_vis.grid(row=1, column=0, padx=50, pady=30)

        # Inference
        self.inf_vis = ctk.CTkButton(self.start_menu_frame, text="Inference & Visualization", command=self.inference_init)
        self.inf_vis.grid(row=1, column=1, padx=50, pady=30)


        # Placeholder for other widgets that will appear dynamically
        self.video_frame = None
        self.timeline_slider = None
        self.vis_label = None
        self.control_buttons_frame = None
        self.side_panel = None


    def inference_init(self):
        self.mode = 2
        self.title.configure(text="VSVis Visualizer\n\n- Inference & Visualization -\nInference Folder Select")
        self.folder_path = filedialog.askdirectory(title="Select Folder")
        foldername = self.folder_path.split("/")[-1]
        self.title.configure(text=f"VSVis Visualizer\n\n\nInference Running - {foldername}")
        if self.folder_path:
            self.start_menu_frame.pack_forget()
            # self.run_inference()            
            infer = threading.Thread(target=self.run_inference)
            infer.daemon = True
            infer.start()
            while self.cap_rep == 0:  #keeps running until the variable finished becomes True
                app.update()  #updates the tkinter window
            
            self.create_viz_ui()

        else:
            self.title.configure(text="VSVis Visualizer\n\n- Inference & Visualization -\n(!) Folder Select Error")
        

    def run_inference(self):
            
            self.title.configure(text=f"VSVis Visualizer\n\n- Inference & Visualization -\nRunning Inference - Initializing...")

            # Check CUDA availability
            print("CUDA Available:", torch.cuda.is_available())
            print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

            # Load DenseAV model
            model = torch.hub.load('mhamilton723/DenseAV', 'language').cuda()
            print("DenseAV model loaded successfully!")

            self.title.configure(text=f"VSVis Visualizer\n\n- Inference & Visualization -\nRunning Inference - Model Loaded. Loading Image and Audio...")

            # Paths
            image_name = [f for f in os.listdir(self.folder_path) if f.endswith(".png")][0]
            audio_name = [f for f in os.listdir(self.folder_path) if f.endswith(".wav")][0]
            image_path = os.path.join(self.folder_path, image_name)
            audio_path = os.path.join(self.folder_path, audio_name)
            output_video_path = os.path.join(self.folder_path, "temp_video.mp4")

            # Create a video matching the audio length
            image = Image.open(image_path).convert("RGB")
            image_cv = np.array(image)
            height, width, _ = image_cv.shape

            audio_duration = librosa.get_duration(path=audio_path)
            fps = 6  # Frames per second
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
            audio_input = {"audio": waveform.cuda()}

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
                T.Lambda(lambda x: x.cuda()),
            ])

            frames_tensor = torch.stack([transform(frame) for frame in frames])
            print(f"Video frames tensor shape: {frames_tensor.shape}")  # Should be [num_frames, 3, 224, 224]

            # Model Inference

            self.title.configure(text=f"VSVis Visualizer\n\n- Inference & Visualization -\nRunning Inference - Running Inference...")

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

            self.title.configure(text=f"VSVis Visualizer\n\n- Inference & Visualization -\nInference Complete. Initializing Visualization...")

            # Defining Helpers            
            def get_inferno_with_alpha():
                plasma = plt.cm.inferno(np.linspace(0, 1, 256))
                alphas = np.linspace(0, 1, 256)
                plasma_with_alpha = np.zeros((256, 4))
                plasma_with_alpha[:, 0:3] = plasma[:, 0:3]
                plasma_with_alpha[:, 3] = alphas
                return mcolors.ListedColormap(plasma_with_alpha)

            def alpha_blend_layers(layers):
                blended_image = layers[0]
                for layer in layers[1:]:
                    rgb1, alpha1 = blended_image[:, :3, :, :], blended_image[:, 3:4, :, :]
                    rgb2, alpha2 = layer[:, :3, :, :], layer[:, 3:4, :, :]
                    alpha_out = alpha2 + alpha1 * (1 - alpha2)
                    rgb_out = (rgb2 * alpha2 + rgb1 * alpha1 * (1 - alpha2)) / alpha_out.clamp(min=1e-7)
                    blended_image = torch.cat([rgb_out, alpha_out], dim=1)
                return (blended_image[:, :3] * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            
            def _prep_sims_for_plotting(sim_by_head, n_frames, vh, vw):
                with torch.no_grad():
                    results = defaultdict(list)

                    sims = sim_by_head.max(dim=1).values

                    n_audio_feats = sims.shape[-1]
                    for frame_num in range(n_frames):
                        selected_audio_feat = int((frame_num / n_frames) * n_audio_feats)

                        selected_sim = F.interpolate(
                            sims[frame_num, :, :, selected_audio_feat].unsqueeze(0).unsqueeze(0),
                            size=(vh, vw),
                            mode="bicubic")

                        results["sims_all"].append(selected_sim)

                        for head in range(sim_by_head.shape[1]):
                            selected_sim = F.interpolate(
                                sim_by_head[frame_num, head, :, :, selected_audio_feat].unsqueeze(0).unsqueeze(0),
                                size=(vh, vw),
                                mode="bicubic")
                            results[f"sims_{head + 1}"].append(selected_sim)

                    results = {k: torch.cat(v, dim=0) for k, v in results.items()}
                    return results
                
            prepped_sims = _prep_sims_for_plotting(sim_by_head, num_frames, height, width)
            sims_all = prepped_sims["sims_all"].clamp_min(0)
            sims_all -= sims_all.min()
            sims_all = sims_all / sims_all.max()

            self.tensor = sims_all.squeeze()
            cmap = get_inferno_with_alpha()
            layer = torch.tensor(cmap(sims_all.squeeze().detach().cpu())).permute(0, 3, 1, 2)
            
            self.overlay = layer.permute(0, 2, 3, 1).cpu().numpy()
            # np.save(os.path.join(self.folder_path, "overlay.npy"), self.overlay)

            self.duration = audio_duration
            self.total_frames = num_frames
            self.cap_rep = 1
            pygame.mixer.music.load(audio_path)

            self.title.configure(text=f"VSVis Visualizer\n\n- Inference & Visualization -\nVisualization Ready")


    def no_infer_init(self):
        # Show a file dialog to select the video file
        self.mode = 1
        self.title.configure(text="VSVis Visualizer\n\nOnly Visualization\nFolder Select")
        self.folder_path = filedialog.askdirectory(title="Select Folder")
        if self.folder_path:
            self.title.configure(text=f"VSVis Visualizer\n\n- Only Visualization -\nRunning - {self.folder_path.split('/')[-1]}")
            # Hide the start menu
            # self.start_menu_frame.grid_remove()
            self.start_menu_frame.pack_forget()
            self.cap_rep = 1

            # open audio file
            audio_name = [f for f in os.listdir(self.folder_path) if f.endswith(".wav")][0]
            audio_path = os.path.join(self.folder_path, audio_name)

            self.duration = librosa.get_duration(path=audio_path)
            pygame.mixer.music.load(audio_path)

            # open overlay np file
            overlay_name = [f for f in os.listdir(self.folder_path) if f.endswith(".npy")][0]
            overlay_path = os.path.join(self.folder_path, overlay_name)

            self.overlay = np.load(overlay_path)
            self.total_frames = self.overlay.shape[0]


            # Create the video display and controls
            self.create_viz_ui()

    def create_viz_ui(self):
        # Video display frame
        self.video_frame = ctk.CTkFrame(self, height=400, width=600)
        self.video_frame.pack(pady=20)

        self.vis_label = ctk.CTkLabel(self.video_frame, text="")
        self.vis_label.pack()

        # Timeline slider
        self.timeline_slider = ctk.CTkSlider(
            self, from_=0, to=self.total_frames-1, command=self.seek_video, width=600
        )
        self.timeline_slider.set(0)
        self.timeline_slider.pack(pady=10)

        # Control buttons frame
        self.control_buttons_frame = ctk.CTkFrame(self)
        self.control_buttons_frame.pack(pady=10)

        self.play_button = ctk.CTkButton(
            self.control_buttons_frame, text="⏵/ ⏸", command=self.play_pause
        )
        self.play_button.grid(row=0, column=0, padx=21)

        self.range_mode = ctk.CTkButton(
            self.control_buttons_frame, text="Time Period", command=self.range_slider_init
        )
                # Connect Matplotlib events
        self.range_mode.grid(row=0, column=1, padx=21)
        

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.chart_canvas = FigureCanvasTkAgg(self.fig, self.video_frame)
        self.chart_widget = self.chart_canvas.get_tk_widget()
        self.chart_widget.pack(fill=ctk.BOTH, expand=True)
        self.chart_canvas.mpl_connect("button_press_event", self.on_press)
        self.chart_canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.chart_canvas.mpl_connect("button_release_event", self.on_release)

        self.update_chart(self.current_frame)

    def update_chart(self, index):
        # Clear the axis
        self.ax.clear()

        # Update the index
        self.current_frame = int(index)
        image_name = [f for f in os.listdir(self.folder_path) if f.endswith(".png")][0]
        image_path = os.path.join(self.folder_path, image_name)
        bg_image = plt.imread(image_path)
        self.ax.imshow(bg_image, aspect='auto', alpha=1)

        self.ax.imshow(self.overlay[self.current_frame], aspect='auto', alpha=1)

        # Redraw the chart
        self.chart_canvas.draw()

    def play_pause(self):
        if self.cap_rep and not self.playing:
            self.playing = True
            threading.Thread(target=self.play).start()
            try:
                pygame.mixer.music.play()
                pygame.mixer.music.set_pos(self.current_frame / self.total_frames * self.duration)
            except Exception as e:
                print("Error playing audio - ", e)

        elif self.cap_rep and self.playing:
            self.playing = False
            try:
                pygame.mixer.music.pause()
            except:
                # print("Error pausing audio.")
                pass
            self.update_vis_label("Paused.")
        
        self.update_vis_label("")

    def play(self):
        # Get FPS for proper timing between frames
        fps = self.total_frames / self.duration

        while self.cap_rep and self.playing:
            if self.current_frame < self.total_frames-1:
                start_time = time.time()
                if self.ranger == 0:  
                    self.update_chart(self.current_frame+1)
                else:
                    self.current_frame+=1

                self.timeline_slider.set(self.current_frame)
                time_taken = (time.time() - start_time)
                # print(f"Time taken: {time_taken * 1000:.2f} ms | Expected time: {1000 / fps:.2f} ms")
                time.sleep(max((1 / fps) - time_taken, 0) )
            else:
                self.playing = False
                pygame.mixer.music.stop()
                self.seek_video(0)
                self.update_vis_label("End")
                break

    def seek_video(self, value):
        if self.cap_rep:
            self.current_frame = int(value)
            self.update_chart(self.current_frame)
            try:
                pygame.mixer.music.set_pos(self.current_frame / self.total_frames * self.duration)
            except:
                # print("Error pausing audio.")  
                pass      


    def update_vis_label(self, text):
        self.vis_label.configure(image="", text=text)

    def range_slider_init(self):
        from RangeSlider.RangeSlider import RangeSliderH 

        if self.ranger == 1:
            self.range_slider.destroy()
            self.range_btn.destroy()
            self.eval_btn.destroy()
            self.ranger = 0
            self.range_mode.configure(text="Time Period")
        else:
            hLeft = ctk.DoubleVar(value = 0.2)  #left handle variable initialised to value 0.2
            hRight = ctk.DoubleVar(value = 0.85)  #right handle variable initialised to value 0.85
            self.range_slider = RangeSliderH( app , [hLeft, hRight] , padX = 12, Height= 33, Width=624 , bgColor='#222222', font_color='#222222', font_size=0)
            self.range_slider.pack()

            self.time_period_frame = ctk.CTkFrame(self)
            self.time_period_frame.pack(pady=10)

            self.range_btn = ctk.CTkButton(self.time_period_frame, text="Done", command=self.range_show)
            self.eval_btn = ctk.CTkButton(
            self.time_period_frame, text="Evaluation Mode", command=self.toggle_drawing)
            self.eval_btn.grid(row=0, column=2, padx=21)
            self.range_btn.grid(row=0, column=1, padx=21)
            self.ranger = 1
            self.range_mode.configure(text="Back to Playback")
    
    def range_show(self):
        vals = self.range_slider.getValues()
        lval = int(vals[0]*self.total_frames)
        rval = int(vals[1]*self.total_frames)

        self.ax.clear()

        # Selectintg index
        sel_ind = self.overlay[lval:rval]

        # averaging
        avg = np.mean(sel_ind, axis=0)
        
        image_name = [f for f in os.listdir(self.folder_path) if f.endswith(".png")][0]
        image_path = os.path.join(self.folder_path, image_name)
        bg_image = plt.imread(image_path)
        self.ax.imshow(bg_image, aspect='auto', alpha=1)
        
        self.ax.imshow(avg, aspect='auto', alpha=1)

        # Redraw the chart
        self.chart_canvas.draw()
        self.update_vis_label(f"Selected Range: {lval} to {rval}")
        self.seek_video(lval)
        self.timeline_slider.set(self.current_frame)
        self.calculate_evaluation_metrics()


    def toggle_drawing(self):
        if self.mode == 1: 
            self.update_vis_label("Evaluation Mode for Only Visualization Coming Soon!") 
            return

        """Toggle the bounding box drawing mode."""
        self.drawing_enabled = not self.drawing_enabled
        self.eval_btn.configure(
            text="Exit Evaluations" if self.drawing_enabled else "Evaluation Mode"    
        )
        self.update_vis_label("Select an area in the image" if self.drawing_enabled else "Evaluation Disabled")
        self.geometry("800x800" if self.drawing_enabled else "800x750")
        if self.drawing_enabled:
            self.eval_res_frame = ctk.CTkFrame(self)
            self.eval_res_frame.pack(pady=10)
            self.eval_as_obj = ctk.CTkLabel(self.eval_res_frame, text="Alignment Score \n(Object)\n\n-")
            self.eval_as_obj.grid(row=0, column=0, padx=21)
            self.eval_as_word = ctk.CTkLabel(self.eval_res_frame, text="Alignment Score \n(Word)\n\n-")
            self.eval_as_word.grid(row=0, column=1, padx=21)
            self.eval_glc_obj = ctk.CTkLabel(self.eval_res_frame, text="Glancing Score \n(Object)\n\n-")
            self.eval_glc_obj.grid(row=0, column=2, padx=21)
            self.eval_glc_word = ctk.CTkLabel(self.eval_res_frame, text="Glancing Score \n(Word)\n\n-")
            self.eval_glc_word.grid(row=0, column=3, padx=21)
        else:
            self.eval_res_frame.pack_forget()
            self.clear_current_bounding_boxes()


    def on_press(self, event):
        """Handles the mouse press event."""
        if self.drawing_enabled and event.inaxes:
            # Clear the previous bounding box
            if self.rect:
                self.clear_current_bounding_boxes()
            
            self.start_x, self.start_y = event.xdata, event.ydata
            self.rect = self.ax.add_patch(
                plt.Rectangle((self.start_x, self.start_y), 0, 0,
                              edgecolor="red", facecolor="none", linewidth=2)
            )
            self.is_drawing = True
            self.chart_canvas.draw()

        if not self.drawing_enabled:
            # Remove all bounding boxes
            self.clear_current_bounding_boxes()


    def on_motion(self, event):
        """Handles the mouse motion event."""
        if self.drawing_enabled and self.is_drawing and event.inaxes and self.rect:
            width = event.xdata - self.start_x
            height = event.ydata - self.start_y
            self.rect.set_width(width)
            self.rect.set_height(height)
            self.rect.set_xy((self.start_x, self.start_y))
            self.chart_canvas.draw()

    def on_release(self, event):
        """Handles the mouse release event."""
        if self.drawing_enabled and event.inaxes and self.rect:
            x0, y0 = self.rect.get_xy()
            x0, y0 = int(x0), int(y0)
            width = self.rect.get_width()
            height = self.rect.get_height()
            x1 = int(x0 + width)
            y1 = int(y0 + height)

            self.bbox_coordinates = (min(x0, x1),
                                     min(y0, y1),
                                     max(x0, x1),
                                     max(y0, y1))
            self.is_drawing = False
            print(f"Bounding Box Coordinates: {self.bbox_coordinates}")
            self.update_vis_label(f"Bounding Box Coordinates: {self.bbox_coordinates}")
            self.calculate_evaluation_metrics()


    def clear_current_bounding_boxes(self):
        """Remove all bounding boxes."""
        if self.rect:
            self.rect.remove()
            self.rect = None
        
        self.start_x = self.start_y = None
        self.bbox_coordinates = None
        self.is_first_click = True
        self.chart_canvas.draw()


    def calculate_evaluation_metrics(self):
        
        if self.bbox_coordinates and self.range_slider:
            vals = self.range_slider.getValues()
            lval = int(vals[0]*self.total_frames)
            rval = int(vals[1]*self.total_frames)

            as_obj = get_alignment_score_object(
                self.tensor,
                lval,
                rval,
                self.bbox_coordinates
            )

            as_word = get_alignment_score_word(
                self.tensor,
                lval,
                rval,
                self.bbox_coordinates
            )
            
            glc_obj = get_glancing_score_object(
                self.tensor,
                lval,
                rval,
                self.bbox_coordinates
            )

            glc_word = get_alignment_score_word(
                self.tensor,
                lval,
                rval,
                self.bbox_coordinates
            )
            # print(as_obj)
            # print(as_word)
            # print(glc_obj)
            # print(glc_word)
            self.eval_as_obj.configure(text=f"Alignment Score \n(Object)\n\n{as_obj:f}")
            self.eval_as_word.configure(text=f"Alignment Score \n(Word)\n\n{as_word:f}")
            self.eval_glc_obj.configure(text=f"Glancing Score \n(Object)\n\n{glc_obj:f}")
            self.eval_glc_word.configure(text=f"Glancing Score \n(Word)\n\n{glc_word:f}")
            # glc_word = eval.get_glancing_score_word()
            # as_word = eval.get_alignment_score_word()
            # glc_obj = eval.get_glancing_score_object()
            # glc_word = eval.get_glancing_score_word()

    def quit(self):
        self.playing = False
        if self.cap:
            self.cap.release()
        pygame.mixer.quit()
        self.destroy()

if __name__ == "__main__":
    app = VSVisUI()
    app.mainloop()