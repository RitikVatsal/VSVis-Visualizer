# VSVis: Visualization Tool for Vision-Speech Models

**VSVis** is an open-source interactive tool for visualizing spatiotemporal attention tensors in Vision-Speech Models (VSMs). This tool supports both visualization and inference of model outputs, providing insights into the alignment between audio-visual features.

---

## Features
- Visualize spatiotemporal attention from pre-computed model outputs or during inference.
- Supports audio and image-based input for models.
- Modular design adaptable to various Vision-Speech Models.
- Evaluation metrics for attention patterns: **Alignment** and **Glancing Scores** (Khorrami & Räsänen, 2021).
- Compatible with **DenseAV** ([Hamilton et al., 2024](https://github.com/mhamilton723/DenseAV)).

---

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-enabled GPU recommended for inference

### Steps

1. **Clone the repository:**

```bash
git clone <repository_url>
cd VSVis-Visualizer
```

2. **Set up a virtual environment:**

```bash
python3.11 -m venv new_env_311
source new_env_311/bin/activate  # macOS/Linux
```

3. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. *(Optional)* **Install Python-Tk if running on macOS:**

```bash
brew install python-tk
```

---

## Usage

### Running the Application
Start the application with:

```bash
python3.11 VisUI.py
```

### Modes of Operation

#### 1. **Only Visualization:**
- Requires a folder containing:
  - `overlay.npy`: Pre-computed attention overlay
  - `image.png`: Image input
  - `audio.wav`: Corresponding audio file
- Select this mode to visualize pre-computed attention data.

#### 2. **Inference and Visualization:**
- Requires a folder containing:
  - `image.png`: Input image
  - `audio.wav`: Corresponding audio
- The tool runs inference using the DenseAV model to generate visualizations.

---

## Evaluation Metrics

The tool calculates attention scores based on the following:

- **Alignment Scores:** Measures the correlation between attention and specific regions.
- **Glancing Scores:** Evaluates the spread of attention in spatial and temporal dimensions.

---

## Tested Platforms

VSVis has been tested on the following platforms:

- **macOS (M1 chipset)**
- **Linux** with CUDA-enabled GPUs

---

## Dataset and Fine-tuning

The **VisText2Speech Dataset** can be used to fine-tune the DenseAV model for chart-captioning tasks. The dataset includes chart images and corresponding speech captions, converted using Matcha-TTS. 

VSVis builds upon the DenseAV framework for spatiotemporal attention visualization. For the original implementation, refer to the [DenseAV repository](https://github.com/mhamilton723/DenseAV).

The codebase supporting fine-tuning DenseAV on VisText2Speech is available [here](https://github.com/aishaeldeeb/vgs_finetune/tree/main).

---

## References

- Hamilton et al., 2024: [DenseAV GitHub Repository](https://github.com/mhamilton723/DenseAV)
- Khorrami & Räsänen, 2021: Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models

---

## License

This tool is open-source under the MIT License.

VSVis incorporates portions of the DenseAV codebase, which is also licensed under the MIT License. For details, see the original [DenseAV License](https://github.com/mhamilton723/DenseAV/blob/main/LICENSE).
