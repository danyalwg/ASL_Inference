# ASL_Inference: I3D-based Real-Time Sign Language Recognition

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Directory Structure](#directory-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Downloading the Pretrained Model](#downloading-the-pretrained-model)  
- [Running Inference (CLI)](#running-inference-cli)  
- [Running the Streamlit Application](#running-the-streamlit-application)  
- [Results](#results)  
- [Customization](#customization)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## Project Overview

ASL_Inference is a Python-based sign language recognition engine that leverages the Inflated 3D Convolutional Network (I3D) architecture to process raw video files and output predicted glosses with confidence scores. The system consists of:

- A command-line interface (`inference.py`) for batch processing of videos.  
- A web-based user interface built with Streamlit (`streamlit_app.py`) for interactive inference.  
- Core model implementation in `pytorch_i3d.py` based on the I3D paper by Carreira & Zisserman.  
- Video preprocessing utilities in `videotransforms.py`.  

Designed for ease of deployment, ASL_Inference can be integrated into client workflows for real-time or offline sign language analysis.

## Features

- **Batch Inference**: Automatically process all videos in a specified folder and generate a consolidated result file.  
- **Top-K Predictions**: Outputs top 10 predicted labels with confidence scores.  
- **Streamlit UI**: Upload videos via browser and view live inference results with progress bars.  
- **Modular Architecture**: Swap or fine-tune models via standardized interface.  
- **Extensible**: Add custom preprocessing or postprocessing steps easily.  

## Directory Structure

```text
ASL_Inference/
├── data/
│   └── WLASL2000/             # Raw .mp4 files for inference
├── models/
│   └── asl2000/               # Pretrained model checkpoints
├── preprocess/
│   └── wlasl_class_list.txt   # Label mapping
├── __pycache__/               # Python cache files
├── inference.py               # Command-line inference script
├── pytorch_i3d.py             # I3D model definition
├── videotransforms.py         # Video cropping/transforms
└── streamlit_app.py           # Streamlit web app
```

## Prerequisites

- **Operating System**: Linux, macOS, or Windows  
- **Python**: 3.8 or higher  
- **CUDA**: Compatible GPU drivers for PyTorch (optional but recommended)  
- **Hardware**: GPU with ≥8 GB VRAM for real-time performance  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/danyalwg/ASL_Inference.git
cd ASL_Inference
```

2. Install Python dependencies:

```bash
pip3 install --upgrade pip
pip3 install torch torchvision numpy opencv-python streamlit
```

## Downloading the Pretrained Model

Download the compressed model archive from Google Drive:  
- **Model**: [[Google Drive Link Placeholder](https://drive.google.com/drive/folders/1uSNDrJtP1f0g553kKQMiCsj7fsyww-n_?usp=drive_link)]  

Uncompress the archive into the `models/asl2000/` directory so you end up with:

```text
ASL_Inference/
└── models/
    └── asl2000/
        └── FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt
```

Ensure that `inference.py` and `streamlit_app.py` can access `models/asl2000/`.

## Running Inference (CLI)

Process all videos in `data/WLASL2000/` and save predictions to `inference_results.txt`:

```bash
python3 inference.py
```

- **Output**: `inference_results.txt` containing top-10 predictions per video.  
- **Logs**: Console output shows progress and skipped files.  

## Running the Streamlit Application

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

1. Open the displayed URL (e.g., `http://localhost:8501`).  
2. Upload a single video (`.mp4` or `.mov`).  
3. Click **▶ Start Inference** to view predictions and confidence bars.  
4. Download results as CSV via the **Export Results** button.  

## Results

- **CLI Output**: A text file (`inference_results.txt`) with sectioned top-10 lists.  
- **Web UI**: Real-time display and CSV download of `(label, confidence%)`.  

## Customization

- **Model Architecture**: Modify `pytorch_i3d.py` to add or remove endpoints.  
- **Preprocessing**: Edit `videotransforms.py` to include your own transforms.  
- **Inference Logic**: Adjust `inference.py` for alternative scoring, ranking, or output formats.  

## Contributing

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes and open a pull request.  
4. Ensure code passes linting and basic tests.  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

## Contact

For questions or support, contact:  
- **Maintainer**: Mr. Danyal — danyalwg@gmail.com  
