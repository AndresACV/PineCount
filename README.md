# 🍍 PineCount AI
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLO](https://img.shields.io/badge/YOLO-v8-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <img src="https://github.com/AndresACV/PineCount/raw/main/assets/pinecount-logo.svg" alt="PineCount AI Logo" width="200"/>
</p>

> An advanced computer vision system for detecting and counting pineapple blooms in drone-captured imagery. This application combines YOLOv8 object detection with explainable AI techniques to provide insights into the model's decision-making process through an intuitive Streamlit interface.

## ✨ Features

- **🖼️ Image Processing**: Upload and preprocess drone images for optimal detection
- **🔍 Bloom Detection**: Utilize YOLOv8 to accurately identify and count pineapple blooms
- **🧠 Explainable AI (XAI)**: Visualize and understand the model's decisions with heatmaps and feature importance
- **📊 Performance Monitoring**: Track and visualize model performance metrics over time
- **🌐 Interactive Interface**: User-friendly Streamlit interface for all functionalities

## 🏗️ Project Structure

```
PineCount-AI/
├── src/
│   ├── main.py          # Main Streamlit application
│   ├── model.py         # Functions to load, preprocess, and run the model
│   ├── xai.py           # Functions to generate model explanations (XAI)
│   └── monitoring.py    # Functions to log and visualize performance metrics
│
├── weights/
│   └── weights.pt       # Pre-trained YOLOv8 model weights
│
├── logs/                # Directory for performance logs (created at runtime)
│
├── assets/              # Images and other static assets
│   └── pinecount-logo.svg  # Project logo
│
├── requirements.txt     # List of necessary libraries
└── README.md            # This file
```

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/AndresACV/PineCount-AI.git
   cd PineCount-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run src/main.py
   ```

2. Use the web interface to:
   - Upload drone images of pineapple fields
   - Run the detection model to count pineapple blooms
   - View visualizations of detected blooms
   - Generate and explore model explanations
   - Monitor model performance metrics

## 🧪 Model Information

The pineapple bloom detection model is implemented using YOLOv8, a state-of-the-art object detection framework. The pre-trained weights are stored in the `weights/weights.pt` file.

## 🔍 Explainability

The XAI features include:
- Confidence score visualization for each detected bloom
- Attention heatmaps showing which parts of the image the model focuses on
- Detailed statistics about detections including confidence levels and size distributions

## 📊 Monitoring

Performance metrics tracked include:
- Inference time
- Bloom count distribution
- Confidence levels
- Summary statistics

## 📦 Requirements

- Python 3.9+
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- Streamlit
- OpenCV
- SHAP
- Other dependencies listed in `requirements.txt`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

Andrés Calvo - [GitHub Profile](https://github.com/AndresACV)

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Streamlit](https://streamlit.io/) for the web application framework
- [SHAP](https://github.com/slundberg/shap) for model explainability tools
