import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import os
import logging
from ultralytics import YOLO

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define a wrapper class for the YOLO model to count pineapple blooms
class PineappleBloomCounter:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.device = next(self.model.parameters()).device
        
    def forward(self, x):
        # This is a wrapper method to make the interface consistent
        return self.predict(x)
    
    def predict(self, x):
        # Convert tensor to format expected by YOLO
        if isinstance(x, torch.Tensor):
            # If input is a tensor, convert to numpy for YOLO
            if x.dim() == 4:  # batch of images
                images = [img.cpu().numpy() for img in x]
            else:  # single image
                images = [x.cpu().numpy()]
        else:
            # If input is already numpy or PIL
            images = [x]
        
        # Run prediction
        results = self.model(images)
        
        # Count detections (assuming pineapple blooms are class 0)
        count = sum(len(result.boxes) for result in results)
        
        return count

def load_model(weights_path):
    """
    Load the pre-trained pineapple bloom counting model.
    
    Args:
        weights_path (str): Path to the model weights file
        
    Returns:
        model: The loaded model wrapper
    """
    try:
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load the YOLO model
        yolo_model = YOLO(weights_path)
        
        # Create our wrapper
        model = PineappleBloomCounter(yolo_model)
        
        logger.info(f"Model loaded successfully from {weights_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image):
    """
    Preprocess the input image for the model.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        numpy.ndarray: Preprocessed image ready for the model
    """
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # No need for additional preprocessing as YOLO handles this internally
        logger.info("Image preprocessed successfully")
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict(model, image):
    """
    Run inference on the preprocessed image.
    
    Args:
        model: The loaded model
        image: Preprocessed image
        
    Returns:
        tuple: (count of pineapple blooms, confidence score)
    """
    try:
        # Run prediction
        results = model.model(image, verbose=False)
        
        # Count detections
        count = len(results[0].boxes)
        
        # Calculate average confidence
        if count > 0:
            confidence = results[0].boxes.conf.mean().item() * 100
        else:
            confidence = 0.0
            
        logger.info(f"Prediction completed: {count} blooms detected with {confidence:.2f}% confidence")
        return count, confidence
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def visualize_detection(image, results):
    """
    Create a visualization of the detection results.
    
    Args:
        image: Original image
        results: Detection results from the model
        
    Returns:
        image: Visualization with detection boxes
    """
    # Get the plotted result from YOLO
    if results and len(results) > 0:
        # Plot results with boxes
        result_plot = results[0].plot()
        return result_plot
    else:
        return image
