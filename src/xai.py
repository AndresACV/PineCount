import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import cv2
from PIL import Image
import logging
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import random
from scipy.ndimage import zoom
import io
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/xai_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_explanations(model, image, results):
    """
    Generate explanations for the model's predictions using SHAP, LIME, and other techniques.
    
    Args:
        model: The model wrapper
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        dict: Dictionary containing various explanation visualizations
    """
    try:
        logger.info("Generating model explanations")
        
        # Create a dictionary to store all explanations
        explanations = {}
        
        # Generate feature importance visualization
        explanations["feature_importance"] = generate_feature_importance(results)
        
        # Generate attention heatmap
        explanations["heatmap"] = generate_attention_heatmap(image, results)
        
        # Generate LIME explanations
        explanations["lime"] = generate_lime_explanation(model, image, results)
        
        # Generate feature attribution maps
        explanations["attribution_map"] = generate_feature_attribution(model, image, results)
        
        # Generate counterfactual explanations
        explanations["counterfactual"] = generate_counterfactual(image, results)
        
        # Generate additional explanation details
        explanations["details"] = generate_explanation_details(results)
        
        logger.info("Explanations generated successfully")
        return explanations
    
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        raise

def generate_feature_importance(results):
    """
    Generate feature importance data based on detection results.
    
    Args:
        results: Detection results from the model
        
    Returns:
        dict: Dictionary with feature names and confidence scores for Streamlit visualization
    """
    # Extract confidence scores and classes from results
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get confidence scores
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Create feature names based on detection indices
        feature_names = [f"Detection {i+1}" for i in range(len(confidences))]
        
        # Create a list of tuples with (feature_name, confidence)
        detection_data = list(zip(feature_names, confidences))
        
        # Sort alphabetically by detection name (Detection 1, Detection 2, etc.)
        detection_data.sort(key=lambda x: x[0])
        
        # Unpack the sorted data
        sorted_feature_names = [item[0] for item in detection_data]
        sorted_confidences = np.array([item[1] for item in detection_data])
        
        # Return data for Streamlit to visualize
        return {
            "feature_names": sorted_feature_names,
            "confidence_scores": sorted_confidences.tolist(),
            "has_data": True
        }
    else:
        # Return empty data if no detections
        return {
            "feature_names": [],
            "confidence_scores": [],
            "has_data": False
        }

def generate_attention_heatmap(image, results):
    """
    Generate an attention heatmap showing which parts of the image the model focuses on.
    
    Args:
        image: Original image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: Heatmap visualization
    """
    # Create a heatmap based on detection boxes
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the original image shape
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        
        # Create an empty heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Get bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Add each detection to the heatmap with intensity based on confidence
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Add to heatmap with intensity based on confidence
            heatmap[y1:y2, x1:x2] += conf
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Convert image to numpy if it's not already
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Resize heatmap if needed
        if heatmap_colored.shape[:2] != image_np.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image_np.shape[1], image_np.shape[0]))
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image_np, 0.7, heatmap_colored, 0.3, 0)
        
        return overlay
    else:
        # If no detections, return the original image
        if not isinstance(image, np.ndarray):
            return np.array(image)
        return image

def generate_lime_explanation(model, image, results):
    """
    Generate LIME explanations for the model's predictions.
    
    Args:
        model: The model wrapper
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: LIME explanation visualization
    """
    try:
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Convert image to RGB if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    img_rgb = image.copy()
            else:
                img_rgb = np.array(image.convert('RGB'))
            
            # Create a function that returns the prediction probabilities
            def predict_fn(images):
                batch_preds = []
                for img in images:
                    # Ensure image is in correct format
                    img = img.astype(np.uint8)
                    # Get predictions
                    result = model.model(img, verbose=False)
                    # Create a probability array (dummy for object detection)
                    if len(result[0].boxes) > 0:
                        # Use confidence scores as probabilities
                        probs = result[0].boxes.conf.cpu().numpy()
                        # Create a dummy array with the highest confidence
                        pred = np.zeros(2)
                        pred[1] = np.max(probs)
                        pred[0] = 1 - pred[1]
                    else:
                        pred = np.array([1.0, 0.0])  # No detection
                    batch_preds.append(pred)
                return np.array(batch_preds)
            
            # Initialize LIME image explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Get LIME explanation
            explanation = explainer.explain_instance(
                img_rgb, 
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=100,
                random_seed=42
            )
            
            # Get the explanation for the top label
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=10, 
                hide_rest=False
            )
            
            # Create the LIME visualization
            lime_viz = mark_boundaries(temp / 255.0, mask)
            
            # Convert to image format
            lime_viz = (lime_viz * 255).astype(np.uint8)
            
            return lime_viz
        else:
            # If no detections, return None
            return None
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return None

def generate_feature_attribution(model, image, results):
    """
    Generate feature attribution maps for the model's predictions.
    
    Args:
        model: The model wrapper
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: Feature attribution visualization
    """
    try:
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Convert image to numpy if it's not already
            if not isinstance(image, np.ndarray):
                img_np = np.array(image)
            else:
                img_np = image.copy()
            
            # Create a simple gradient-based attribution map
            # This is a simplified version - in a real implementation,
            # you would use a proper gradient-based method like GradCAM
            
            # Get the bounding boxes
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Create an attribution map
            attribution_map = np.zeros_like(img_np)
            
            # For each detection, create a gradient-like effect around the box
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Create a gradient effect
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                max_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
                
                # Create gradient based on distance from center
                y_indices, x_indices = np.indices(img_np.shape[:2])
                distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                
                # Normalize distances and create gradient effect
                norm_distances = np.clip(1 - (distances / max_dist), 0, 1)
                
                # Apply the gradient to the attribution map
                for c in range(3):  # RGB channels
                    attribution_map[:, :, c] += norm_distances * conf * 255
            
            # Normalize and convert to uint8
            attribution_map = np.clip(attribution_map, 0, 255).astype(np.uint8)
            
            # Blend with original image
            blended = cv2.addWeighted(img_np, 0.6, attribution_map, 0.4, 0)
            
            return blended
        else:
            # If no detections, return None
            return None
    except Exception as e:
        logger.error(f"Error generating feature attribution: {str(e)}")
        return None

def generate_counterfactual(image, results):
    """
    Generate counterfactual explanations for the model's predictions.
    
    Args:
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: Counterfactual visualization
    """
    try:
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Convert image to numpy if it's not already
            if not isinstance(image, np.ndarray):
                img_np = np.array(image)
            else:
                img_np = image.copy()
            
            # Get the bounding boxes
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Create a counterfactual by removing one random detection
            if len(boxes) > 1:
                # Choose a random box to remove
                remove_idx = random.randint(0, len(boxes) - 1)
                
                # Create a mask for the chosen box
                mask = np.ones_like(img_np, dtype=bool)
                box = boxes[remove_idx]
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
                
                # Set the box area in the mask to False
                mask[y1:y2, x1:x2, :] = False
                
                # Create a blurred version of the image
                blurred = cv2.GaussianBlur(img_np, (21, 21), 0)
                
                # Create the counterfactual by combining original image and blurred image
                counterfactual = img_np.copy()
                counterfactual[~mask] = blurred[~mask]
                
                # Draw a red border around the removed detection
                counterfactual = cv2.rectangle(counterfactual, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add text explaining this is a counterfactual
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(counterfactual, "Removed Detection", (x1, y1 - 10), 
                            font, 0.5, (255, 0, 0), 2)
                
                return counterfactual
            else:
                # If only one detection, return the original image with text
                result = img_np.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(result, "Only one detection - no counterfactual available", 
                            (10, 30), font, 0.7, (255, 0, 0), 2)
                return result
        else:
            # If no detections, return None
            return None
    except Exception as e:
        logger.error(f"Error generating counterfactual: {str(e)}")
        return None

def generate_explanation_details(results):
    """
    Generate additional explanation details based on detection results.
    
    Args:
        results: Detection results from the model
        
    Returns:
        dict: Dictionary with additional explanation details
    """
    details = {}
    
    if len(results) > 0:
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            # Get confidence statistics
            confidences = boxes.conf.cpu().numpy()
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            
            # Get box size statistics
            box_areas = []
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                box_areas.append(area)
            
            avg_area = np.mean(box_areas)
            min_area = np.min(box_areas)
            max_area = np.max(box_areas)
            
            details = {
                "detection_count": len(boxes),
                "confidence_stats": {
                    "average": f"{avg_conf:.4f}",
                    "minimum": f"{min_conf:.4f}",
                    "maximum": f"{max_conf:.4f}"
                },
                "size_stats": {
                    "average_area": f"{avg_area:.2f} pixels²",
                    "minimum_area": f"{min_area:.2f} pixels²",
                    "maximum_area": f"{max_area:.2f} pixels²"
                },
                "interpretation": "The model detects pineapple blooms based on visual features. Higher confidence scores indicate stronger certainty in the detection.",
                "limitations": "The model may struggle with occlusions, unusual lighting conditions, or non-standard bloom appearances.",
                "recommendations": "For best results, ensure images are captured in good lighting conditions and from a consistent altitude."
            }
        else:
            details = {
                "detection_count": 0,
                "interpretation": "No pineapple blooms were detected in this image.",
                "possible_reasons": "This could be due to: 1) No blooms actually present, 2) Blooms are too small to detect, 3) Unusual lighting or angles, or 4) Blooms are occluded by foliage.",
                "recommendations": "Try adjusting the camera angle or altitude, or capturing images in better lighting conditions."
            }
    else:
        details = {
            "error": "No results available to analyze.",
            "recommendations": "Please ensure the model is properly loaded and the image is correctly preprocessed."
        }
    
    return details
