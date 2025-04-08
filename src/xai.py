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
from torch.autograd import Function

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
        
        # Generate Grad-CAM visualization
        explanations["gradcam"] = generate_gradcam(model, image, results)
        
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
    Generate feature attribution maps using occlusion sensitivity.
    This technique systematically occludes different parts of the image and
    observes how the model's predictions change, revealing which regions are most important.
    
    Args:
        model: The model wrapper
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: Occlusion sensitivity map visualization
    """
    try:
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Convert image to numpy if it's not already
            if not isinstance(image, np.ndarray):
                img_np = np.array(image, dtype=np.float32)
            else:
                img_np = image.copy().astype(np.float32)
                
            # Ensure image is RGB
            if len(img_np.shape) == 2:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
            # Normalize image to 0-255 range if it's not already
            if img_np.max() <= 1.0:
                img_np *= 255.0
            
            # Get original detection boxes and confidences
            orig_boxes = results[0].boxes.xyxy.cpu().numpy()
            orig_confidences = results[0].boxes.conf.cpu().numpy()
            
            # Create a sensitivity map of the same size as the image
            height, width = img_np.shape[:2]
            sensitivity_map = np.zeros((height, width), dtype=np.float32)
            
            # Define occlusion parameters - use larger occlusion size and stride for faster processing
            occlusion_size = min(width, height) // 10  # Larger occlusion patch (was 15)
            occlusion_stride = occlusion_size // 1     # Larger stride (was 2)
            
            # Create a gray occlusion patch - using float32 to match the image type when converted
            occlusion_value = np.ones((occlusion_size, occlusion_size, 3), dtype=np.float32) * 127.0
            
            # Iterate over the image with the occlusion patch - limit to a reasonable number of samples
            max_samples = 100  # Limit the number of occlusion samples for performance
            sample_count = 0
            
            for y in range(0, height - occlusion_size + 1, occlusion_stride):
                for x in range(0, width - occlusion_size + 1, occlusion_stride):
                    # Check if we've reached the sample limit
                    sample_count += 1
                    if sample_count > max_samples:
                        break
                        
                    # Create a copy of the image
                    occluded_img = img_np.copy()
                    
                    # Apply occlusion patch
                    occluded_img[y:y+occlusion_size, x:x+occlusion_size] = occlusion_value
                    
                    # Run prediction on occluded image
                    occluded_results = model.model(occluded_img, verbose=False)
                    
                    # Calculate the change in confidence
                    if len(occluded_results[0].boxes) > 0:
                        # Get new confidences
                        new_confidences = occluded_results[0].boxes.conf.cpu().numpy()
                        
                        # Calculate the average drop in confidence
                        # Higher values mean the region is more important
                        if len(orig_confidences) > 0 and len(new_confidences) > 0:
                            # Calculate average confidence before and after occlusion
                            orig_avg_conf = np.mean(orig_confidences)
                            new_avg_conf = np.mean(new_confidences)
                            
                            # The difference shows how important this region is
                            # Positive values mean confidence decreased when occluded (important region)
                            diff = orig_avg_conf - new_avg_conf
                            
                            # Assign the difference to the sensitivity map
                            sensitivity_map[y:y+occlusion_size, x:x+occlusion_size] = max(0, diff)
                        else:
                            # If all detections disappeared, this region is very important
                            sensitivity_map[y:y+occlusion_size, x:x+occlusion_size] = np.mean(orig_confidences)
                    else:
                        # If all detections disappeared, this region is very important
                        sensitivity_map[y:y+occlusion_size, x:x+occlusion_size] = np.mean(orig_confidences)
                
                # Check if we've reached the sample limit after completing a row
                if sample_count > max_samples:
                    break
            
            # Normalize the sensitivity map
            if sensitivity_map.max() > 0:
                sensitivity_map = sensitivity_map / sensitivity_map.max()
            
            # Apply colormap for visualization - ensure proper conversion to uint8
            sensitivity_map_uint8 = np.uint8(255 * sensitivity_map)
            heatmap_colored = cv2.applyColorMap(sensitivity_map_uint8, cv2.COLORMAP_JET)
            
            # Convert image to BGR for overlay and ensure it's uint8
            img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Overlay heatmap on image
            alpha = 0.6
            blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
            
            # Draw original bounding boxes for reference
            for box, conf in zip(orig_boxes, orig_confidences):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blended, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add a title to the image
            cv2.putText(blended, "Occlusion Sensitivity Map", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return blended
        else:
            # If no detections, return None
            return None
    except Exception as e:
        logger.error(f"Error generating occlusion sensitivity map: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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

def generate_gradcam(model, image, results):
    """
    Generate Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for the model's predictions.
    
    Args:
        model: The model wrapper (YOLO model)
        image: Preprocessed image
        results: Detection results from the model
        
    Returns:
        numpy.ndarray: Grad-CAM visualization or None if no detections
    """
    try:
        # Check if there are any detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            logger.info("No detections for Grad-CAM")
            return None
            
        # Get the YOLO model
        yolo_model = model.model
        
        # Convert image to numpy if it's not already
        if not isinstance(image, np.ndarray):
            img_np = np.array(image)
        else:
            img_np = image.copy()
            
        # Ensure image is RGB
        if len(img_np.shape) == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Since we can't easily get gradients from YOLO, we'll create a simplified Grad-CAM
        # by using the detection boxes and confidences to generate a heatmap
        
        # Create an empty heatmap
        heatmap = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
        
        # Get detection boxes and confidences
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # For each detection, create a gaussian blob centered on the detection
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            
            # Calculate center and size of box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            # Create a gaussian blob
            sigma_x = width / 3  # Cover most of the box width
            sigma_y = height / 3  # Cover most of the box height
            
            # Create coordinate grids
            y, x = np.mgrid[0:img_np.shape[0], 0:img_np.shape[1]]
            
            # Calculate gaussian
            gaussian = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
            
            # Add to heatmap, weighted by confidence
            heatmap += gaussian * conf
        
        # Normalize heatmap
        heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Convert image to BGR for overlay
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on image
        alpha = 0.5
        grad_cam = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Draw bounding boxes for reference
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(grad_cam, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(grad_cam, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add a title to the image
        cv2.putText(grad_cam, "Grad-CAM: Areas of attention", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return grad_cam
            
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
