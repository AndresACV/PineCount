import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Import from other modules
from model import load_model, preprocess_image, predict, visualize_detection
from xai import generate_explanations
from monitoring import log_performance, display_metrics

# Disable file watcher to avoid torch._classes.__path__._path error
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Set page configuration
st.set_page_config(
    page_title="Pineapple Bloom Counter",
    page_icon="🍍",
    layout="wide"
)

# Define a constant for image display width
DEFAULT_IMAGE_WIDTH = 600

def main():
    # Application title and description
    st.title("🍍 Pineapple Bloom Counting with XAI")
    st.markdown("""
    This application allows you to:
    * Upload drone-captured images of pineapple fields
    * Count pineapple blooms using a pre-trained YOLOv8 model
    * Visualize model explanations using XAI techniques
    * Monitor model performance metrics
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Explanation", "Performance Monitoring"])
    
    # Load the model
    with st.spinner("Loading model..."):
        try:
            model = load_model("weights/weights.pt")
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.stop()
    
    if page == "Home":
        st.header("Pineapple Bloom Detection")
        
        # Image upload
        uploaded_file = st.file_uploader("Choose a drone image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image with fixed width
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width=DEFAULT_IMAGE_WIDTH)
            
            # Preprocess image
            with st.spinner("Preprocessing image..."):
                processed_img = preprocess_image(image)
                # Don't display the preprocessed image as it's just the numpy array
            
            # Run prediction
            if st.button("Count Pineapple Blooms"):
                with st.spinner("Running model..."):
                    start_time = pd.Timestamp.now()
                    results = model.model(processed_img, verbose=False)
                    end_time = pd.Timestamp.now()
                    
                    # Get count and confidence
                    count = len(results[0].boxes)
                    confidence = results[0].boxes.conf.mean().item() * 100 if count > 0 else 0.0
                    
                    # Display results
                    st.success(f"Detected {count} pineapple blooms with {confidence:.2f}% confidence")
                    
                    # Display detection visualization with fixed width
                    result_plot = results[0].plot()
                    st.image(result_plot, caption="Detection Results", width=DEFAULT_IMAGE_WIDTH)
                    
                    # Log performance
                    inference_time = (end_time - start_time).total_seconds()
                    log_performance(inference_time, count, confidence)
                    
                    # Store results for explanation
                    st.session_state.image = processed_img
                    st.session_state.results = results
                    st.session_state.model = model
                    
                    # Option to generate explanations
                    if st.button("Explain Results"):
                        st.session_state.page = "Model Explanation"
                        st.experimental_rerun()
    
    elif page == "Model Explanation":
        st.header("Model Explanation (XAI)")
        
        if 'image' in st.session_state and 'results' in st.session_state and 'model' in st.session_state:
            with st.spinner("Generating explanations..."):
                try:
                    explanations = generate_explanations(st.session_state.model, st.session_state.image, st.session_state.results)
                    
                    # Create tabs for different explanation types
                    xai_tabs = st.tabs(["Feature Importance", "Attention Heatmap", "LIME", "Feature Attribution", "Counterfactual", "Details"])
                    
                    with xai_tabs[0]:
                        # Display SHAP values
                        st.subheader("Feature Importance")
                        if explanations["feature_importance"]["has_data"]:
                            # Create feature importance chart directly in Streamlit
                            feature_data = pd.DataFrame({
                                'Detection': explanations["feature_importance"]["feature_names"],
                                'Confidence': explanations["feature_importance"]["confidence_scores"]
                            })
                            
                            # Ensure proper ordering by creating a categorical type with the correct order
                            feature_data['Detection'] = pd.Categorical(
                                feature_data['Detection'],
                                categories=sorted(feature_data['Detection'], key=lambda x: int(x.split()[1])),
                                ordered=True
                            )
                            
                            # Sort by the categorical column to ensure proper display order
                            feature_data = feature_data.sort_values('Detection')
                            
                            # Use Streamlit's native chart capabilities
                            st.bar_chart(
                                feature_data.set_index('Detection')['Confidence'],
                                use_container_width=True
                            )
                            
                            # Add explanation text
                            st.markdown("""
                            **Feature Importance**: This chart shows the confidence scores for each detected pineapple bloom.
                            Higher confidence scores indicate the model is more certain about the detection.
                            """)
                        else:
                            st.info("No detections to analyze for feature importance.")
                    
                    with xai_tabs[1]:
                        # Display heatmap
                        st.subheader("Attention Heatmap")
                        if explanations["heatmap"] is not None:
                            st.image(explanations["heatmap"], caption="Model Attention Heatmap", width=DEFAULT_IMAGE_WIDTH)
                        else:
                            st.info("No heatmap available for this image.")
                    
                    with xai_tabs[2]:
                        # Display LIME explanation
                        st.subheader("LIME Explanation")
                        if explanations["lime"] is not None:
                            st.image(explanations["lime"], caption="LIME Explanation", width=DEFAULT_IMAGE_WIDTH)
                            st.markdown("""
                            **LIME Explanation**: This visualization shows which parts of the image are most important for the model's prediction.
                            The highlighted regions indicate areas that positively contribute to the detection of pineapple blooms.
                            """)
                        else:
                            st.info("No LIME explanation available for this image.")
                    
                    with xai_tabs[3]:
                        # Display feature attribution map
                        st.subheader("Feature Attribution Map")
                        if explanations["attribution_map"] is not None:
                            st.image(explanations["attribution_map"], caption="Feature Attribution Map", width=DEFAULT_IMAGE_WIDTH)
                            st.markdown("""
                            **Feature Attribution Map**: This visualization shows the gradient of importance across the image.
                            Brighter areas indicate regions that have a stronger influence on the model's predictions.
                            """)
                        else:
                            st.info("No feature attribution map available for this image.")
                    
                    with xai_tabs[4]:
                        # Display counterfactual explanation
                        st.subheader("Counterfactual Explanation")
                        if explanations["counterfactual"] is not None:
                            st.image(explanations["counterfactual"], caption="Counterfactual Explanation", width=DEFAULT_IMAGE_WIDTH)
                            st.markdown("""
                            **Counterfactual Explanation**: This visualization shows what would happen if one of the detected blooms was removed.
                            The red box indicates the removed detection. This helps understand how the model's predictions would change if certain features were absent.
                            """)
                        else:
                            st.info("No counterfactual explanation available for this image.")
                    
                    with xai_tabs[5]:
                        # Display additional explanations
                        st.subheader("Explanation Details")
                        
                        # Create expandable sections for different aspects of the explanation
                        with st.expander("Detection Statistics", expanded=True):
                            if "detection_count" in explanations["details"]:
                                st.write(f"**Number of Detections:** {explanations['details']['detection_count']}")
                            
                            if "confidence_stats" in explanations["details"]:
                                conf_stats = explanations["details"]["confidence_stats"]
                                st.write("**Confidence Statistics:**")
                                st.write(f"- Average: {conf_stats['average']}")
                                st.write(f"- Minimum: {conf_stats['minimum']}")
                                st.write(f"- Maximum: {conf_stats['maximum']}")
                            
                            if "size_stats" in explanations["details"]:
                                size_stats = explanations["details"]["size_stats"]
                                st.write("**Size Statistics:**")
                                st.write(f"- Average Area: {size_stats['average_area']}")
                                st.write(f"- Minimum Area: {size_stats['minimum_area']}")
                                st.write(f"- Maximum Area: {size_stats['maximum_area']}")
                        
                        with st.expander("Model Interpretation", expanded=True):
                            if "interpretation" in explanations["details"]:
                                st.write(f"**Interpretation:** {explanations['details']['interpretation']}")
                            
                            if "limitations" in explanations["details"]:
                                st.write(f"**Limitations:** {explanations['details']['limitations']}")
                        
                        with st.expander("Recommendations", expanded=True):
                            if "recommendations" in explanations["details"]:
                                st.write(f"**Recommendations:** {explanations['details']['recommendations']}")
                            
                            if "possible_reasons" in explanations["details"]:
                                st.write(f"**Possible Reasons:** {explanations['details']['possible_reasons']}")
                
                except Exception as e:
                    st.error(f"Error generating explanations: {str(e)}")
        else:
            st.info("Please upload an image and run the model first to generate explanations.")
    
    elif page == "Performance Monitoring":
        st.header("Model Performance Monitoring")
        
        try:
            # Display metrics
            metrics = display_metrics()
            
            if not metrics["inference_times"].empty:
                # Create columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show performance over time
                    st.subheader("Inference Time Trend")
                    st.line_chart(metrics["inference_times"])
                
                with col2:
                    # Show confidence levels
                    st.subheader("Confidence Levels")
                    st.line_chart(metrics["confidence_levels"])
                
                # Show distribution of counts
                st.subheader("Bloom Count Distribution")
                st.bar_chart(metrics["count_distribution"])
                
                # Show summary statistics in an expandable section
                with st.expander("Summary Statistics", expanded=True):
                    st.write(f"Average Inference Time: {metrics['summary_stats']['avg_inference_time']:.4f} seconds")
                    st.write(f"Average Confidence: {metrics['summary_stats']['avg_confidence']:.2f}%")
                    st.write(f"Total Images Processed: {metrics['summary_stats']['total_images_processed']}")
            else:
                st.info("No performance data available yet. Process some images first.")
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")

if __name__ == "__main__":
    main()
