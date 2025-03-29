import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monitoring_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to store performance logs
LOGS_DIR = "logs"
PERFORMANCE_LOG_FILE = os.path.join(LOGS_DIR, "performance_logs.csv")

def initialize_logs():
    """
    Initialize the logs directory and files if they don't exist.
    """
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
            logger.info(f"Created logs directory at {LOGS_DIR}")
        
        # Create performance log file if it doesn't exist
        if not os.path.exists(PERFORMANCE_LOG_FILE):
            # Create a DataFrame with the necessary columns
            df = pd.DataFrame(columns=[
                'timestamp', 'inference_time', 'bloom_count', 'confidence'
            ])
            # Save to CSV
            df.to_csv(PERFORMANCE_LOG_FILE, index=False)
            logger.info(f"Created performance log file at {PERFORMANCE_LOG_FILE}")
    
    except Exception as e:
        logger.error(f"Error initializing logs: {str(e)}")
        raise

def log_performance(inference_time, bloom_count, confidence):
    """
    Log model performance metrics.
    
    Args:
        inference_time (float): Time taken for inference in seconds
        bloom_count (int): Number of pineapple blooms detected
        confidence (float): Confidence score of the prediction
    """
    try:
        # Initialize logs if needed
        initialize_logs()
        
        # Create a new log entry
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'inference_time': inference_time,
            'bloom_count': bloom_count,
            'confidence': confidence
        }
        
        # Load existing logs
        try:
            logs_df = pd.read_csv(PERFORMANCE_LOG_FILE)
        except:
            # If file is empty or corrupted, create a new DataFrame
            logs_df = pd.DataFrame(columns=[
                'timestamp', 'inference_time', 'bloom_count', 'confidence'
            ])
        
        # Append new log entry using pd.concat instead of the deprecated append method
        logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
        
        # Save updated logs
        logs_df.to_csv(PERFORMANCE_LOG_FILE, index=False)
        
        logger.info(f"Logged performance metrics: {log_entry}")
    
    except Exception as e:
        logger.error(f"Error logging performance: {str(e)}")
        raise

def display_metrics():
    """
    Generate visualizations and statistics from the logged performance metrics.
    
    Returns:
        dict: Dictionary containing various metrics and visualizations
    """
    try:
        # Initialize logs if needed
        initialize_logs()
        
        # Load performance logs
        try:
            logs_df = pd.read_csv(PERFORMANCE_LOG_FILE)
            logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        except:
            # If file is empty or corrupted, return empty metrics
            logger.warning("No performance logs found or logs are corrupted")
            return {
                "inference_times": pd.DataFrame(),
                "count_distribution": pd.DataFrame(),
                "confidence_levels": pd.DataFrame()
            }
        
        # Calculate metrics
        metrics = {}
        
        # Inference time trend
        inference_times = logs_df[['timestamp', 'inference_time']].set_index('timestamp')
        metrics["inference_times"] = inference_times
        
        # Bloom count distribution
        count_distribution = logs_df['bloom_count'].value_counts().sort_index()
        metrics["count_distribution"] = count_distribution
        
        # Confidence levels over time
        confidence_levels = logs_df[['timestamp', 'confidence']].set_index('timestamp')
        metrics["confidence_levels"] = confidence_levels
        
        # Calculate summary statistics
        summary_stats = {
            "avg_inference_time": logs_df['inference_time'].mean(),
            "min_inference_time": logs_df['inference_time'].min(),
            "max_inference_time": logs_df['inference_time'].max(),
            "avg_confidence": logs_df['confidence'].mean(),
            "total_images_processed": len(logs_df)
        }
        metrics["summary_stats"] = summary_stats
        
        logger.info("Generated performance metrics successfully")
        return metrics
    
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        raise

def generate_performance_report():
    """
    Generate a comprehensive performance report.
    
    Returns:
        str: HTML string containing the performance report
    """
    try:
        # Get metrics
        metrics = display_metrics()
        
        # Create HTML report
        html = """
        <h1>Pineapple Bloom Counter - Performance Report</h1>
        <p>Generated on: {date}</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Inference Time</td><td>{avg_time:.4f} seconds</td></tr>
            <tr><td>Minimum Inference Time</td><td>{min_time:.4f} seconds</td></tr>
            <tr><td>Maximum Inference Time</td><td>{max_time:.4f} seconds</td></tr>
            <tr><td>Average Confidence</td><td>{avg_conf:.2f}%</td></tr>
            <tr><td>Total Images Processed</td><td>{total_images}</td></tr>
        </table>
        """.format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            avg_time=metrics["summary_stats"]["avg_inference_time"],
            min_time=metrics["summary_stats"]["min_inference_time"],
            max_time=metrics["summary_stats"]["max_inference_time"],
            avg_conf=metrics["summary_stats"]["avg_confidence"],
            total_images=metrics["summary_stats"]["total_images_processed"]
        )
        
        logger.info("Generated performance report successfully")
        return html
    
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        raise
