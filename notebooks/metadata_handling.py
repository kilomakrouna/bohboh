"""
Metadata handling code for tomographic reconstruction.
This file provides examples of how to load and use metadata from text files
that accompany TIFF images.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add the src directory to the path
sys.path.append('../src')

# Import our custom modules
from utils import load_projections_with_metadata, create_projection_geometry

def load_data_with_metadata(data_path, pattern="*.tif*", angle_pattern='_(\d+)deg'):
    """
    Load projection data with metadata from text files.
    
    Args:
        data_path (str): Path to directory with TIFF files and metadata
        pattern (str): Pattern for TIFF files
        angle_pattern (str): Pattern to extract angles from filenames if metadata unavailable
        
    Returns:
        tuple: (projections, angles, metadata)
    """
    print(f"Loading projections and metadata from {data_path}...")
    
    # Load projections with metadata
    projections, angles, metadata = load_projections_with_metadata(
        data_path, pattern, angle_pattern
    )
    
    # Print available metadata
    print(f"Loaded {len(projections)} projections with shape {projections[0].shape}")
    print(f"Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print("\nMetadata found:")
    for key, value in metadata.items():
        if key != 'geometry':  # Skip printing geometry details
            print(f"  {key}: {value}")
    
    # If geometry info is available in metadata, use it
    if 'geometry' in metadata:
        print("\nGeometry information from metadata:")
        for key, value in metadata['geometry'].items():
            print(f"  {key}: {value}")
        
        # Create geometry dictionary
        geometry = create_projection_geometry(
            angles,
            (projections.shape[1], projections.shape[2]),
            metadata['geometry']['source_origin_dist'],
            metadata['geometry']['origin_detector_dist']
        )
    else:
        # Use default values if not available in metadata
        print("\nNo geometry information found in metadata. Using default values.")
        source_origin_dist = 500.0  # mm
        origin_detector_dist = 500.0  # mm
        
        # Create geometry dictionary
        geometry = create_projection_geometry(
            angles,
            (projections.shape[1], projections.shape[2]),
            source_origin_dist,
            origin_detector_dist
        )
    
    return projections, angles, metadata, geometry

# Example of how to use this in the notebook:
"""
# Use the metadata handling code
from metadata_handling import load_data_with_metadata

# Load data with metadata
projections, angles, metadata, geometry = load_data_with_metadata(data_path)

# Use the data and geometry information for reconstruction
# ...

# You can also check if specific parameters are available in the metadata
if 'exposure_time' in metadata:
    print(f"Exposure time: {metadata['exposure_time']} ms")
"""
