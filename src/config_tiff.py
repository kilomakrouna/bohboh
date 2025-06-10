# -*- coding: utf-8 -*-
"""
Configuration file for the CT Reconstruction Project.

!! IMPORTANT !!
This configuration is set up for loading multiple 2D projection files (e.g., TIFFs).
You MUST find the metadata file (.txt, .info, etc.) associated with your
downloaded projection dataset (e.g., the Helsinki Walnut dataset) and update
the GEOMETRY parameters below with the correct values from that file.
"""

import os
import numpy as np

# --- 1. Path Definitions ---
BASE_DIR = os.getcwd()

# Input Data Paths
# !! MUST BE ADJUSTED !! Point this to the directory containing the .tif files
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROJECTION_DIR = os.path.join(DATA_DIR, 'projections', 'walnut') # Example name
# !! MUST BE ADJUSTED !! Point to where flat/dark files are, if they exist for this dataset
FLAT_FIELD_DIR = os.path.join(DATA_DIR, 'flat_fields') # Or maybe PROJECTION_DIR if they are together
DARK_FIELD_DIR = os.path.join(DATA_DIR, 'dark_fields') # Or maybe PROJECTION_DIR

# Output Paths
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
RECON_SLICES_DIR = os.path.join(OUTPUT_DIR, 'reconstructed_slices')
RECON_VOLUME_DIR = os.path.join(OUTPUT_DIR, 'reconstructed_volume')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_FILE = os.path.join(OUTPUT_DIR, 'reconstruction.log')

# Ground Truth Path (Optional)
# If you have a corresponding ground truth volume
GROUND_TRUTH_PATH = None # os.path.join(DATA_DIR, 'ground_truth', 'some_volume.mha')
# If using the raw walnut volume (400x296x352, USHORT) as GT, add these:
# GT_VOLUME_SHAPE = (352, 296, 400) # Z, Y, X
# GT_DATA_TYPE = np.uint16

# --- 2. Data Loading Parameters ---
# Settings for loading multiple projection image files
PROJECTION_FILE_FORMAT = 'tif'
# !! MUST BE ADJUSTED !! Match the filename pattern exactly
PROJECTION_FILE_PATTERN = '*_walnut_dose_10_*.tif' # Use wildcard '*' for variable parts
PROJECTION_INDEX_RANGE = None # Load all matching files

# Flat/Dark field file names (if available for this dataset)
# !! MUST BE ADJUSTED !! Check metadata/filenames
FLAT_FIELD_AVG_FILENAME = 'flat_field_avg.tif' # Example name
DARK_FIELD_AVG_FILENAME = 'dark_field_avg.tif' # Example name

# RAW_DATA_TYPE is NOT needed when loading standard image files
# RAW_DATA_TYPE = None

# --- 3. Geometry and Setup Parameters ---
# !! MUST BE ADJUSTED based on the metadata file for this specific dataset !!

NUM_PROJECTIONS = 361 # Example: Number of .tif files found (loader will update this)
ANGLE_START_DEG = 0.0   # Example: Get from metadata
ANGLE_END_DEG = 360.0 # Example: Get from metadata
ANGLES_DEG = np.linspace(ANGLE_START_DEG, ANGLE_END_DEG, NUM_PROJECTIONS, endpoint=False) # Loader recalculates
ANGLES_RAD = np.deg2rad(ANGLES_DEG) # Loader recalculates

# Detector Parameters (from metadata)
DETECTOR_PIXEL_SIZE_MM = 0.1 # Example: mm per pixel
DETECTOR_SHAPE = (2368, 2240) # Example: (height, width) - Loader will update from first image

GEOMETRY_TYPE = 'cone' # Should be cone beam for walnut datasets

# Distances (from metadata) in mm
DISTANCE_SOURCE_DETECTOR_MM = 500.0 # Example: SDD
DISTANCE_SOURCE_OBJECT_MM = 100.0  # Example: SOD

# Center of Rotation (COR) offset (from metadata) in pixels
# Note: Helsinki dataset description mentions a shift of 4 pixels might be needed!
CENTER_OF_ROTATION_OFFSET_PX = -4.0 # Example: Try applying the suggested shift

# --- 4. Preprocessing Parameters ---
# !! MUST BE ADJUSTED !! Enable if flat/dark files exist and paths/names are correct
PERFORM_FLAT_FIELD_CORRECTION = False # Set based on availability
PERFORM_DARK_FIELD_CORRECTION = False # Set based on availability
PERFORM_LOG_CONVERSION = True # Usually True for raw detector data

FLAT_FIELD_MODE = 'average'
DARK_FIELD_MODE = 'average'

PERFORM_BEAM_HARDENING_CORRECTION = False
PERFORM_SCATTER_CORRECTION = False
PERFORM_RING_REMOVAL = False # Can enable later if needed

# --- 5. Reconstruction Parameters ---
RECONSTRUCTION_ALGORITHM = 'FDK' # Requires ASTRA or TIGRE for cone beam
FBP_FILTER_NAME = 'ramp'
ITERATIVE_NUM_ITERATIONS = 50
ITERATIVE_RELAXATION_PARAM = 0.1
RECON_VOXEL_SIZE_MM = DETECTOR_PIXEL_SIZE_MM # Match detector by default
# Adjust output shape based on expected size/resolution
RECON_VOLUME_SHAPE = (512, 512, 512) # Example: Smaller output volume (Z, Y, X)
USE_GPU = True



# --- !! NEW: Memory Optimization !! ---
# Process the reconstruction volume in chunks along the Z-axis
# Smaller chunks use less memory per step but might have more overhead.
# Set to None or 0 to disable chunking.
RECON_Z_CHUNK_SIZE = 64 # Example: Reconstruct 64 slices at a time


# --- 6. Postprocessing Parameters ---
PERFORM_POST_FILTERING = False
PERFORM_INTENSITY_SCALING = True
INTENSITY_SCALE_RANGE = (0, 255)
INTENSITY_CLIP = True
PERFORM_MASKING = False

# --- 7. Evaluation Parameters ---
PERFORM_EVALUATION = False # Set to True if GT is available and configured
EVALUATION_METRICS = ['mse', 'psnr', 'ssim']

# --- 8. Visualization Parameters ---
SAVE_PLOTS = True
SHOW_PLOTS = False # Set True for interactive plots (blocks execution)
VISUALIZE_SLICE_AXIS = 0
VISUALIZE_SLICE_INDEX = RECON_VOLUME_SHAPE[VISUALIZE_SLICE_AXIS] // 2
VISUALIZATION_CMAP = 'gray'

# --- 9. Miscellaneous ---
LOG_LEVEL = 'INFO'
NUM_CORES = -1
EPSILON = 1e-9 # Small number to avoid div by zero / log(0)

# --- Helper function to create output directories ---
def ensure_output_dirs_exist():
    """Creates all defined output directories if they don't exist."""
    dirs_to_create = [
        OUTPUT_DIR,
        RECON_SLICES_DIR,
        RECON_VOLUME_DIR,
        PLOT_DIR
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    print("Checked/created output directories.")

print("Configuration loaded.")

