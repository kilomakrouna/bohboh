# -*- coding: utf-8 -*-
"""
Configuration file for the CT Reconstruction Project.

This file centralizes all parameters needed for data loading, preprocessing,
reconstruction, postprocessing, evaluation, and visualization.

Adjust the values below based on your specific experimental setup, data,
and desired processing steps.
"""

import os
import numpy as np
from pathlib import Path

# --- 1. Path Definitions ---
# Define the base directory of your project if needed, otherwise use relative paths.
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Example if config.py is in src/
BASE_DIR = os.getcwd() # Assumes you run scripts from the project root

# Input Data Paths
# !! MUST BE ADJUSTED by the user !!
DATA_DIR = Path("a")
PROJECTION_DIR = Path("b") # Subdirectory for a specific scan
FLAT_FIELD_DIR = Path("c")
DARK_FIELD_DIR = Path("d")

# Output Paths
# Directory where all results will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Subdirectories for specific output types
RECON_SLICES_DIR = os.path.join(OUTPUT_DIR, 'reconstructed_slices')
RECON_VOLUME_DIR = os.path.join(OUTPUT_DIR, 'reconstructed_volume')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_FILE = os.path.join(OUTPUT_DIR, 'reconstruction.log') # Optional: For logging progress/errors

# Ground Truth Path (Optional, for evaluation)
# !! MUST BE ADJUSTED if you have ground truth data !!
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, 'ground_truth', 'phantom_volume.mha') # Example path

# --- 2. Data Loading Parameters ---
# File format of the projection images (e.g., 'tif', 'png', 'dcm')
PROJECTION_FILE_FORMAT = 'tif'
# File naming pattern (if needed, e.g., 'proj_{:03d}.tif') - Use None if files are just listed
PROJECTION_FILE_PATTERN = 'proj_*.tif' # Using glob pattern here
# Range of projection indices to load (if pattern uses numbers) - Use None to load all matching files
PROJECTION_INDEX_RANGE = None # Example: range(0, 360)

# Flat/Dark field file names (if single files are used)
FLAT_FIELD_AVG_FILENAME = 'flat_avg.tif' # Example name for averaged flat field
DARK_FIELD_AVG_FILENAME = 'dark_avg.tif' # Example name for averaged dark field

# --- 3. Geometry and Setup Parameters ---
# !! MUST BE ADJUSTED based on the actual CT scanner geometry !!

# Number of projections (detector images)
# Can often be inferred from loaded files, but useful to have as a check
NUM_PROJECTIONS = 360 # Example: 360 projections over 360 degrees

# Angular range (degrees)
# Full scan (360) or half scan (180) is common
ANGLE_START_DEG = 0.0
ANGLE_END_DEG = 180.0 # For parallel beam, 180 is sufficient. Cone beam often needs 360.
ANGLES_DEG = np.linspace(ANGLE_START_DEG, ANGLE_END_DEG, NUM_PROJECTIONS, endpoint=False)
ANGLES_RAD = np.deg2rad(ANGLES_DEG) # Angles in radians, often needed by libraries

# Detector Parameters
# !! MUST BE ADJUSTED !!
DETECTOR_PIXEL_SIZE_MM = 0.1 # Size of a single detector pixel in mm (width/height)
DETECTOR_SHAPE = (1024, 1024) # Number of pixels (rows, columns) - Will be read from data, but good for checks

# Geometry Type ('parallel', 'cone', 'fan')
# This determines which reconstruction algorithms are suitable
GEOMETRY_TYPE = 'cone' # Flat panels typically imply cone-beam

# Distances (for cone/fan beam) in mm
# !! MUST BE ADJUSTED !!
DISTANCE_SOURCE_DETECTOR_MM = 1000.0 # (SDD)
DISTANCE_SOURCE_OBJECT_MM = 750.0  # (SOD)
# DISTANCE_OBJECT_DETECTOR_MM = DISTANCE_SOURCE_DETECTOR_MM - DISTANCE_SOURCE_OBJECT_MM # (DOD) - Calculated

# Center of Rotation (COR) offset in pixels
# Offset of the rotation axis from the center of the detector (horizontal pixel index)
# Can be critical for reconstruction quality. Might need calibration/estimation.
CENTER_OF_ROTATION_OFFSET_PX = 0.0 # Assume perfectly centered for now

# --- 4. Preprocessing Parameters ---
# Flags to enable/disable specific preprocessing steps
PERFORM_FLAT_FIELD_CORRECTION = True
PERFORM_DARK_FIELD_CORRECTION = True
PERFORM_LOG_CONVERSION = True # Convert transmission to attenuation (-log(I/I0))

# Parameters for specific corrections
FLAT_FIELD_MODE = 'average' # 'average' (use single avg file) or 'per_projection' (if flats change)
DARK_FIELD_MODE = 'average' # 'average' or 'per_projection'

# Beam hardening correction (Placeholder - requires specific algorithm choice)
PERFORM_BEAM_HARDENING_CORRECTION = False
BEAM_HARDENING_METHOD = None # e.g., 'polynomial', 'empirical'
BEAM_HARDENING_PARAMS = {}

# Scatter correction (Placeholder - requires specific algorithm choice)
PERFORM_SCATTER_CORRECTION = False
SCATTER_METHOD = None # e.g., 'beam_stop', 'simulation', 'filter'
SCATTER_PARAMS = {}

# Ring artifact removal (often applied to sinograms)
PERFORM_RING_REMOVAL = False
RING_REMOVAL_METHOD = 'wavelet_fft' # Example method
RING_REMOVAL_PARAMS = {'level': 5, 'sigma': 1.0} # Example parameters

# --- 5. Reconstruction Parameters ---
# Choice of reconstruction algorithm
# Options: 'FBP' (Filtered Back Projection), 'FDK' (Feldkamp-Davis-Kress for cone-beam),
# 'SIRT' (Simultaneous Iterative Recon. Technique), 'SART' (Simultaneous Algebraic Recon. Technique),
# 'CGLS' (Conjugate Gradient Least Squares) etc.
# NOTE: FBP/FDK are faster but less robust to noise/artifacts than iterative methods.
# NOTE: FBP in skimage assumes parallel beam. FDK/SIRT/SART/CGLS often require ASTRA/TIGRE.
RECONSTRUCTION_ALGORITHM = 'FDK' # Choose based on geometry and library availability

# Parameters for FBP/FDK
FBP_FILTER_NAME = 'ramp' # Common filters: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'

# Parameters for Iterative Algorithms (SIRT, SART, CGLS etc.)
ITERATIVE_NUM_ITERATIONS = 50 # Number of iterations to run
ITERATIVE_RELAXATION_PARAM = 0.1 # Relaxation parameter (often specific to algorithm)
ITERATIVE_STOPPING_CRITERION = 'max_iterations' # Or 'tolerance'
ITERATIVE_TOLERANCE = 1e-4 # Tolerance for stopping criterion

# Reconstruction Volume Parameters
# Define the desired size/extent of the reconstructed 3D volume
# Can be same as detector or smaller/larger with different voxel sizes
RECON_VOXEL_SIZE_MM = DETECTOR_PIXEL_SIZE_MM # Start with detector pixel size mapped to volume
RECON_VOLUME_SHAPE = (DETECTOR_SHAPE[0], DETECTOR_SHAPE[1], DETECTOR_SHAPE[1]) # (Z, Y, X) - Assuming square detector projection maps to XY plane

# GPU Acceleration (if using libraries like ASTRA/TIGRE)
USE_GPU = True # Set to False if no compatible GPU or CUDA setup

# --- 6. Postprocessing Parameters ---
# Flags to enable/disable postprocessing steps
PERFORM_POST_FILTERING = False
POST_FILTER_METHOD = 'median' # e.g., 'median', 'gaussian', 'bilateral'
POST_FILTER_PARAMS = {'size': 3} # Example parameter (kernel size)

PERFORM_INTENSITY_SCALING = True
INTENSITY_SCALE_RANGE = (0, 255) # e.g., scale to 8-bit range
INTENSITY_CLIP = True # Clip values outside the range

PERFORM_MASKING = False # Apply a mask (e.g., circular mask)
MASK_RADIUS_RATIO = 0.95 # Ratio of the mask radius to the volume width/2

# --- 7. Evaluation Parameters ---
# Flag to enable evaluation against ground truth (if available)
PERFORM_EVALUATION = False # Set to True if GROUND_TRUTH_PATH is valid

# Metrics to compute (from scikit-image.metrics or custom)
EVALUATION_METRICS = ['mse', 'psnr', 'ssim'] # Mean Squared Error, Peak Signal-to-Noise Ratio, Structural Similarity Index

# --- 8. Visualization Parameters ---
# Flags to control saving/displaying visualizations
SAVE_PLOTS = True
SHOW_PLOTS = False # Set to True for interactive display (blocks execution)

# Slice indices to visualize/save (for 3D volumes)
VISUALIZE_SLICE_AXIS = 0 # Axis along which to slice (0=Z, 1=Y, 2=X)
VISUALIZE_SLICE_INDEX = RECON_VOLUME_SHAPE[VISUALIZE_SLICE_AXIS] // 2 # Middle slice

# Colormap for grayscale images
VISUALIZATION_CMAP = 'gray'

# --- 9. Miscellaneous ---
# Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL = 'INFO'

# Number of CPU cores to use for parallel processing (where applicable)
# Set to -1 to use all available cores, 1 for no parallelism
NUM_CORES = -1

# --- End of Configuration ---

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

# Automatically ensure directories exist when this module is imported
# ensure_output_dirs_exist()
# You might prefer to call this explicitly from your main script instead.

print("Configuration loaded.")

