# -*- coding: utf-8 -*-
"""
Module for loading CT data including projections, flat fields, and dark fields.
Loads multiple projection image files (e.g., TIFF, PNG).
"""

import os
import glob
import numpy as np
import imageio.v3 as iio # Use imageio v3 API
import logging
from typing import Tuple, Optional, List, Union

# Import configuration settings
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    raise

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)


def natural_sort_key(s: str, _nsre=None) -> list:
    """
    Key for natural sorting (e.g., 'image1.tif', 'image10.tif').
    Helps ensure projections are loaded in the correct order.
    """
    import re
    if _nsre is None:
        _nsre = re.compile(r'(\d+)')
    # Extract numbers, converting them to int for proper numerical sorting
    parts = _nsre.split(s)
    # Handle cases where filename might start with numbers or have multiple number groups
    # Prioritize the last number group for sorting projection indices typically found at the end
    key_parts = []
    for text in parts:
        if text.isdigit():
            key_parts.append(int(text))
        else:
            key_parts.append(text.lower())
    # Return the list ensuring numerical parts are integers
    return key_parts


def load_projections(cfg: object) -> Optional[np.ndarray]:
    """
    Loads a series of 2D projection images from the specified directory.

    Args:
        cfg: The configuration object containing parameters like
             PROJECTION_DIR, PROJECTION_FILE_PATTERN, etc.

    Returns:
        A 3D NumPy array containing the projection data (num_projections, height, width),
        or None if loading fails. Updates cfg.DETECTOR_SHAPE and cfg.NUM_PROJECTIONS.
    """
    proj_dir = cfg.PROJECTION_DIR
    pattern = cfg.PROJECTION_FILE_PATTERN
    index_range = cfg.PROJECTION_INDEX_RANGE

    if not os.path.isdir(proj_dir):
        log.error(f"Projection directory not found: {proj_dir}")
        return None

    search_path = os.path.join(proj_dir, pattern)
    log.info(f"Searching for projection files matching: {search_path}")
    file_list = glob.glob(search_path)

    if not file_list:
        log.error(f"No projection files found matching pattern '{pattern}' in {proj_dir}")
        return None

    # Sort files naturally to ensure correct order based on indices
    try:
        # Use the last number found in the filename as the primary sort key
        file_list.sort(key=lambda s: natural_sort_key(s)[-1] if isinstance(natural_sort_key(s)[-1], int) else float('inf'))
        log.debug(f"Sorted file list (first 5): {[os.path.basename(f) for f in file_list[:5]]}")
    except Exception as e:
        log.warning(f"Could not perform robust natural sort on filenames, using standard sort. Error: {e}")
        file_list.sort()

    # Filter by index range if specified
    if index_range is not None:
        # This assumes filenames contain indices that correspond to the range
        # A more robust implementation might parse indices from filenames
        try:
            file_list = file_list[index_range.start : index_range.stop : index_range.step]
            log.info(f"Filtered files using index range: {index_range}. {len(file_list)} files remaining.")
        except Exception as e:
            log.error(f"Failed to apply index range {index_range} to file list. Error: {e}")
            return None

    if not file_list:
        log.error("No projection files remaining after filtering.")
        return None

    log.info(f"Found {len(file_list)} projection files to load.")

    projections = []
    first_shape = None
    for i, file_path in enumerate(file_list):
        try:
            log.debug(f"Loading projection {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
            # Use imageio to read various formats. Add format hint if needed.
            img = iio.imread(file_path) # Returns NumPy array

            # Ensure image is grayscale (2D)
            if img.ndim == 3:
                log.warning(f"Projection {os.path.basename(file_path)} is RGB, converting to grayscale (average).")
                img = np.mean(img, axis=2).astype(img.dtype)
            elif img.ndim != 2:
                log.error(f"Projection {os.path.basename(file_path)} has unexpected dimensions: {img.ndim}. Skipping.")
                continue

            if first_shape is None:
                first_shape = img.shape
                log.info(f"Detected projection shape: {first_shape} (height, width)")
                # Update config with actual detector shape from data
                cfg.DETECTOR_SHAPE = first_shape
            elif img.shape != first_shape:
                log.error(f"Projection {os.path.basename(file_path)} has mismatched shape {img.shape}, expected {first_shape}. Skipping.")
                continue

            projections.append(img)

        except FileNotFoundError:
            log.error(f"File not found during loading: {file_path}. Stopping.")
            return None
        except Exception as e:
            log.error(f"Error loading file {file_path}: {e}")
            return None # Or decide to skip the file and continue

    if not projections:
        log.error("Failed to load any valid projection images.")
        return None


    log.info(f"Loaded {len(projections)} projections successfully.")
    # Stack into a 3D array (projections, height, width)


    import dask.array as da

    chunk_shape = 'auto' # or e.g., (1000, 1000) or (500, 500)
    dask_projections = [da.from_array(p, chunks=chunk_shape) for p in projections]

    # 2. Stack the list of Dask arrays along the first axis (axis=0).
    #    - This creates a new *lazy* Dask array representing the stacked result.
    #    - No significant computation or memory allocation happens yet.
    projections_stack = da.stack(dask_projections, axis=0)


    log.info(f"Successfully loaded {projections_stack.shape[0]} projections.")
    log.info(f"Final projection stack shape: {projections_stack.shape} (projections, height, width)")

    # Update config with actual number of projections loaded
    cfg.NUM_PROJECTIONS = projections_stack.shape[0]
    # Recalculate angles based on actual number of projections loaded
    cfg.ANGLES_DEG = np.linspace(cfg.ANGLE_START_DEG, cfg.ANGLE_END_DEG, cfg.NUM_PROJECTIONS, endpoint=False)
    cfg.ANGLES_RAD = np.deg2rad(cfg.ANGLES_DEG)
    log.info(f"Updated config: NUM_PROJECTIONS={cfg.NUM_PROJECTIONS}, angles recalculated.")

    return projections_stack


def load_calibration_image(dir_path: str, avg_filename: str, expected_shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """
    Loads a single averaged calibration image (flat or dark field).
    """
    if not dir_path or not avg_filename: return None # Added check
    if not os.path.isdir(dir_path):
        log.warning(f"Calibration directory not found: {dir_path}")
        return None

    file_path = os.path.join(dir_path, avg_filename)
    if not os.path.isfile(file_path):
        log.warning(f"Average calibration file not found: {file_path}")
        return None

    try:
        log.info(f"Loading average calibration file: {file_path}")
        img = iio.imread(file_path)

        # Ensure image is grayscale (2D)
        if img.ndim == 3:
            log.warning(f"Calibration file {avg_filename} is RGB, converting to grayscale (average).")
            img = np.mean(img, axis=2).astype(img.dtype)
        elif img.ndim != 2:
            log.error(f"Calibration file {avg_filename} has unexpected dimensions: {img.ndim}. Cannot use.")
            return None

        # Validate shape if expected shape is provided
        if expected_shape is not None and img.shape != expected_shape:
            log.error(f"Calibration file {avg_filename} shape {img.shape} mismatch. Expected {expected_shape}.")
            return None

        log.info(f"Successfully loaded calibration file {avg_filename} with shape {img.shape}.")
        return img.astype(np.float32) # Convert to float32 for processing

    except Exception as e:
        log.error(f"Error loading calibration file {file_path}: {e}")
        return None


def load_flat_fields(cfg: object, expected_shape: Optional[Tuple[int, int]]) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """
    Loads flat field data based on the configuration mode.
    """
    if not cfg.PERFORM_FLAT_FIELD_CORRECTION:
        log.info("Flat field correction is disabled in config. Skipping loading.")
        return None

    mode = cfg.FLAT_FIELD_MODE
    dir_path = cfg.FLAT_FIELD_DIR

    if mode == 'average':
        return load_calibration_image(dir_path, getattr(cfg, 'FLAT_FIELD_AVG_FILENAME', None), expected_shape)
    elif mode == 'per_projection':
        log.warning("Loading 'per_projection' flat fields is not fully implemented yet.")
        return None
    else:
        log.error(f"Unknown FLAT_FIELD_MODE: {mode}")
        return None


def load_dark_fields(cfg: object, expected_shape: Optional[Tuple[int, int]]) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """
    Loads dark field data based on the configuration mode.
    """
    if not cfg.PERFORM_DARK_FIELD_CORRECTION:
        log.info("Dark field correction is disabled in config. Skipping loading.")
        return None

    mode = cfg.DARK_FIELD_MODE
    dir_path = cfg.DARK_FIELD_DIR

    if mode == 'average':
        return load_calibration_image(dir_path, getattr(cfg, 'DARK_FIELD_AVG_FILENAME', None), expected_shape)
    elif mode == 'per_projection':
        log.warning("Loading 'per_projection' dark fields is not fully implemented yet.")
        return None
    else:
        log.error(f"Unknown DARK_FIELD_MODE: {mode}")
        return None

def load_ground_truth(cfg: object) -> Optional[np.ndarray]:
    """
    Loads the ground truth volume for evaluation purposes.
    """
    if not cfg.PERFORM_EVALUATION or not hasattr(cfg, 'GROUND_TRUTH_PATH') or not cfg.GROUND_TRUTH_PATH:
        log.info("Ground truth loading skipped (evaluation disabled or path not set).")
        return None

    file_path = cfg.GROUND_TRUTH_PATH
    if not os.path.isfile(file_path):
        log.error(f"Ground truth file not found: {file_path}")
        return None

    try:
        log.info(f"Loading ground truth volume: {file_path}")
        # Adapt if GT is raw volume
        if file_path.lower().endswith('.raw'):
             log.warning("Attempting to load ground truth from .raw file. Requires GT_VOLUME_SHAPE and GT_DATA_TYPE in config.")
             try:
                 gt_shape = cfg.GT_VOLUME_SHAPE # e.g., (352, 296, 400) for the walnut volume Z,Y,X
                 gt_dtype = cfg.GT_DATA_TYPE # e.g., np.uint16
                 gt_data_1d = np.fromfile(file_path, dtype=gt_dtype)
                 expected_elements = np.prod(gt_shape)
                 if gt_data_1d.size != expected_elements:
                      log.error(f"Ground truth raw file size mismatch: Read {gt_data_1d.size}, expected {expected_elements} for shape {gt_shape}.")
                      return None
                 volume = gt_data_1d.reshape(gt_shape)
             except AttributeError as e:
                 log.error(f"Missing config parameter for loading raw ground truth: {e}. Need GT_VOLUME_SHAPE and GT_DATA_TYPE.")
                 return None
             except Exception as e_raw:
                 log.error(f"Error loading raw ground truth volume {file_path}: {e_raw}")
                 return None
        else:
             # Load standard formats (mha, tif stack, etc.)
             volume = iio.volread(file_path)

        log.info(f"Successfully loaded ground truth volume with shape {volume.shape}.")
        return volume.astype(np.float32)
    except ImportError:
        log.error(f"Failed to load volume {file_path}. Missing dependency for this format? (e.g., pip install imageio[itk])")
        return None
    except Exception as e:
        log.error(f"Error loading ground truth volume {file_path}: {e}")
        return None


# Example usage (for testing purposes)
if __name__ == '__main__':
    print("--- Running Data Loader (TIFF) Test ---")
    # ... (rest of test script remains similar) ...
    config.ensure_output_dirs_exist()
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False
    log.info("Starting data loader (TIFF) test script.")
    projections_data = load_projections(config)
    if projections_data is not None:
        log.info(f"Projections loaded successfully. Shape: {projections_data.shape}")
        detected_shape = (projections_data.shape[1], projections_data.shape[2])
        flat_field_data = load_flat_fields(config, detected_shape)
        dark_field_data = load_dark_fields(config, detected_shape)
    else:
        log.error("Failed to load projections.")
    ground_truth_data = load_ground_truth(config)
    log.info("Data loader (TIFF) test script finished.")
    print("--- End of Data Loader Test ---")

