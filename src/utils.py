# -*- coding: utf-8 -*-
"""
Utility functions for the CT Reconstruction Project.

Includes helper functions for tasks like:
- Saving the final 3D volume to a file.
- Potentially other common tasks (e.g., configuration loading, timing).
"""

import numpy as np
import os
import logging
import time
from typing import Optional, Dict, Any
import imageio.v3 as iio

# Import configuration settings
try:
    import config
    # Define EPSILON using config if available, otherwise default
    EPSILON = getattr(config, 'EPSILON', 1e-9)
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    EPSILON = 1e-9 # Default epsilon if config fails
    raise

# Configure logging
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=getattr(config, config.LOG_LEVEL, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def save_volume(
    volume: np.ndarray,
    filename: str,
    output_dir: Optional[str] = None,
    file_format: str = 'mha', # MetaImage is good for volumes
    cfg: Optional[object] = None # Pass config for default dir
) -> bool:
    """
    Saves the 3D volume data to a file.

    Args:
        volume: The 3D NumPy array (Z, Y, X) to save.
        filename: The desired name for the output file (e.g., 'reconstruction.mha').
        output_dir: Directory to save the volume. If None, uses RECON_VOLUME_DIR
                    from config (if cfg is provided). Defaults to current dir otherwise.
        file_format: The file format to use ('mha', 'tif', 'npy').
                     'mha' (MetaImage) or 'tif' (multi-page TIFF) are common for volumes.
                     'npy' is NumPy's native binary format.
        cfg: The configuration object (optional, used for default output directory).

    Returns:
        True if saving was successful, False otherwise.
    """
    if volume is None:
        log.error("Cannot save volume: Input volume data is None.")
        return False

    if output_dir is None:
        if cfg and hasattr(cfg, 'RECON_VOLUME_DIR'):
            output_dir = cfg.RECON_VOLUME_DIR
        else:
            output_dir = '.' # Default to current directory if no config/dir specified
            log.warning(f"Output directory not specified and config unavailable. Saving to current directory: {os.path.abspath(output_dir)}")

    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    save_path = os.path.join(output_dir, filename)

    log.info(f"Attempting to save volume (shape: {volume.shape}, dtype: {volume.dtype}) to {save_path} (format: {file_format})...")
    start_time = time.time()

    try:
        if file_format.lower() == 'mha':
            # Requires imageio plugin (e.g., imageio[itk])
            iio.imwrite(save_path, volume, extension='.mha') # Use imwrite for volumes too
        elif file_format.lower() in ['tif', 'tiff']:
            # Save as multi-page TIFF
            # Ensure data is in suitable format (e.g., uint8, uint16)
            # Determine appropriate dtype based on volume range or config
            if volume.dtype == np.float32 or volume.dtype == np.float64:
                 log.warning("Saving float volume as TIFF. Scaling to uint16 for compatibility.")
                 vol_min, vol_max = np.min(volume), np.max(volume)
                 if vol_max - vol_min > EPSILON:
                     norm_volume = (volume - vol_min) / (vol_max - vol_min)
                 else:
                     norm_volume = np.zeros_like(volume)
                 volume_to_save = (norm_volume * 65535).astype(np.uint16)
            else:
                 volume_to_save = volume # Assume integer type already

            iio.imwrite(save_path, volume_to_save, extension='.tif') # imageio handles multi-page TIFF
        elif file_format.lower() == 'npy':
            np.save(save_path, volume)
        else:
            log.error(f"Unsupported file format for saving volume: '{file_format}'. Choose 'mha', 'tif', or 'npy'.")
            return False

        end_time = time.time()
        log.info(f"Volume successfully saved to {save_path} in {end_time - start_time:.2f} seconds.")
        return True

    except ImportError as e:
         log.error(f"Failed to save volume to {save_path}. Missing dependency for format '{file_format}'? (e.g., pip install imageio[itk] for mha). Error: {e}")
         return False
    except Exception as e:
        log.error(f"Failed to save volume to {save_path}: {e}")
        return False


# --- Potentially add other utility functions ---
# E.g., function to load configuration from a YAML file
# import yaml
# def load_config_from_yaml(yaml_path):
#     try:
#         with open(yaml_path, 'r') as f:
#             config_data = yaml.safe_load(f)
#         # You might want to merge this with the default config object
#         # or return a new object/dict
#         log.info(f"Loaded configuration from {yaml_path}")
#         return config_data
#     except FileNotFoundError:
#         log.error(f"Configuration file not found: {yaml_path}")
#         return None
#     except Exception as e:
#         log.error(f"Error loading configuration from {yaml_path}: {e}")
#         return None


# Example usage (for testing purposes)
if __name__ == '__main__':
    print("--- Running Utilities Test ---")

    # Ensure output directories exist for logging and saving
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting utilities test script.")

    # --- Create Dummy Volume ---
    log.warning("Creating dummy volume data for saving test.")
    z_dim, y_dim, x_dim = config.RECON_VOLUME_SHAPE
    dummy_volume = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
    center_z, center_y, center_x = z_dim // 2, y_dim // 2, x_dim // 2
    radius = min(z_dim, y_dim, x_dim) // 4
    z, y, x = np.ogrid[:z_dim, :y_dim, :x_dim]
    mask_sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
    dummy_volume[mask_sphere] = 100.0
    log.info(f"Created dummy volume with shape: {dummy_volume.shape}")

    # --- Test Saving Volume ---
    filename_base = "test_saved_volume"
    success_mha = save_volume(dummy_volume, f"{filename_base}.mha", cfg=config, file_format='mha')
    success_tif = save_volume(dummy_volume, f"{filename_base}.tif", cfg=config, file_format='tif')
    success_npy = save_volume(dummy_volume, f"{filename_base}.npy", cfg=config, file_format='npy')

    if success_mha or success_tif or success_npy:
        log.info("Volume saving test completed.")
        print(f"\nCheck output volume files in: {config.RECON_VOLUME_DIR}")
    else:
        log.error("Volume saving test failed for all formats.")

    # --- Test other utils if added ---
    # e.g., test config loading

    log.info("Utilities test script finished.")
    print(f"\nCheck log file for details: {config.LOG_FILE}")
    print("--- End of Utilities Test ---")

