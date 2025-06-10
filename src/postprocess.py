# -*- coding: utf-8 -*-
"""
Module for postprocessing the reconstructed CT volume.

Includes functions for:
- Filtering (median, gaussian)
- Intensity scaling and clipping
- Applying a cylindrical mask
"""

import numpy as np
import logging
import scipy.ndimage as ndi
from skimage.filters import gaussian as skimage_gaussian # Use skimage gaussian for better boundary handling potentially
from skimage.morphology import ball, disk # For potential morphological operations if needed
import time

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
    logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def apply_post_filtering(volume: np.ndarray, cfg: object) -> np.ndarray:
    """
    Applies filtering to the reconstructed volume to reduce noise.

    Args:
        volume: The 3D reconstructed volume (Z, Y, X).
        cfg: The configuration object containing filter settings.

    Returns:
        The filtered 3D volume.
    """
    if not cfg.PERFORM_POST_FILTERING:
        log.info("Post-reconstruction filtering disabled.")
        return volume

    method = cfg.POST_FILTER_METHOD.lower()
    params = cfg.POST_FILTER_PARAMS
    log.info(f"Applying post-reconstruction filter: {method} with params: {params}")
    start_time = time.time()
    filtered_volume = volume # Start with original if method unknown

    try:
        if method == 'median':
            filter_size = params.get('size', 3)
            log.info(f"Applying 3D median filter with size {filter_size}...")
            # Use scipy.ndimage.median_filter for N-dimensional median filtering
            filtered_volume = ndi.median_filter(volume, size=filter_size)

        elif method == 'gaussian':
            sigma = params.get('sigma', 1.0)
            truncate = params.get('truncate', 4.0)
            mode = params.get('mode', 'reflect') # Boundary handling mode
            log.info(f"Applying 3D Gaussian filter with sigma={sigma}, truncate={truncate}, mode='{mode}'...")
            # Use scikit-image gaussian for potentially better boundary handling
            filtered_volume = skimage_gaussian(
                volume,
                sigma=sigma,
                mode=mode,
                truncate=truncate,
                channel_axis=None, # Treat as single channel 3D volume
                preserve_range=True # Keep original intensity range
            )
            # Or use scipy: filtered_volume = ndi.gaussian_filter(volume, sigma=sigma, mode=mode, truncate=truncate)


        # Add other filters if needed (e.g., bilateral - might be very slow in 3D)
        # elif method == 'bilateral':
        #     log.warning("Bilateral filter selected - can be very slow for 3D volumes.")
        #     # Requires skimage.filters.rank.bilateral or custom implementation
        #     # sigma_color = params.get('sigma_color', 0.1 * (volume.max() - volume.min()))
        #     # sigma_spatial = params.get('sigma_spatial', 1.0)
        #     # filtered_volume = skimage.restoration.denoise_bilateral(volume, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)


        else:
            log.warning(f"Unsupported post-filter method: '{method}'. Returning original volume.")
            return volume

        end_time = time.time()
        log.info(f"Filtering completed in {end_time - start_time:.2f} seconds.")
        return filtered_volume.astype(volume.dtype) # Ensure original dtype

    except Exception as e:
        log.error(f"Error applying post-filter method '{method}': {e}")
        return volume # Return original volume on error


def apply_intensity_scaling(volume: np.ndarray, cfg: object) -> np.ndarray:
    """
    Scales the volume intensity to a specified range and optionally clips values.

    Args:
        volume: The 3D reconstructed volume.
        cfg: The configuration object with scaling/clipping settings.

    Returns:
        The intensity-adjusted 3D volume.
    """
    if not cfg.PERFORM_INTENSITY_SCALING:
        log.info("Intensity scaling/clipping disabled.")
        return volume

    min_val, max_val = cfg.INTENSITY_SCALE_RANGE
    clip = cfg.INTENSITY_CLIP
    log.info(f"Applying intensity scaling to range [{min_val}, {max_val}]. Clip: {clip}")

    # Get current min/max of the volume data
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    log.debug(f"Original volume intensity range: [{vol_min:.4f}, {vol_max:.4f}]")

    if vol_max - vol_min < EPSILON:
        log.warning("Volume intensity range is near zero. Scaling might produce unexpected results.")
        # Handle constant volume case: scale to the middle of the target range or min_val
        scaled_volume = np.full_like(volume, (min_val + max_val) / 2.0)
    else:
        # Perform linear scaling: NewValue = ((OldValue - OldMin) / (OldMax - OldMin)) * (NewMax - NewMin) + NewMin
        scaled_volume = ((volume - vol_min) / (vol_max - vol_min)) * (max_val - min_val) + min_val

    if clip:
        log.debug("Clipping scaled volume to target range.")
        scaled_volume = np.clip(scaled_volume, min_val, max_val)

    log.info("Intensity scaling applied.")
    # Consider the output dtype. If scaling to [0, 255], maybe convert to uint8?
    # For now, keep float32 for potential further processing.
    return scaled_volume.astype(np.float32)


def apply_masking(volume: np.ndarray, cfg: object) -> np.ndarray:
    """
    Applies a cylindrical mask to the volume.
    Assumes the Z-axis is the cylinder axis.

    Args:
        volume: The 3D reconstructed volume (Z, Y, X).
        cfg: The configuration object with masking settings.

    Returns:
        The masked 3D volume.
    """
    if not cfg.PERFORM_MASKING:
        log.info("Masking disabled.")
        return volume

    radius_ratio = cfg.MASK_RADIUS_RATIO
    log.info(f"Applying cylindrical mask with radius ratio: {radius_ratio}")
    start_time = time.time()

    z_dim, y_dim, x_dim = volume.shape
    if y_dim != x_dim:
        log.warning(f"Volume Y ({y_dim}) and X ({x_dim}) dimensions are different. Cylindrical mask assumes center based on square XY plane.")

    # Center of the XY plane
    center_y = (y_dim - 1) / 2.0
    center_x = (x_dim - 1) / 2.0

    # Calculate the maximum radius based on the smaller of X/Y dimensions
    max_radius = min(y_dim, x_dim) / 2.0
    mask_radius = max_radius * radius_ratio
    mask_radius_sq = mask_radius ** 2
    log.debug(f"Calculated mask radius: {mask_radius:.2f} pixels")

    # Create coordinate grids for Y and X
    y_coords, x_coords = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')

    # Calculate squared distance from center for each pixel in the XY plane
    dist_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2

    # Create the 2D mask for the XY plane
    xy_mask = dist_sq <= mask_radius_sq

    # Expand the 2D mask to 3D (apply same mask to all Z slices)
    # This creates a cylindrical mask along the Z axis
    mask_3d = np.repeat(xy_mask[np.newaxis, :, :], z_dim, axis=0)

    # Apply the mask
    masked_volume = np.copy(volume) # Work on a copy
    masked_volume[~mask_3d] = 0 # Set values outside the mask to 0 (or another background value)

    end_time = time.time()
    log.info(f"Masking applied in {end_time - start_time:.2f} seconds.")
    return masked_volume.astype(volume.dtype)


# --- Main Postprocessing Pipeline Function ---

def postprocess_volume(volume: np.ndarray, cfg: object) -> np.ndarray:
    """
    Runs the full postprocessing pipeline on the reconstructed volume.

    Args:
        volume: The 3D reconstructed volume (Z, Y, X).
        cfg: The configuration object.

    Returns:
        The fully postprocessed 3D volume.
    """
    log.info("--- Starting Postprocessing Pipeline ---")
    if volume is None:
        log.error("Input volume for postprocessing is None. Skipping.")
        return None

    processed_volume = volume # Start with the input volume

    # 1. Filtering
    processed_volume = apply_post_filtering(processed_volume, cfg)
    log.info("Post-filtering step completed.")

    # 2. Masking (Apply mask before intensity scaling?)
    # Often masking is done after filtering but before final scaling/quantization.
    processed_volume = apply_masking(processed_volume, cfg)
    log.info("Masking step completed.")

    # 3. Intensity Scaling / Clipping
    processed_volume = apply_intensity_scaling(processed_volume, cfg)
    log.info("Intensity scaling step completed.")

    log.info("--- Postprocessing Pipeline Finished Successfully ---")
    return processed_volume


# Example usage (for testing purposes)
if __name__ == '__main__':
    import os
    import imageio.v3 as iio # To save sample slice

    print("--- Running Postprocessor Test ---")

    # Ensure output directories exist for logging and saving plots
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting postprocessor test script.")

    # --- Create Dummy Reconstructed Volume ---
    log.warning("Using dummy reconstructed volume data for postprocessing test.")
    z_dim, y_dim, x_dim = config.RECON_VOLUME_SHAPE
    dummy_volume = np.random.rand(z_dim, y_dim, x_dim).astype(np.float32) * 100
    # Add some structure (e.g., a sphere)
    center_z, center_y, center_x = z_dim // 2, y_dim // 2, x_dim // 2
    radius = min(z_dim, y_dim, x_dim) // 4
    z, y, x = np.ogrid[:z_dim, :y_dim, :x_dim]
    mask_sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
    dummy_volume[mask_sphere] += 50 # Increase intensity inside sphere
    dummy_volume += np.random.normal(0, 10, size=dummy_volume.shape) # Add some noise
    log.info(f"Created dummy volume with shape: {dummy_volume.shape}, range: [{np.min(dummy_volume):.2f}, {np.max(dummy_volume):.2f}]")

    # --- Run Postprocessing ---
    postprocessed_vol = postprocess_volume(dummy_volume, config)

    # --- Process Results ---
    if postprocessed_vol is not None:
        log.info(f"Postprocessing test successful. Final volume shape: {postprocessed_vol.shape}")
        log.info(f"Final Volume min/max values: {np.min(postprocessed_vol):.4f} / {np.max(postprocessed_vol):.4f}")

        # Optional: Save a central slice for visual inspection
        try:
            slice_idx = postprocessed_vol.shape[0] // 2 # Middle Z slice
            post_slice = postprocessed_vol[slice_idx, :, :]
            post_slice_path = os.path.join(config.RECON_SLICES_DIR, f'test_postprocessed_slice.png')

            # Normalize slice for saving as image (use final range from config if scaled)
            if config.PERFORM_INTENSITY_SCALING:
                 min_out, max_out = config.INTENSITY_SCALE_RANGE
                 post_slice_norm = (post_slice - min_out) / (max_out - min_out + EPSILON)
            else:
                 post_slice_norm = (post_slice - np.min(post_slice)) / (np.max(post_slice) - np.min(post_slice) + EPSILON)

            post_slice_norm = np.clip(post_slice_norm, 0, 1) # Ensure range [0, 1]
            post_slice_uint8 = (post_slice_norm * 255).astype(np.uint8)

            iio.imwrite(post_slice_path, post_slice_uint8, prefer_uint8=True)
            log.info(f"Saved sample postprocessed slice to {post_slice_path}")

        except ImportError:
            log.warning("imageio not found. Cannot save sample postprocessed slice.")
        except Exception as e:
            log.error(f"Error saving sample postprocessed slice: {e}")
    else:
        log.error("Postprocessing test failed.")

    log.info("Postprocessor test script finished.")
    print("--- End of Postprocessor Test ---")
    print(f"Check log file for details: {config.LOG_FILE}")

