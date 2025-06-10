# -*- coding: utf-8 -*-
"""
Module for preprocessing CT projection data.

Includes functions for:
- Flat-field and dark-field correction
- Logarithmic conversion (transmission to attenuation)
- Optional ring artifact removal (placeholder)
- Placeholders for beam hardening and scatter correction
"""

import numpy as np
import logging
from typing import Optional, Dict, Any
import scipy.ndimage as ndi
# from skimage.restoration import denoise_tv_chambolle # Example for ring removal

# Import configuration settings
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    raise

# Configure logging (inherits config from data_loader if run together, or sets up its own)
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Small epsilon to prevent division by zero or log(0)
EPSILON = 1e-9

def apply_flat_dark_correction(
    projections: np.ndarray,
    flat_field: Optional[np.ndarray],
    dark_field: Optional[np.ndarray],
    cfg: object
) -> Optional[np.ndarray]:
    """
    Applies flat-field and dark-field correction to projection data.

    Formula: Corrected = (Projection - Dark) / (Flat - Dark + EPSILON)

    Args:
        projections: 3D array of raw projection images (n_proj, height, width).
        flat_field: 2D array (or 3D if per_projection) of flat-field images.
        dark_field: 2D array (or 3D if per_projection) of dark-field images.
        cfg: The configuration object.

    Returns:
        Corrected projection data as a 3D float32 array, or None if correction
        cannot be performed due to missing data when enabled.
    """
    if not cfg.PERFORM_FLAT_FIELD_CORRECTION and not cfg.PERFORM_DARK_FIELD_CORRECTION:
        log.info("Flat and dark field correction disabled. Returning raw projections.")
        return projections.astype(np.float32) # Ensure float type

    corrected_projections = projections.astype(np.float32) # Work with float copies

    # --- Dark Field Correction ---
    if cfg.PERFORM_DARK_FIELD_CORRECTION:
        if dark_field is None:
            log.error("Dark field correction enabled, but no dark field data provided/loaded.")
            return None
        log.info("Applying dark field correction...")
        # Ensure dark_field is broadcastable (e.g., add axes for 2D dark on 3D proj)
        if dark_field.ndim == 2 and corrected_projections.ndim == 3:
            dark_field_b = dark_field[np.newaxis, :, :] # Add projection axis
        else:
            dark_field_b = dark_field # Assume compatible shape (e.g., per-projection darks)

        if dark_field_b.shape[1:] != corrected_projections.shape[1:]:
             log.error(f"Dark field shape {dark_field.shape} incompatible with projection shape {corrected_projections.shape}.")
             return None
        if dark_field_b.shape[0] != 1 and dark_field_b.shape[0] != corrected_projections.shape[0]:
             log.error(f"Dark field projection number {dark_field_b.shape[0]} incompatible with projection number {corrected_projections.shape[0]}.")
             return None

        corrected_projections -= dark_field_b
        log.debug("Dark field subtraction complete.")
    else:
        log.info("Dark field correction disabled.")
        # If only flat field is enabled, we still need a baseline (assume dark=0)
        dark_field_b = np.zeros_like(projections[0], dtype=np.float32)[np.newaxis, :, :] # Use 0 as dark reference


    # --- Flat Field Correction ---
    if cfg.PERFORM_FLAT_FIELD_CORRECTION:
        if flat_field is None:
            log.error("Flat field correction enabled, but no flat field data provided/loaded.")
            return None
        log.info("Applying flat field correction...")

        if flat_field.ndim == 2 and corrected_projections.ndim == 3:
            flat_field_b = flat_field[np.newaxis, :, :]
        else:
            flat_field_b = flat_field

        if flat_field_b.shape[1:] != corrected_projections.shape[1:]:
             log.error(f"Flat field shape {flat_field.shape} incompatible with projection shape {corrected_projections.shape}.")
             return None
        if flat_field_b.shape[0] != 1 and flat_field_b.shape[0] != corrected_projections.shape[0]:
             log.error(f"Flat field projection number {flat_field_b.shape[0]} incompatible with projection number {corrected_projections.shape[0]}.")
             return None

        # Calculate the denominator (Flat - Dark)
        # Use the same dark_field_b as subtracted above (or the zero array if dark correction was off)
        denominator = flat_field_b - dark_field_b
        denominator[denominator <= 0] = EPSILON # Prevent division by zero or negative values
        log.debug(f"Denominator (Flat - Dark) calculated. Min value: {np.min(denominator)}")

        # Apply correction: (Projection - Dark) / (Flat - Dark)
        corrected_projections /= denominator
        log.info("Flat field correction applied.")

    else:
        log.info("Flat field correction disabled.")
        # If only dark correction was applied, the result is (Proj - Dark)
        # If neither was applied, result is Proj.

    # Clip values to a reasonable range (e.g., 0 to 1 for transmission) if needed,
    # although log conversion handles <0 later. Values > 1 might indicate issues.
    corrected_projections = np.clip(corrected_projections, EPSILON, None) # Clip below EPSILON
    if np.any(corrected_projections > 1.5): # Arbitrary threshold warning
         log.warning("Corrected transmission values significantly > 1 detected. Check flat/dark fields.")

    return corrected_projections


def apply_log_conversion(projections: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Converts transmission data to attenuation data using the negative logarithm.

    Formula: Attenuation = -log(Transmission)

    Args:
        projections: 3D array of transmission data (corrected projections).
        cfg: The configuration object.

    Returns:
        Attenuation data (sinogram) as a 3D float32 array, or None on error.
    """
    if not cfg.PERFORM_LOG_CONVERSION:
        log.info("Log conversion disabled. Returning transmission data.")
        return projections # Should already be float32

    log.info("Applying negative logarithm conversion (Transmission -> Attenuation)...")
    if projections is None:
        log.error("Cannot apply log conversion: input projections are None.")
        return None

    # Ensure no non-positive values before taking log
    if np.any(projections <= 0):
        log.warning(f"Detected {np.sum(projections <= 0)} non-positive values in transmission data before log. Clipping to {EPSILON}.")
        projections = np.maximum(projections, EPSILON) # Clip values <= 0 to EPSILON

    try:
        attenuation_data = -np.log(projections)
        log.info("Log conversion successful.")
        # Check for extreme values which might indicate problems
        if np.any(np.isinf(attenuation_data)) or np.any(np.isnan(attenuation_data)):
             log.warning("Infinite or NaN values detected after log conversion. Check input data.")
             attenuation_data = np.nan_to_num(attenuation_data, copy=False, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)

        return attenuation_data.astype(np.float32)

    except Exception as e:
        log.error(f"Error during log conversion: {e}")
        return None


def apply_ring_removal(sinogram: np.ndarray, cfg: object) -> np.ndarray:
    """
    Applies ring artifact removal algorithm to the sinogram (attenuation data).

    Placeholder function - specific implementation depends on chosen method.

    Args:
        sinogram: 3D array of attenuation data (n_proj, height, width).
                  Note: Ring removal is often applied slice-by-slice to the
                  (n_proj, n_detector_pixels) sinogram representation.
                  This function assumes input is (n_proj, height, width) and
                  needs reshaping or slice-wise processing.
        cfg: The configuration object containing RING_REMOVAL_METHOD and PARAMS.

    Returns:
        The sinogram with ring artifacts reduced.
    """
    if not cfg.PERFORM_RING_REMOVAL:
        log.info("Ring removal disabled.")
        return sinogram

    method = cfg.RING_REMOVAL_METHOD
    params = cfg.RING_REMOVAL_PARAMS
    log.info(f"Applying ring removal using method: {method} with params: {params}")

    # Ring artifacts appear as vertical stripes in the sinogram view (angle vs detector pixel).
    # Processing is typically done independently for each detector row (slice).
    processed_sinogram = np.copy(sinogram) # Work on a copy
    num_proj, height, width = sinogram.shape

    # Reshape or iterate slice-by-slice
    for i in range(height): # Iterate through detector rows (slices)
        slice_sinogram = sinogram[:, i, :] # Shape (n_proj, width)
        log.debug(f"Processing slice {i}/{height} for ring removal.")

        try:
            if method == 'median_filter':
                # Simple approach: Apply a horizontal median filter to the sinogram slice
                filter_size = params.get('size', 3)
                slice_filtered = ndi.median_filter(slice_sinogram, size=(1, filter_size))
                # Subtract the smoothed version to get the rings, then subtract from original
                rings = slice_sinogram - slice_filtered
                processed_sinogram[:, i, :] = slice_sinogram - rings # Or apply smoothing directly? Check literature.
                log.warning("Median filter ring removal is a basic approach, may blur data.")

            elif method == 'wavelet_fft':
                 # More advanced method using wavelets and FFT filtering
                 # Requires libraries like PyWavelets, skimage, etc.
                 # Example using skimage (conceptual - needs refinement):
                 # from skimage.restoration import denoise_wavelet
                 # sigma = params.get('sigma', 1.0)
                 # level = params.get('level', 5)
                 # # Denoise in detector direction (axis=1)
                 # smoothed_slice = denoise_wavelet(slice_sinogram, sigma=sigma, wavelet='db4', mode='soft', wavelet_levels=level, channel_axis=None) # Check channel_axis
                 # processed_sinogram[:, i, :] = smoothed_slice # Or subtract difference? Consult method details.
                 log.warning(f"Ring removal method '{method}' requires specific implementation/libraries (e.g., PyWavelets, skimage). Placeholder used.")
                 # Keep original slice for now if method not implemented
                 processed_sinogram[:, i, :] = slice_sinogram


            # Add other methods based on literature/research
            # elif method == 'some_other_method':
            #    processed_sinogram[:, i, :] = some_other_ring_removal(slice_sinogram, **params)

            else:
                log.warning(f"Ring removal method '{method}' not implemented. Skipping.")
                processed_sinogram[:, i, :] = slice_sinogram # Keep original

        except Exception as e:
            log.error(f"Error applying ring removal method '{method}' on slice {i}: {e}")
            processed_sinogram[:, i, :] = slice_sinogram # Revert to original on error

    log.info("Ring removal processing finished.")
    return processed_sinogram.astype(np.float32)


def apply_beam_hardening_correction(projections: np.ndarray, cfg: object) -> np.ndarray:
    """
    Placeholder for beam hardening correction.

    Args:
        projections: 3D array of attenuation data (sinogram).
        cfg: Configuration object.

    Returns:
        Corrected sinogram data.
    """
    if not cfg.PERFORM_BEAM_HARDENING_CORRECTION:
        log.info("Beam hardening correction disabled.")
        return projections

    method = cfg.BEAM_HARDENING_METHOD
    params = cfg.BEAM_HARDENING_PARAMS
    log.warning(f"Beam hardening correction enabled (method: {method}), but it is NOT IMPLEMENTED yet. Returning uncorrected data.")
    # --- Implementation required based on chosen method ---
    # e.g., polynomial correction, linearization based on material, etc.
    # corrected_projections = perform_bh_correction(projections, method, params)
    # return corrected_projections
    return projections


def apply_scatter_correction(projections: np.ndarray, cfg: object) -> np.ndarray:
    """
    Placeholder for scatter correction.

    Args:
        projections: 3D array of attenuation data (sinogram) or transmission data.
                     (Depends on when scatter correction is typically applied).
        cfg: Configuration object.

    Returns:
        Corrected projection data.
    """
    if not cfg.PERFORM_SCATTER_CORRECTION:
        log.info("Scatter correction disabled.")
        return projections

    method = cfg.SCATTER_METHOD
    params = cfg.SCATTER_PARAMS
    log.warning(f"Scatter correction enabled (method: {method}), but it is NOT IMPLEMENTED yet. Returning uncorrected data.")
    # --- Implementation required based on chosen method ---
    # e.g., beam stop extrapolation, kernel methods, Monte Carlo simulation based, etc.
    # corrected_projections = perform_scatter_correction(projections, method, params)
    # return corrected_projections
    return projections


# --- Main Preprocessing Pipeline Function ---

def preprocess_data(
    raw_projections: np.ndarray,
    flat_field: Optional[np.ndarray],
    dark_field: Optional[np.ndarray],
    cfg: object
) -> Optional[np.ndarray]:
    """
    Runs the full preprocessing pipeline on the raw projection data.

    Args:
        raw_projections: 3D array of raw projection images.
        flat_field: 2D or 3D array of flat-field images.
        dark_field: 2D or 3D array of dark-field images.
        cfg: The configuration object.

    Returns:
        The fully preprocessed projection data (attenuation sinogram),
        ready for reconstruction, or None if a critical step fails.
    """
    log.info("--- Starting Preprocessing Pipeline ---")

    # 1. Flat/Dark Field Correction (Yields Transmission Data)
    projections = apply_flat_dark_correction(raw_projections, flat_field, dark_field, cfg)
    if projections is None:
        log.error("Preprocessing failed during flat/dark field correction.")
        return None
    log.info("Flat/Dark field correction step completed.")

    # 2. Scatter Correction (Placeholder - Apply before or after log?)
    # Scatter is often estimated/removed from transmission data. Check literature.
    projections = apply_scatter_correction(projections, cfg)
    if projections is None:
        log.error("Preprocessing failed during scatter correction.")
        return None
    log.info("Scatter correction step completed (placeholder).")


    # 3. Logarithmic Conversion (Transmission -> Attenuation)
    sinogram = apply_log_conversion(projections, cfg)
    if sinogram is None:
        log.error("Preprocessing failed during log conversion.")
        return None
    log.info("Log conversion step completed.")

    # 4. Beam Hardening Correction (Placeholder - Applied to attenuation data)
    sinogram = apply_beam_hardening_correction(sinogram, cfg)
    if sinogram is None:
        log.error("Preprocessing failed during beam hardening correction.")
        return None
    log.info("Beam hardening correction step completed (placeholder).")

    # 5. Ring Artifact Removal (Applied to attenuation sinogram)
    sinogram = apply_ring_removal(sinogram, cfg)
    if sinogram is None:
        log.error("Preprocessing failed during ring removal.")
        return None
    log.info("Ring removal step completed.")

    log.info("--- Preprocessing Pipeline Finished Successfully ---")
    return sinogram


# Example usage (for testing purposes)
if __name__ == '__main__':
    import data_loader # Import data_loader for testing

    print("--- Running Preprocessor Test ---")

    # Ensure output directories exist for logging
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting preprocessor test script.")

    # --- Load Test Data ---
    # This assumes data_loader can find some data based on config.py settings
    # You might need to create dummy data for testing if real data isn't available
    log.info("Loading data for preprocessing test...")
    raw_proj_data = data_loader.load_projections(config)

    if raw_proj_data is not None:
        detected_shape = (raw_proj_data.shape[1], raw_proj_data.shape[2])
        flat_data = data_loader.load_flat_fields(config, detected_shape)
        dark_data = data_loader.load_dark_fields(config, detected_shape)

        # --- Run Preprocessing ---
        log.info("Running the main preprocessing function...")
        preprocessed_sinogram = preprocess_data(raw_proj_data, flat_data, dark_data, config)

        if preprocessed_sinogram is not None:
            log.info(f"Preprocessing successful. Final sinogram shape: {preprocessed_sinogram.shape}")
            log.info(f"Sinogram data type: {preprocessed_sinogram.dtype}")
            log.info(f"Sinogram min/max values: {np.min(preprocessed_sinogram):.4f} / {np.max(preprocessed_sinogram):.4f}")

            # Optional: Save a slice of the sinogram for visual inspection
            try:
                import matplotlib.pyplot as plt
                slice_idx = preprocessed_sinogram.shape[1] // 2 # Middle slice
                sino_slice_path = os.path.join(config.PLOT_DIR, 'preprocessed_sinogram_slice.png')
                plt.figure(figsize=(10, 5))
                plt.imshow(preprocessed_sinogram[:, slice_idx, :], cmap=config.VISUALIZATION_CMAP, aspect='auto')
                plt.title(f'Preprocessed Sinogram (Slice {slice_idx})')
                plt.xlabel('Detector Pixel')
                plt.ylabel('Projection Angle Index')
                plt.colorbar(label='Attenuation')
                plt.tight_layout()
                plt.savefig(sino_slice_path)
                plt.close()
                log.info(f"Saved sample preprocessed sinogram slice to {sino_slice_path}")
            except ImportError:
                log.warning("Matplotlib not found. Cannot save sample sinogram plot.")
            except Exception as e:
                log.error(f"Error saving sample sinogram plot: {e}")

        else:
            log.error("Preprocessing pipeline failed.")
    else:
        log.error("Failed to load raw projection data. Cannot run preprocessing test.")

    log.info("Preprocessor test script finished.")
    print("--- End of Preprocessor Test ---")
    print(f"Check log file for details: {config.LOG_FILE}")
