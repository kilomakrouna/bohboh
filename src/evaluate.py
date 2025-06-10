# -*- coding: utf-8 -*-
"""
Module for evaluating the quality of the reconstructed CT volume.

Compares the reconstruction against a ground truth volume (if available)
using metrics like Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR),
and Structural Similarity Index (SSIM).
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Any

# Import configuration settings
try:
    import config
    # Define EPSILON using config if available, otherwise default
    EPSILON = getattr(config, 'EPSILON', 1e-9)
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    EPSILON = 1e-9 # Default epsilon if config fails
    raise

# Import metrics from scikit-image
try:
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
    SKIMAGE_METRICS_AVAILABLE = True
    log_msg = "scikit-image metrics library found."
except ImportError:
    SKIMAGE_METRICS_AVAILABLE = False
    log_msg = "scikit-image not found or metrics module unavailable. Evaluation functions (MSE, PSNR, SSIM) will not work."
    # Raise an error or handle gracefully depending on whether evaluation is critical
    # For now, log an error if evaluation is enabled later.

# Configure logging
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=getattr(config, config.LOG_LEVEL, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

log.info(log_msg) # Log skimage availability status


def calculate_metrics(
    reconstructed_volume: np.ndarray,
    ground_truth_volume: np.ndarray,
    cfg: object
) -> Optional[Dict[str, float]]:
    """
    Calculates specified evaluation metrics between the reconstructed volume
    and the ground truth volume.

    Args:
        reconstructed_volume: The 3D reconstructed volume (Z, Y, X).
        ground_truth_volume: The 3D ground truth volume (Z, Y, X).
        cfg: The configuration object containing EVALUATION_METRICS list.

    Returns:
        A dictionary containing the calculated metric names and their values,
        or None if evaluation cannot be performed.
    """
    if not cfg.PERFORM_EVALUATION:
        log.info("Evaluation disabled in config. Skipping metric calculation.")
        return None

    if not SKIMAGE_METRICS_AVAILABLE:
        log.error("Cannot calculate metrics: scikit-image metrics module not available.")
        return None

    if reconstructed_volume is None or ground_truth_volume is None:
        log.error("Cannot calculate metrics: Reconstructed or ground truth volume is None.")
        return None

    if reconstructed_volume.shape != ground_truth_volume.shape:
        log.error(f"Shape mismatch: Reconstructed volume {reconstructed_volume.shape} vs Ground Truth {ground_truth_volume.shape}. Cannot calculate metrics.")
        # Optional: Add logic here to resize/resample one volume to match the other if appropriate
        return None

    metrics_to_calculate = cfg.EVALUATION_METRICS
    results = {}
    log.info(f"Calculating evaluation metrics: {metrics_to_calculate}")

    # --- Ensure data types are appropriate for metrics ---
    # Convert to float for calculations, handle potential range differences
    recon_eval = reconstructed_volume.astype(np.float64)
    gt_eval = ground_truth_volume.astype(np.float64)

    # Optional: Normalize intensity ranges if they differ significantly
    # This depends on whether metrics should compare absolute values or relative structures
    # Example normalization to [0, 1] (use with caution):
    # norm_recon = (recon_eval - np.min(recon_eval)) / (np.max(recon_eval) - np.min(recon_eval) + EPSILON)
    # norm_gt = (gt_eval - np.min(gt_eval)) / (np.max(gt_eval) - np.min(gt_eval) + EPSILON)
    # recon_eval = norm_recon
    # gt_eval = norm_gt

    # Determine data range for PSNR and SSIM
    # Use ground truth range or a theoretical max if known (e.g., max attenuation value)
    data_range = np.max(gt_eval) - np.min(gt_eval)
    if data_range < EPSILON:
        log.warning("Ground truth data range is near zero. PSNR and SSIM might be undefined or misleading.")
        # Handle constant ground truth case if necessary
        data_range = None # Let skimage handle it or set default

    for metric_name in metrics_to_calculate:
        metric_name_lower = metric_name.lower()
        try:
            if metric_name_lower == 'mse':
                mse_val = mean_squared_error(gt_eval, recon_eval)
                results['MSE'] = mse_val
                log.info(f"MSE: {mse_val:.6f}")

            elif metric_name_lower == 'psnr':
                if data_range is None or data_range < EPSILON:
                     log.warning("Cannot calculate PSNR: data_range is zero or undefined.")
                     results['PSNR'] = np.nan # Or some indicator value
                else:
                    psnr_val = peak_signal_noise_ratio(gt_eval, recon_eval, data_range=data_range)
                    results['PSNR'] = psnr_val
                    log.info(f"PSNR: {psnr_val:.4f} dB")

            elif metric_name_lower == 'ssim':
                # SSIM is often calculated slice-by-slice or using a 3D version if available
                # skimage's ssim is typically 2D but can handle 3D if specified
                # Check documentation for multichannel/3D usage. Default might compare slices.
                # For 3D comparison, set win_size appropriately (must be odd and <= dimensions)
                # data_range is important here too.
                win_size = min(7, *[d for d in gt_eval.shape if d >= 7]) # Ensure window size is odd and fits dims
                if win_size % 2 == 0: win_size -=1 # Make odd if even

                if data_range is None or data_range < EPSILON:
                    log.warning("Cannot calculate SSIM: data_range is zero or undefined.")
                    results['SSIM'] = np.nan
                elif win_size < 3:
                    log.warning(f"Cannot calculate SSIM: Volume dimensions too small for default window size. Min dim: {min(gt_eval.shape)}")
                    results['SSIM'] = np.nan
                else:
                    # Use gaussian weights by default, multichannel=False assumes grayscale 3D volume
                    ssim_val = structural_similarity(
                        gt_eval,
                        recon_eval,
                        data_range=data_range,
                        win_size=win_size,
                        channel_axis=None, # Treat as single channel 3D volume
                        gaussian_weights=True,
                        sigma=1.5, # Standard deviation for Gaussian weights
                        use_sample_covariance=False # Use default covariance calculation
                    )
                    results['SSIM'] = ssim_val
                    log.info(f"SSIM: {ssim_val:.6f}")

            else:
                log.warning(f"Unsupported metric specified in config: '{metric_name}'. Skipping.")

        except Exception as e:
            log.error(f"Error calculating metric '{metric_name}': {e}")
            results[metric_name.upper()] = np.nan # Indicate error

    log.info("Finished calculating metrics.")
    return results


# Example usage (for testing purposes)
if __name__ == '__main__':
    import data_loader # To load ground truth if available

    print("--- Running Evaluator Test ---")

    # Ensure output directories exist for logging
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting evaluator test script.")

    # --- Load or Create Dummy Data ---
    log.warning("Using dummy reconstructed and ground truth volumes for evaluator test.")

    # Attempt to load real ground truth if configured and available
    gt_volume = data_loader.load_ground_truth(config)

    # Create dummy reconstructed volume (e.g., ground truth + noise)
    recon_volume = None
    if gt_volume is not None:
        log.info(f"Using loaded ground truth volume (shape: {gt_volume.shape}) for test.")
        # Create a dummy reconstruction based on ground truth
        noise_level = 0.1 * (np.max(gt_volume) - np.min(gt_volume)) # 10% noise relative to range
        recon_volume = gt_volume + np.random.normal(0, noise_level, size=gt_volume.shape)
        recon_volume = recon_volume.astype(gt_volume.dtype) # Match dtype
        log.info("Created dummy reconstruction by adding noise to ground truth.")
    else:
        # If no ground truth loaded, create fully dummy data
        log.info("No ground truth loaded. Creating fully dummy data for test.")
        z_dim, y_dim, x_dim = config.RECON_VOLUME_SHAPE
        gt_volume = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
        recon_volume = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)

        # Add simple structure (e.g., sphere in GT, slightly different sphere in Recon)
        center_z, center_y, center_x = z_dim // 2, y_dim // 2, x_dim // 2
        radius = min(z_dim, y_dim, x_dim) // 4
        z, y, x = np.ogrid[:z_dim, :y_dim, :x_dim]

        mask_sphere_gt = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
        gt_volume[mask_sphere_gt] = 100.0

        mask_sphere_recon = (x - (center_x+1))**2 + (y - (center_y+1))**2 + (z - center_z)**2 <= (radius*0.9)**2 # Slightly shifted/smaller
        recon_volume[mask_sphere_recon] = 90.0 # Slightly different intensity
        recon_volume += np.random.normal(0, 5, size=recon_volume.shape) # Add noise

    log.info(f"GT shape: {gt_volume.shape}, Recon shape: {recon_volume.shape}")

    # --- Run Evaluation ---
    # Enable evaluation for the test run if it wasn't already
    original_eval_flag = config.PERFORM_EVALUATION
    config.PERFORM_EVALUATION = True # Temporarily enable for test
    log.info(f"Temporarily setting PERFORM_EVALUATION = True for test.")

    evaluation_results = calculate_metrics(recon_volume, gt_volume, config)

    # Restore original config setting
    config.PERFORM_EVALUATION = original_eval_flag
    log.info(f"Restored PERFORM_EVALUATION to {original_eval_flag}.")

    # --- Process Results ---
    if evaluation_results is not None:
        log.info("Evaluation test successful.")
        print("\n--- Evaluation Results ---")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.6f}")
        print("------------------------")
    else:
        log.error("Evaluation test failed or was skipped.")

    log.info("Evaluator test script finished.")
    print(f"\nCheck log file for details: {config.LOG_FILE}")
    print("--- End of Evaluator Test ---")

