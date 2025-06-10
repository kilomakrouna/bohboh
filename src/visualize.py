# -*- coding: utf-8 -*-
"""
Module for visualizing CT data and reconstruction results.

Includes functions for plotting:
- Sinogram slices
- Orthogonal slices (XY, XZ, YZ) of the reconstructed volume
- Saving individual slices as images
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import Optional, Tuple, Union
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

# --- Plotting Functions ---

def plot_sinogram_slice(
    sinogram: np.ndarray,
    slice_index: int,
    cfg: object,
    filename: str = "sinogram_slice.png"
) -> None:
    """
    Plots and optionally saves a specific slice (detector row) of the sinogram.

    Args:
        sinogram: 3D sinogram data (n_proj, height, width).
        slice_index: The index of the detector row (height dimension) to plot.
        cfg: The configuration object.
        filename: Name of the file to save the plot.
    """
    if sinogram is None:
        log.warning("Sinogram data is None. Cannot plot slice.")
        return
    if not (0 <= slice_index < sinogram.shape[1]):
        log.error(f"Invalid slice_index {slice_index} for sinogram height {sinogram.shape[1]}.")
        return

    log.info(f"Generating plot for sinogram slice {slice_index}...")
    slice_data = sinogram[:, slice_index, :] # Shape (n_proj, width)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        slice_data,
        cmap=cfg.VISUALIZATION_CMAP,
        extent=[0, slice_data.shape[1], cfg.ANGLE_END_DEG, cfg.ANGLE_START_DEG], # x=detector pixels, y=angles
        aspect='auto'
    )
    plt.title(f'Sinogram (Slice/Row {slice_index})')
    plt.xlabel('Detector Pixel (Width)')
    plt.ylabel('Projection Angle (degrees)')
    plt.colorbar(label='Attenuation Value')
    plt.tight_layout()

    if cfg.SAVE_PLOTS:
        save_path = os.path.join(cfg.PLOT_DIR, filename)
        try:
            plt.savefig(save_path)
            log.info(f"Saved sinogram slice plot to {save_path}")
        except Exception as e:
            log.error(f"Failed to save sinogram slice plot to {save_path}: {e}")

    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close() # Close the figure window if not showing interactively


def plot_reconstructed_slices(
    volume: np.ndarray,
    cfg: object,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    filename: str = "reconstructed_slices.png",
    suptitle: str = "Reconstructed Volume Slices"
) -> None:
    """
    Plots orthogonal slices (XY, XZ, YZ) of the reconstructed volume.

    Args:
        volume: The 3D reconstructed volume (Z, Y, X).
        cfg: The configuration object.
        slice_indices: Tuple of (z_index, y_index, x_index) for the slices.
                       If None, uses the middle indices.
        filename: Name of the file to save the plot.
        suptitle: Overall title for the figure.
    """
    if volume is None:
        log.warning("Reconstructed volume data is None. Cannot plot slices.")
        return

    z_dim, y_dim, x_dim = volume.shape

    if slice_indices is None:
        # Default to middle slices
        z_idx = z_dim // 2
        y_idx = y_dim // 2
        x_idx = x_dim // 2
    else:
        z_idx, y_idx, x_idx = slice_indices
        # Validate indices
        if not (0 <= z_idx < z_dim and 0 <= y_idx < y_dim and 0 <= x_idx < x_dim):
            log.error(f"Invalid slice indices ({z_idx}, {y_idx}, {x_idx}) for volume shape ({z_dim}, {y_dim}, {x_dim}). Using middle slices instead.")
            z_idx, y_idx, x_idx = z_dim // 2, y_dim // 2, x_dim // 2

    log.info(f"Generating orthogonal slice plot at indices (Z={z_idx}, Y={y_idx}, X={x_idx})...")

    # Extract slices
    slice_xy = volume[z_idx, :, :] # XY plane at specified Z
    slice_xz = volume[:, y_idx, :] # XZ plane at specified Y
    slice_yz = volume[:, :, x_idx] # YZ plane at specified X

    # Determine shared intensity range for consistent coloring
    vmin = np.min(volume)
    vmax = np.max(volume)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{suptitle} (Z={z_idx}, Y={y_idx}, X={x_idx})", fontsize=14)

    # Plot XY slice
    im1 = axes[0].imshow(slice_xy, cmap=cfg.VISUALIZATION_CMAP, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    axes[0].set_title(f'XY Plane (Z = {z_idx})')
    axes[0].set_xlabel('X Pixel Index')
    axes[0].set_ylabel('Y Pixel Index')
    axes[0].axhline(y_idx, color='r', lw=0.5, linestyle='--') # Line indicating YZ slice position
    axes[0].axvline(x_idx, color='g', lw=0.5, linestyle='--') # Line indicating XZ slice position

    # Plot XZ slice
    im2 = axes[1].imshow(slice_xz, cmap=cfg.VISUALIZATION_CMAP, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest', aspect=z_dim/x_dim if x_dim else 1)
    axes[1].set_title(f'XZ Plane (Y = {y_idx})')
    axes[1].set_xlabel('X Pixel Index')
    axes[1].set_ylabel('Z Pixel Index')
    axes[1].axhline(z_idx, color='b', lw=0.5, linestyle='--') # Line indicating XY slice position
    axes[1].axvline(x_idx, color='g', lw=0.5, linestyle='--') # Line indicating YZ slice position


    # Plot YZ slice
    im3 = axes[2].imshow(slice_yz, cmap=cfg.VISUALIZATION_CMAP, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest', aspect=z_dim/y_dim if y_dim else 1)
    axes[2].set_title(f'YZ Plane (X = {x_idx})')
    axes[2].set_xlabel('Y Pixel Index')
    axes[2].set_ylabel('Z Pixel Index')
    axes[2].axhline(z_idx, color='b', lw=0.5, linestyle='--') # Line indicating XY slice position
    axes[2].axvline(y_idx, color='r', lw=0.5, linestyle='--') # Line indicating XZ slice position


    # Add a single colorbar for all subplots
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7, label='Reconstructed Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if cfg.SAVE_PLOTS:
        save_path = os.path.join(cfg.PLOT_DIR, filename)
        try:
            plt.savefig(save_path)
            log.info(f"Saved reconstructed slices plot to {save_path}")
        except Exception as e:
            log.error(f"Failed to save reconstructed slices plot to {save_path}: {e}")

    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_volume_slices(
    volume: np.ndarray,
    cfg: object,
    axis: int = 0,
    slice_range: Optional[Union[int, range, list]] = None,
    output_dir: Optional[str] = None,
    prefix: str = "slice_",
    file_format: str = "png"
) -> None:
    """
    Saves individual slices or a range of slices from the volume as image files.

    Args:
        volume: The 3D reconstructed volume (Z, Y, X).
        cfg: The configuration object (used for default output dir).
        axis: The axis along which to slice (0=Z, 1=Y, 2=X).
        slice_range: Index, range object, or list of indices to save.
                     If None, saves the middle slice along the specified axis.
        output_dir: Directory to save slices. If None, uses RECON_SLICES_DIR from config.
        prefix: Prefix for the output filenames (e.g., "slice_Z_").
        file_format: Output image format (e.g., 'png', 'tif').
    """
    if volume is None:
        log.warning("Volume data is None. Cannot save slices.")
        return

    dims = volume.shape
    if not (0 <= axis < 3):
        log.error(f"Invalid axis {axis}. Must be 0, 1, or 2.")
        return

    if output_dir is None:
        output_dir = cfg.RECON_SLICES_DIR
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    axis_labels = ['Z', 'Y', 'X']
    filename_prefix = f"{prefix}{axis_labels[axis]}_"

    if slice_range is None:
        # Default to middle slice
        indices_to_save = [dims[axis] // 2]
    elif isinstance(slice_range, int):
        indices_to_save = [slice_range]
    elif isinstance(slice_range, range):
        indices_to_save = list(slice_range)
    elif isinstance(slice_range, list):
        indices_to_save = slice_range
    else:
        log.error("Invalid slice_range type. Use int, range, list, or None.")
        return

    log.info(f"Saving slices along axis {axis_labels[axis]} to {output_dir}...")
    saved_count = 0
    # Determine intensity range for consistent normalization (optional, could normalize each slice individually)
    vol_min = np.min(volume)
    vol_max = np.max(volume)

    for idx in indices_to_save:
        if not (0 <= idx < dims[axis]):
            log.warning(f"Slice index {idx} out of bounds for axis {axis} (size {dims[axis]}). Skipping.")
            continue

        if axis == 0: # Z-axis -> XY slice
            slice_data = volume[idx, :, :]
        elif axis == 1: # Y-axis -> XZ slice
            slice_data = volume[:, idx, :]
        else: # X-axis -> YZ slice
            slice_data = volume[:, :, idx]

        # Normalize slice data to [0, 1] for standard image formats
        if vol_max - vol_min > EPSILON:
            slice_norm = (slice_data - vol_min) / (vol_max - vol_min)
        else:
            slice_norm = np.zeros_like(slice_data) # Handle constant volume

        slice_norm = np.clip(slice_norm, 0, 1)

        # Convert to uint8 or uint16 based on format/preference
        if file_format.lower() in ['tif', 'tiff']:
             # Save as float32 TIFF or scale to uint16
             # slice_img = slice_data.astype(np.float32) # Save float data directly
             slice_img = (slice_norm * 65535).astype(np.uint16) # Scale to uint16
        else:
             slice_img = (slice_norm * 255).astype(np.uint8) # Scale to uint8 for PNG/JPG

        filename = f"{filename_prefix}{idx:04d}.{file_format}"
        save_path = os.path.join(output_dir, filename)

        try:
            iio.imwrite(save_path, slice_img)
            saved_count += 1
            log.debug(f"Saved slice {idx} to {save_path}")
        except Exception as e:
            log.error(f"Failed to save slice {idx} to {save_path}: {e}")

    log.info(f"Finished saving {saved_count} slices.")


# Example usage (for testing purposes)
if __name__ == '__main__':
    print("--- Running Visualizer Test ---")

    # Ensure output directories exist for logging and saving plots
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting visualizer test script.")

    # --- Create Dummy Data ---
    log.warning("Using dummy data for visualizer test.")
    n_proj = config.NUM_PROJECTIONS
    z_dim, y_dim, x_dim = config.RECON_VOLUME_SHAPE
    height, width = config.DETECTOR_SHAPE

    # Dummy sinogram (simple cylinder projection)
    dummy_sinogram = np.zeros((n_proj, height, width), dtype=np.float32)
    center_x, center_y = width // 2, height // 2
    radius_sino = min(width, height) // 4
    for i, angle in enumerate(config.ANGLES_RAD):
        offset = int(radius_sino * np.cos(angle + np.pi/2))
        col_start = max(0, center_x - radius_sino // 4 + offset)
        col_end = min(width, center_x + radius_sino // 4 + offset)
        row_start = max(0, center_y - radius_sino)
        row_end = min(height, center_y + radius_sino)
        if col_start < col_end and row_start < row_end:
             dummy_sinogram[i, row_start:row_end, col_start:col_end] = 0.01

    # Dummy reconstructed volume (simple sphere)
    dummy_volume = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
    center_z, center_y, center_x = z_dim // 2, y_dim // 2, x_dim // 2
    radius_vol = min(z_dim, y_dim, x_dim) // 4
    z, y, x = np.ogrid[:z_dim, :y_dim, :x_dim]
    mask_sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius_vol**2
    dummy_volume[mask_sphere] = 100.0

    log.info(f"Created dummy sinogram ({dummy_sinogram.shape}) and volume ({dummy_volume.shape})")

    # --- Test Plotting Functions ---
    # Plot sinogram slice
    plot_sinogram_slice(dummy_sinogram, slice_index=height // 2, cfg=config, filename="test_sinogram_slice.png")

    # Plot reconstructed slices
    plot_reconstructed_slices(dummy_volume, cfg=config, filename="test_reconstructed_slices.png", suptitle="Test Reconstruction")

    # Save some volume slices
    save_volume_slices(dummy_volume, cfg=config, axis=0, slice_range=range(center_z - 2, center_z + 3), prefix="test_slice_") # Save 5 Z slices
    save_volume_slices(dummy_volume, cfg=config, axis=1, slice_range=center_y, prefix="test_slice_") # Save middle Y slice
    save_volume_slices(dummy_volume, cfg=config, axis=2, slice_range=[center_x-10, center_x, center_x+10], prefix="test_slice_") # Save 3 specific X slices

    log.info("Visualizer test script finished.")
    if config.SAVE_PLOTS:
        print(f"Check output plots in: {config.PLOT_DIR}")
        print(f"Check output slices in: {config.RECON_SLICES_DIR}")
    if config.SHOW_PLOTS:
        print("Plots were displayed interactively.")
    print(f"Check log file for details: {config.LOG_FILE}")
    print("--- End of Visualizer Test ---")

