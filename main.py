# -*- coding: utf-8 -*-
"""
Main script for the CT Reconstruction Pipeline.

This script orchestrates the entire process:
1. Loads configuration.
2. Loads projection and calibration data.
3. Preprocesses the data.
4. Performs reconstruction.
5. Postprocesses the reconstructed volume.
6. Visualizes results (optional).
7. Saves the final volume.
8. Performs evaluation against ground truth (optional).
"""

import os
import sys
import time
import logging
import argparse

# --- Add src directory to Python path ---
# This allows importing modules from the 'src' directory where our files reside.
# Adjust the path if your structure is different.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import Project Modules ---
try:
    import config
    import data_loader
    import preprocess
    import reconstruction
    import postprocess
    import visualize
    import evaluate
    import utils
except ImportError as e:
    print(f"Error: Failed to import one or more project modules: {e}")
    print("Ensure that 'main.py' is in the project root directory and all module files")
    print(f"(config.py, data_loader.py, ...) are inside a subdirectory named 'src'.")
    print(f"Also check that the 'src' directory was added to sys.path: {SRC_DIR}")
    sys.exit(1) # Exit if core modules can't be imported

# --- Global Variables ---
# Configure logging (combines file and console output)
log = logging.getLogger() # Get root logger
log.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO)) # Set level from config

# Prevent duplicate handlers if script is run multiple times in same session
if not log.handlers:
    # File Handler
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(name)-15s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='w') # Overwrite log each run
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)

    # Console Handler
    log_console_handler = logging.StreamHandler(sys.stdout)
    log_console_handler.setFormatter(log_formatter) # Or use a simpler format for console
    log.addHandler(log_console_handler)


def run_pipeline(cfg: object):
    """
    Executes the full CT reconstruction pipeline.

    Args:
        cfg: The configuration object.
    """
    total_start_time = time.time()
    log.info("=================================================")
    log.info("=== Starting CT Reconstruction Pipeline ===")
    log.info("=================================================")

    # --- 1. Ensure Output Directories Exist ---
    log.info("--- Step 1: Checking/Creating Output Directories ---")
    try:
        # Call the function defined in config.py or utils.py
        if hasattr(config, 'ensure_output_dirs_exist'):
             config.ensure_output_dirs_exist()
        else:
             # Basic check if function moved to utils
             dirs_to_create = [cfg.OUTPUT_DIR, cfg.RECON_SLICES_DIR, cfg.RECON_VOLUME_DIR, cfg.PLOT_DIR]
             for d in dirs_to_create: os.makedirs(d, exist_ok=True)
             log.info("Checked/created output directories.")
    except Exception as e:
        log.error(f"Failed to create output directories: {e}")
        return # Stop if we can't write output

    # --- 2. Load Data ---
    log.info("--- Step 2: Loading Data ---")
    start_time = time.time()
    raw_projections = data_loader.load_projections(cfg)
    if raw_projections is None:
        log.error("Pipeline stopped: Failed to load projection data.")
        return

    # Use detected shape for loading calibration images
    detected_shape = (raw_projections.shape[1], raw_projections.shape[2])
    flat_field_data = data_loader.load_flat_fields(cfg, detected_shape)
    dark_field_data = data_loader.load_dark_fields(cfg, detected_shape)
    log.info(f"Data loading finished in {time.time() - start_time:.2f} seconds.")

    # --- 3. Preprocess Data ---
    log.info("--- Step 3: Preprocessing Data ---")
    start_time = time.time()
    preprocessed_sinogram = preprocess.preprocess_data(
        raw_projections, flat_field_data, dark_field_data, cfg
    )
    if preprocessed_sinogram is None:
        log.error("Pipeline stopped: Failed during preprocessing.")
        return
    log.info(f"Preprocessing finished in {time.time() - start_time:.2f} seconds.")

    # Optional: Visualize preprocessed sinogram slice
    if cfg.SAVE_PLOTS or cfg.SHOW_PLOTS:
        log.info("Visualizing sample preprocessed sinogram slice...")
        vis_slice_idx = preprocessed_sinogram.shape[1] // 2
        visualize.plot_sinogram_slice(
            preprocessed_sinogram,
            slice_index=vis_slice_idx,
            cfg=cfg,
            filename="preprocessed_sinogram_slice.png"
        )

    # --- 4. Reconstruct Volume ---
    log.info("--- Step 4: Reconstructing Volume ---")
    start_time = time.time()
    reconstructed_volume = reconstruction.reconstruct_volume(preprocessed_sinogram, cfg)
    if reconstructed_volume is None:
        log.error("Pipeline stopped: Failed during reconstruction.")
        return
    log.info(f"Reconstruction finished in {time.time() - start_time:.2f} seconds.")

    # --- 5. Postprocess Volume ---
    log.info("--- Step 5: Postprocessing Volume ---")
    start_time = time.time()
    postprocessed_volume = postprocess.postprocess_volume(reconstructed_volume, cfg)
    if postprocessed_volume is None:
        log.warning("Postprocessing step failed or returned None. Using un-postprocessed volume for saving/evaluation.")
        final_volume = reconstructed_volume # Fallback to un-processed volume
    else:
        final_volume = postprocessed_volume
        log.info(f"Postprocessing finished in {time.time() - start_time:.2f} seconds.")


    # --- 6. Visualize & Save Results ---
    log.info("--- Step 6: Visualizing and Saving Results ---")
    start_time = time.time()
    # Visualize orthogonal slices
    if cfg.SAVE_PLOTS or cfg.SHOW_PLOTS:
        log.info("Visualizing orthogonal slices of the final volume...")
        visualize.plot_reconstructed_slices(
            final_volume,
            cfg=cfg,
            filename=f"final_volume_slices_{cfg.RECONSTRUCTION_ALGORITHM}.png",
            suptitle=f"Final Volume ({cfg.RECONSTRUCTION_ALGORITHM})"
        )

    # Save individual slices (optional, configure slice_range in config if desired)
    # visualize.save_volume_slices(final_volume, cfg=cfg, axis=0, slice_range=None) # Example: save middle Z slice

    # Save the full 3D volume
    volume_filename = f"reconstructed_volume_{cfg.RECONSTRUCTION_ALGORITHM}.mha" # Use mha by default
    save_success = utils.save_volume(final_volume, volume_filename, cfg=cfg, file_format='mha')
    if save_success:
        log.info(f"Final reconstructed volume saved to {os.path.join(cfg.RECON_VOLUME_DIR, volume_filename)}")
    else:
        log.error("Failed to save the final reconstructed volume.")
    log.info(f"Visualization and saving finished in {time.time() - start_time:.2f} seconds.")


    # --- 7. Evaluate (Optional) ---
    if cfg.PERFORM_EVALUATION:
        log.info("--- Step 7: Evaluating Reconstruction ---")
        start_time = time.time()
        ground_truth_volume = data_loader.load_ground_truth(cfg)
        if ground_truth_volume is not None:
            evaluation_results = evaluate.calculate_metrics(final_volume, ground_truth_volume, cfg)
            if evaluation_results:
                log.info("--- Evaluation Results ---")
                for metric, value in evaluation_results.items():
                    log.info(f"{metric}: {value:.6f}")
                log.info("-------------------------")
            else:
                log.warning("Metric calculation failed or returned no results.")
        else:
            log.warning("Evaluation enabled, but failed to load ground truth volume. Skipping metrics calculation.")
        log.info(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")
    else:
        log.info("--- Step 7: Evaluation Skipped (disabled in config) ---")


    # --- Pipeline End ---
    total_end_time = time.time()
    log.info("=================================================")
    log.info(f"=== CT Reconstruction Pipeline Finished ===")
    log.info(f"=== Total execution time: {total_end_time - total_start_time:.2f} seconds ===")
    log.info("=================================================")
    print(f"\nPipeline finished. Check log file for details: {config.LOG_FILE}")
    if save_success:
        print(f"Final volume saved in: {config.RECON_VOLUME_DIR}")
    if cfg.SAVE_PLOTS:
        print(f"Plots saved in: {config.PLOT_DIR}")


if __name__ == "__main__":
    # --- Argument Parsing (Optional) ---
    # Allows overriding config parameters via command line, e.g., input/output dirs
    parser = argparse.ArgumentParser(description="Run the CT Reconstruction Pipeline.")
    # Add arguments here if needed, e.g.:
    # parser.add_argument('--input_dir', type=str, help='Override projection data directory from config.')
    # parser.add_argument('--output_dir', type=str, help='Override main output directory from config.')
    # parser.add_argument('--algo', type=str, choices=['FBP', 'FDK', 'SIRT', 'CGLS'], help='Override reconstruction algorithm.')

    args = parser.parse_args()

    # --- Update Config (Optional) ---
    # If arguments are provided, update the config object before running
    # Example:
    # if args.input_dir:
    #     config.PROJECTION_DIR = args.input_dir
    #     log.info(f"Overriding projection directory from command line: {config.PROJECTION_DIR}")
    # if args.output_dir:
    #     config.OUTPUT_DIR = args.output_dir
    #     # Need to potentially update sub-directories as well
    #     config.RECON_SLICES_DIR = os.path.join(config.OUTPUT_DIR, 'reconstructed_slices')
    #     # ... update other output paths ...
    #     log.info(f"Overriding output directory from command line: {config.OUTPUT_DIR}")
    # if args.algo:
    #      config.RECONSTRUCTION_ALGORITHM = args.algo
    #      log.info(f"Overriding reconstruction algorithm from command line: {config.RECONSTRUCTION_ALGORITHM}")


    # --- Run the Pipeline ---
    run_pipeline(config)

