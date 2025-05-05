"""
Main module for tomographic reconstruction.

This module provides a command-line interface for running the tomographic
reconstruction pipeline on a set of 2D projection images.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Import custom modules
from preprocessing import preprocess_projections
from reconstruction import filtered_backprojection, art_reconstruction, sirt_reconstruction, fdk_reconstruction
from visualization import show_volume_slices, show_volume_3slice, render_surface_plotly
from utils import (
    load_tiff_stack, 
    extract_angles_from_filenames, 
    create_projection_geometry,
    save_numpy_as_vtk,
    save_volume_as_tiff_stack
)

def main():
    """Main function for tomographic reconstruction."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Tomographic Reconstruction')
    
    # Input/output arguments
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing projection images')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory for saving reconstructed volume')
    parser.add_argument('--file-pattern', type=str, default='*.tif*',
                      help='Pattern for selecting projection files')
    
    # Geometry arguments
    parser.add_argument('--source-origin-dist', type=float, default=500.0,
                      help='Distance from source to rotation center (mm)')
    parser.add_argument('--origin-detector-dist', type=float, default=500.0,
                      help='Distance from rotation center to detector (mm)')
    parser.add_argument('--angle-pattern', type=str, default='_(\d+)deg',
                      help='Pattern for extracting angles from filenames')
    
    # Reconstruction arguments
    parser.add_argument('--algorithm', type=str, default='fbp',
                      choices=['fbp', 'art', 'sirt', 'fdk'],
                      help='Reconstruction algorithm')
    parser.add_argument('--volume-size', type=int, nargs=3, default=None,
                      help='Size of reconstructed volume (x, y, z)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Number of iterations for iterative methods')
    
    # Preprocessing arguments
    parser.add_argument('--no-normalize', action='store_true',
                      help='Skip normalization')
    parser.add_argument('--no-denoise', action='store_true',
                      help='Skip denoising')
    parser.add_argument('--no-ring-removal', action='store_true',
                      help='Skip ring artifact removal')
    parser.add_argument('--no-rotation-correction', action='store_true',
                      help='Skip center of rotation correction')
    
    # Output arguments
    parser.add_argument('--output-format', type=str, default='vtk',
                      choices=['vtk', 'tiff', 'both'],
                      help='Output format for reconstructed volume')
    parser.add_argument('--visualize', action='store_true',
                      help='Show visualization after reconstruction')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load projection images
    print(f"Loading projections from {args.input_dir}...")
    projections, filenames = load_tiff_stack(args.input_dir, args.file_pattern)
    
    # Extract angles
    print("Extracting projection angles...")
    angles = extract_angles_from_filenames(filenames, args.angle_pattern)
    
    # Preprocess projections
    print("Preprocessing projections...")
    processed_projections = preprocess_projections(
        projections, 
        angles=angles,
        normalize=not args.no_normalize,
        denoise=not args.no_denoise,
        remove_rings=not args.no_ring_removal,
        correct_rotation=not args.no_rotation_correction
    )
    
    # Set volume size if not specified
    if args.volume_size is None:
        size = max(projections.shape[1], projections.shape[2])
        args.volume_size = (size, size, projections.shape[1])
    
    # Create projection geometry
    geometry = create_projection_geometry(
        angles, 
        (projections.shape[1], projections.shape[2]),
        args.source_origin_dist, 
        args.origin_detector_dist
    )
    
    # Perform reconstruction
    print(f"Performing {args.algorithm.upper()} reconstruction...")
    t_start = time()
    
    if args.algorithm == 'fbp':
        volume = filtered_backprojection(
            processed_projections, 
            angles, 
            args.volume_size
        )
    elif args.algorithm == 'art':
        volume = art_reconstruction(
            processed_projections, 
            angles, 
            args.volume_size, 
            iterations=args.iterations
        )
    elif args.algorithm == 'sirt':
        volume = sirt_reconstruction(
            processed_projections, 
            angles, 
            args.volume_size, 
            iterations=args.iterations
        )
    elif args.algorithm == 'fdk':
        volume = fdk_reconstruction(
            processed_projections, 
            geometry, 
            args.volume_size
        )
    
    t_end = time()
    print(f"Reconstruction completed in {t_end - t_start:.2f} seconds.")
    
    # Save reconstructed volume
    base_filename = os.path.join(args.output_dir, f"{args.algorithm}_reconstruction")
    
    if args.output_format in ['vtk', 'both']:
        print(f"Saving volume as VTK file...")
        save_numpy_as_vtk(volume, f"{base_filename}.vti")
    
    if args.output_format in ['tiff', 'both']:
        print(f"Saving volume as TIFF stack...")
        save_volume_as_tiff_stack(volume, args.output_dir, f"{args.algorithm}_slice")
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing results...")
        show_volume_3slice(volume)
        
        # Create and save isosurface visualization
        try:
            fig = render_surface_plotly(volume)
            fig.write_html(os.path.join(args.output_dir, f"{args.algorithm}_isosurface.html"))
            fig.show()
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
    
    print("Done.")

if __name__ == "__main__":
    main()
