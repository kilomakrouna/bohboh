"""
ASTRA 3D Reconstruction Module

This module provides a class for 3D reconstruction from 2D TIFF images using ASTRA toolkit.
It supports both SIRT and FBP reconstruction algorithms and provides visualization capabilities.
"""

import os
import glob
import json
import configparser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import astra
from PIL import Image
import time
from pathlib import Path


class AstraReconstructor:
    """
    A class for 3D reconstruction from 2D TIFF images using ASTRA toolkit.
    
    This class provides functionality to:
    1. Load and parse configuration files
    2. Load TIFF image stacks
    3. Perform 3D reconstruction using SIRT and FBP algorithms
    4. Visualize the 3D reconstruction results
    
    Attributes:
        config (dict): Configuration parameters for reconstruction
        projections (numpy.ndarray): Projection data loaded from TIFF files
        volume (numpy.ndarray): Reconstructed 3D volume
        proj_geom (dict): Projection geometry for ASTRA
        vol_geom (dict): Volume geometry for ASTRA
    """
    
    def __init__(self, config_json=None, config_ini=None):
        """
        Initialize the AstraReconstructor with optional configuration files.
        
        Args:
            config_json (str, optional): Path to JSON configuration file
            config_ini (str, optional): Path to INI configuration file
        """
        self.config = {
            'detector_pixels': [3072, 3072],
            'detector_size': [427.0, 427.0],
            'distance_source_object': 305.950581856,
            'distance_object_detector': 763.908859988,
            'object_size': [122.11262615207769, 122.11262615207769, 122.11262615207769],
            'volume_size': [256, 256, 256],  # Default reconstruction volume size
            'angles': None,
            'algorithm': 'SIRT',  # Default algorithm
            'iterations': 100,     # Default iterations for iterative methods
        }
        
        self.projections = None
        self.volume = None
        self.proj_geom = None
        self.vol_geom = None
        
        # Load configuration if provided
        if config_json:
            self.load_json_config(config_json)
        if config_ini:
            self.load_ini_config(config_ini)
            
        # Initialize ASTRA geometries
        self._init_geometries()
    
    def load_json_config(self, config_path):
        """
        Load configuration from JSON file.
        
        Args:
            config_path (str): Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract geometry information
            geometry = config_data.get('geometry', {})
            
            # Update configuration
            if 'detectorPixel' in geometry:
                self.config['detector_pixels'] = geometry['detectorPixel']
            if 'detectorSize' in geometry:
                self.config['detector_size'] = geometry['detectorSize']
            if 'distanceSourceObject' in geometry:
                self.config['distance_source_object'] = geometry['distanceSourceObject']
            if 'distanceObjectDetector' in geometry:
                self.config['distance_object_detector'] = geometry['distanceObjectDetector']
            
            # Extract object bounding box
            if 'objectBoundingBox' in geometry:
                bbox = geometry['objectBoundingBox']
                if 'sizeXYZ' in bbox:
                    self.config['object_size'] = bbox['sizeXYZ']
            
            # Extract projection angles
            if 'projectionAngles' in geometry:
                angles = [item['angle'] for item in geometry['projectionAngles']]
                self.config['angles'] = np.array(angles) * np.pi / 180.0  # Convert to radians
            
            print(f"Loaded JSON configuration from {config_path}")
        except Exception as e:
            print(f"Error loading JSON configuration: {e}")
    
    def load_ini_config(self, config_path):
        """
        Load configuration from INI file.
        
        Args:
            config_path (str): Path to INI configuration file
        """
        try:
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Extract relevant parameters
            if 'driverAndRay' in parser:
                section = parser['driverAndRay']
                
                if 'SOD' in section:
                    self.config['distance_source_object'] = float(section['SOD'])
                if 'SDD' in section:
                    total_distance = float(section['SDD'])
                    self.config['distance_object_detector'] = total_distance - self.config['distance_source_object']
                if 'VolSize' in section:
                    vol_size = float(section['VolSize'])
                    # Use this to adjust volume size if needed
                
                # Extract angle information if not already set
                if self.config['angles'] is None and 'Startangle_num' in section and 'Endangle_num' in section:
                    start_angle = float(section['Startangle_num'])
                    end_angle = float(section['Endangle_num'])
                    num_angles = int(section.get('numericUpDown2', 360))
                    self.config['angles'] = np.linspace(start_angle, end_angle, num_angles) * np.pi / 180.0
            
            print(f"Loaded INI configuration from {config_path}")
        except Exception as e:
            print(f"Error loading INI configuration: {e}")
    
    def _init_geometries(self):
        """Initialize ASTRA geometries based on configuration."""
        # Check if angles are available
        if self.config['angles'] is None:
            print("Warning: No projection angles defined. Using default 180 angles.")
            self.config['angles'] = np.linspace(0, np.pi, 180)
        
        # Create volume geometry
        self.vol_geom = astra.create_vol_geom(
            self.config['volume_size'][0],
            self.config['volume_size'][1],
            self.config['volume_size'][2]
        )
        
        # Create cone beam projection geometry
        self.proj_geom = astra.create_proj_geom(
            'cone', 
            self.config['detector_size'][0] / self.config['detector_pixels'][0],
            self.config['detector_size'][1] / self.config['detector_pixels'][1],
            self.config['detector_pixels'][1],
            self.config['detector_pixels'][0],
            self.config['angles'],
            self.config['distance_source_object'] / self.config['object_size'][0] * self.config['volume_size'][0],
            self.config['distance_object_detector'] / self.config['object_size'][0] * self.config['volume_size'][0]
        )
    
    def load_projections(self, tiff_dir, pattern='*.tiff', normalize=True):
        """
        Load projection images from TIFF files.
        
        Args:
            tiff_dir (str): Directory containing TIFF files
            pattern (str, optional): Glob pattern for TIFF files
            normalize (bool, optional): Whether to normalize projection data
            
        Returns:
            numpy.ndarray: Loaded projection data
        """
        # Find all matching TIFF files
        tiff_files = sorted(glob.glob(os.path.join(tiff_dir, pattern)))
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {tiff_dir} with pattern {pattern}")
        
        print(f"Loading {len(tiff_files)} projection images...")
        
        # Load the first image to get dimensions
        first_img = np.array(Image.open(tiff_files[0]))
        height, width = first_img.shape
        
        # Allocate memory for all projections
        self.projections = np.zeros((len(tiff_files), height, width), dtype=np.float32)
        
        # Load all projections
        for i, file_path in enumerate(tiff_files):
            self.projections[i] = np.array(Image.open(file_path))
        
        # Normalize if requested
        if normalize:
            self.projections = (self.projections - np.min(self.projections)) / (np.max(self.projections) - np.min(self.projections))
        
        print(f"Loaded projections with shape {self.projections.shape}")
        return self.projections
    
    def reconstruct(self, algorithm='SIRT', iterations=100, gpu_index=0):
        """
        Perform 3D reconstruction using specified algorithm.
        
        Args:
            algorithm (str, optional): Reconstruction algorithm ('SIRT' or 'FBP')
            iterations (int, optional): Number of iterations for iterative methods
            gpu_index (int, optional): GPU index to use for reconstruction
            
        Returns:
            numpy.ndarray: Reconstructed 3D volume
        """
        if self.projections is None:
            raise ValueError("No projection data loaded. Call load_projections() first.")
        
        # Set GPU index
        astra.astra.set_gpu_index(gpu_index)
        
        # Create ASTRA data objects
        proj_id = astra.data3d.create('-proj3d', self.proj_geom, self.projections)
        
        # Create a data object for the reconstruction
        vol_id = astra.data3d.create('-vol', self.vol_geom)
        
        # Set up the algorithm
        algorithm = algorithm.upper()
        if algorithm == 'SIRT':
            alg_cfg = astra.astra_dict('SIRT3D_CUDA')
            alg_cfg['ProjectionDataId'] = proj_id
            alg_cfg['ReconstructionDataId'] = vol_id
            alg_cfg['option'] = {
                'MinConstraint': 0.0,  # Non-negativity constraint
            }
            
            # Create the algorithm object
            alg_id = astra.algorithm.create(alg_cfg)
            
            # Run the algorithm
            print(f"Running SIRT reconstruction with {iterations} iterations...")
            start_time = time.time()
            astra.algorithm.run(alg_id, iterations)
            elapsed_time = time.time() - start_time
            print(f"Reconstruction completed in {elapsed_time:.2f} seconds")
            
        elif algorithm == 'FBP':
            alg_cfg = astra.astra_dict('FDK_CUDA')
            alg_cfg['ProjectionDataId'] = proj_id
            alg_cfg['ReconstructionDataId'] = vol_id
            
            # Create the algorithm object
            alg_id = astra.algorithm.create(alg_cfg)
            
            # Run the algorithm
            print("Running FBP (FDK) reconstruction...")
            start_time = time.time()
            astra.algorithm.run(alg_id, 1)
            elapsed_time = time.time() - start_time
            print(f"Reconstruction completed in {elapsed_time:.2f} seconds")
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'SIRT' or 'FBP'.")
        
        # Get the result
        self.volume = astra.data3d.get(vol_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        
        return self.volume
    
    def save_volume(self, output_path):
        """
        Save the reconstructed volume to a NumPy file.
        
        Args:
            output_path (str): Path to save the volume
        """
        if self.volume is None:
            raise ValueError("No volume data available. Run reconstruct() first.")
        
        np.save(output_path, self.volume)
        print(f"Volume saved to {output_path}")
    
    def load_volume(self, input_path):
        """
        Load a previously saved volume.
        
        Args:
            input_path (str): Path to the saved volume
            
        Returns:
            numpy.ndarray: Loaded volume data
        """
        self.volume = np.load(input_path)
        print(f"Volume loaded from {input_path} with shape {self.volume.shape}")
        return self.volume
    
    def visualize_slice(self, axis=0, slice_index=None, figsize=(10, 8), cmap='gray'):
        """
        Visualize a slice of the reconstructed volume.
        
        Args:
            axis (int, optional): Axis along which to take the slice (0, 1, or 2)
            slice_index (int, optional): Index of the slice to visualize
            figsize (tuple, optional): Figure size
            cmap (str, optional): Colormap for visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.volume is None:
            raise ValueError("No volume data available. Run reconstruct() first.")
        
        # Determine slice index if not provided
        if slice_index is None:
            slice_index = self.volume.shape[axis] // 2
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract and display the slice
        if axis == 0:
            slice_data = self.volume[slice_index, :, :]
            title = f"X-Slice at index {slice_index}"
        elif axis == 1:
            slice_data = self.volume[:, slice_index, :]
            title = f"Y-Slice at index {slice_index}"
        elif axis == 2:
            slice_data = self.volume[:, :, slice_index]
            title = f"Z-Slice at index {slice_index}"
        else:
            raise ValueError("Axis must be 0, 1, or 2")
        
        # Display the slice
        im = ax.imshow(slice_data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        
        return fig
    
    def visualize_volume_3d(self, threshold=0.1, figsize=(12, 10)):
        """
        Create a 3D visualization of the reconstructed volume.
        
        Args:
            threshold (float, optional): Threshold value for isosurface
            figsize (tuple, optional): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.volume is None:
            raise ValueError("No volume data available. Run reconstruct() first.")
        
        from skimage import measure
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract isosurface
        verts, faces, _, _ = measure.marching_cubes(self.volume, threshold)
        
        # Plot the isosurface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap=cm.coolwarm, lw=0, alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Volume Visualization (threshold={threshold})')
        
        return fig
    
    def visualize_mip(self, axis=2, figsize=(10, 8), cmap='gray'):
        """
        Create a Maximum Intensity Projection (MIP) of the volume.
        
        Args:
            axis (int, optional): Axis along which to project (0, 1, or 2)
            figsize (tuple, optional): Figure size
            cmap (str, optional): Colormap for visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.volume is None:
            raise ValueError("No volume data available. Run reconstruct() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create MIP
        mip = np.max(self.volume, axis=axis)
        
        # Display the MIP
        im = ax.imshow(mip, cmap=cmap)
        
        # Set title based on projection axis
        if axis == 0:
            title = "Maximum Intensity Projection (X-axis)"
        elif axis == 1:
            title = "Maximum Intensity Projection (Y-axis)"
        else:
            title = "Maximum Intensity Projection (Z-axis)"
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        
        return fig
    
    def create_orthogonal_views(self, slice_indices=None, figsize=(15, 5), cmap='gray'):
        """
        Create orthogonal views (XY, XZ, YZ) of the reconstructed volume.
        
        Args:
            slice_indices (list, optional): Indices for each axis [x, y, z]
            figsize (tuple, optional): Figure size
            cmap (str, optional): Colormap for visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.volume is None:
            raise ValueError("No volume data available. Run reconstruct() first.")
        
        # Determine slice indices if not provided
        if slice_indices is None:
            slice_indices = [
                self.volume.shape[0] // 2,
                self.volume.shape[1] // 2,
                self.volume.shape[2] // 2
            ]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # XY plane (Z-slice)
        z_slice = self.volume[:, :, slice_indices[2]]
        im0 = axes[0].imshow(z_slice, cmap=cmap)
        axes[0].set_title(f'XY Plane (Z={slice_indices[2]})')
        fig.colorbar(im0, ax=axes[0])
        
        # XZ plane (Y-slice)
        y_slice = self.volume[:, slice_indices[1], :]
        im1 = axes[1].imshow(y_slice, cmap=cmap)
        axes[1].set_title(f'XZ Plane (Y={slice_indices[1]})')
        fig.colorbar(im1, ax=axes[1])
        
        # YZ plane (X-slice)
        x_slice = self.volume[slice_indices[0], :, :]
        im2 = axes[2].imshow(x_slice, cmap=cmap)
        axes[2].set_title(f'YZ Plane (X={slice_indices[0]})')
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    def set_volume_size(self, size):
        """
        Set the size of the reconstruction volume.
        
        Args:
            size (list or tuple): Size of the volume [nx, ny, nz]
        """
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise ValueError("Volume size must be a list or tuple of length 3")
        
        self.config['volume_size'] = size
        self._init_geometries()
        print(f"Volume size set to {size}")
    
    def set_algorithm_params(self, algorithm='SIRT', iterations=100):
        """
        Set parameters for the reconstruction algorithm.
        
        Args:
            algorithm (str): Reconstruction algorithm ('SIRT' or 'FBP')
            iterations (int): Number of iterations for iterative methods
        """
        self.config['algorithm'] = algorithm
        self.config['iterations'] = iterations
        print(f"Algorithm set to {algorithm} with {iterations} iterations")
