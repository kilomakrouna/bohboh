import os
import numpy as np
import pydicom
import astra
import json
from tqdm import tqdm


class CTReconstructor:
    def __init__(self, config_path='Test/CT4000.json'):
        """
        Initialize the CT Reconstructor with a configuration file
        
        Parameters:
        -----------
        config_path : str
            Path to the CT configuration JSON file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Extract geometry parameters
        self.detector_pixels = self.config['geometry']['detectorPixel']
        self.detector_size = self.config['geometry']['detectorSize']
        self.distance_source_object = self.config['geometry']['distanceSourceObject']
        self.distance_object_detector = self.config['geometry']['distanceObjectDetector']
        self.bounding_box = self.config['geometry']['objectBoundingBox']
        
        # Extract projection angles
        self.angles_rad = np.array([angle_data['angle'] for angle_data in self.config['geometry']['projectionAngles']])
        
        # Set default parameters
        self.use_cone_beam = True  # Use cone-beam geometry since we have source distances
        self.reconstructed_volume = None
        
        print(f"CT Reconstructor initialized")
        print(f"Using {'cone-beam' if self.use_cone_beam else 'parallel-beam'} geometry")
        print(f"Source-object distance: {self.distance_source_object} mm")
        print(f"Object-detector distance: {self.distance_object_detector} mm")
        print(f"Number of angles: {len(self.angles_rad)}")
    
    def load_projections(self, dcm_dir):
        """
        Load DICOM projection images from a directory
        
        Parameters:
        -----------
        dcm_dir : str
            Path to directory containing DICOM files
            
        Returns:
        --------
        projections : ndarray
            3D array of projection images
        """
        dcm_files = sorted([os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        
        print("Loading DICOM projection images...")
        projections = np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in tqdm(dcm_files)], axis=0)
        
        num_projections, num_rows, num_cols = projections.shape
        print(f"Loaded {num_projections} projections, each with shape {num_rows} x {num_cols}")
        
        return projections
    
    def reconstruct(self, projections=None, dcm_dir=None, output_path=None):
        """
        Perform the CT reconstruction
        
        Parameters:
        -----------
        projections : ndarray, optional
            3D array of projection images. If not provided, will load from dcm_dir
        dcm_dir : str, optional
            Path to directory containing DICOM files
        output_path : str, optional
            Path to save the reconstructed volume
            
        Returns:
        --------
        reconstructed_volume : ndarray
            3D reconstructed volume
        """
        if projections is None:
            if dcm_dir is None:
                raise ValueError("Either projections or dcm_dir must be provided")
            projections = self.load_projections(dcm_dir)
        
        num_projections, num_rows, num_cols = projections.shape
        
        # Check if the number of projections matches the angles in the config
        config_angles = len(self.angles_rad)
        if num_projections != config_angles:
            print(f"Warning: Number of projections ({num_projections}) doesn't match config angles ({config_angles})")
            print("Adapting angles to match the number of projections...")
            # Adjust angles to match the number of projections
            if self.use_cone_beam:
                # For cone beam, we typically use angles spanning a full rotation (0 to 2π)
                self.angles_rad = np.linspace(0, 2*np.pi, num_projections, endpoint=False)
            else:
                # For parallel beam, we typically use angles spanning half a rotation (0 to π)
                self.angles_rad = np.linspace(0, np.pi, num_projections, endpoint=False)
        
        # Reconstruct each slice
        reconstructed_slices = []
        
        print("Reconstructing slices...")
        try:
            for i in tqdm(range(num_rows)):
                # Extract the sinogram for slice i
                sinogram = projections[:, i, :]
                
                if self.use_cone_beam:
                    # Check dimensions and try to adjust if there's a mismatch
                    try:
                        # Define cone-beam geometry
                        proj_geom = astra.create_proj_geom(
                            'cone', 
                            1.0, 1.0,  # detector pixel size in mm
                            num_rows, num_cols,  # detector rows, cols
                            self.angles_rad,  # angles in radians
                            self.distance_source_object,  # source to origin distance
                            self.distance_object_detector  # origin to detector distance
                        )
                        
                        # If cone beam is causing issues, try to use fan beam for 2D reconstruction instead
                        sinogram_id = astra.data3d.create('-sino', proj_geom, sinogram)
                    except Exception as e:
                        print(f"Error with cone-beam geometry: {e}")
                        print("Falling back to parallel beam geometry for 2D reconstruction...")
                        self.use_cone_beam = False
                        # Define parallel beam geometry for this slice
                        proj_geom = astra.create_proj_geom('parallel', 1.0, num_cols, self.angles_rad)
                        vol_geom = astra.create_vol_geom(num_cols, num_cols)
                        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
                        rec_id = astra.data2d.create('-vol', vol_geom)
                        cfg = astra.astra_dict('FBP')
                        cfg['ProjectionDataId'] = sinogram_id
                        cfg['ReconstructionDataId'] = rec_id
                        alg_id = astra.algorithm.create(cfg)
                        astra.algorithm.run(alg_id)
                        rec = astra.data2d.get(rec_id)
                        reconstructed_slices.append(rec)
                        astra.algorithm.delete(alg_id)
                        astra.data2d.delete([sinogram_id, rec_id])
                        continue
                    
                    # Define volume geometry based on bounding box size
                    vol_size = int(self.bounding_box['sizeXYZ'][0])  # Use the size from bounding box
                    vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)
                    
                    # Create 3D volume data
                    rec_id = astra.data3d.create('-vol', vol_geom)
                    
                    # Configure the FDK algorithm (cone-beam)
                    cfg = astra.astra_dict('FDK')
                    
                else:
                    # For parallel beam (2D)
                    proj_geom = astra.create_proj_geom('parallel', 1.0, num_cols, self.angles_rad)
                    vol_geom = astra.create_vol_geom(num_cols, num_cols)
                    
                    # Create 2D data objects
                    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
                    rec_id = astra.data2d.create('-vol', vol_geom)
                    
                    # Configure the FBP algorithm (parallel beam)
                    cfg = astra.astra_dict('FBP')
            
                # Set the configuration
                cfg['ProjectionDataId'] = sinogram_id
                cfg['ReconstructionDataId'] = rec_id
                
                # Create and run the algorithm
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                # Get the reconstructed slice
                if self.use_cone_beam:
                    rec = astra.data3d.get(rec_id)
                else:
                    rec = astra.data2d.get(rec_id)
                
                reconstructed_slices.append(rec)
                
                # Clean up memory
                astra.algorithm.delete(alg_id)
                if self.use_cone_beam:
                    astra.data3d.delete([sinogram_id, rec_id])
                else:
                    astra.data2d.delete([sinogram_id, rec_id])
                
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            print("Trying alternative approach with 2D parallel beam reconstruction...")
            
            # Fall back to simple 2D parallel beam reconstruction
            reconstructed_slices = []
            
            # Use 2D parallel beam reconstruction for each slice
            for i in tqdm(range(num_rows)):
                # Extract the sinogram for slice i
                sinogram = projections[:, i, :]
                
                # Define parallel beam geometry
                angles = np.linspace(0, np.pi, num_projections, endpoint=False)
                proj_geom = astra.create_proj_geom('parallel', 1.0, num_cols, angles)
                vol_geom = astra.create_vol_geom(num_cols, num_cols)
                
                # Create 2D data objects
                sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
                rec_id = astra.data2d.create('-vol', vol_geom)
                
                # Configure the FBP algorithm
                cfg = astra.astra_dict('FBP')
                cfg['ProjectionDataId'] = sinogram_id
                cfg['ReconstructionDataId'] = rec_id
                
                # Run the algorithm
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                # Get the reconstructed slice
                rec = astra.data2d.get(rec_id)
                reconstructed_slices.append(rec)
                
                # Clean up memory
                astra.algorithm.delete(alg_id)
                astra.data2d.delete([sinogram_id, rec_id])
        
        # Convert list of 2D slices to 3D volume
        self.reconstructed_volume = np.stack(reconstructed_slices, axis=0)
        
        print(f"Reconstruction done. Volume shape: {self.reconstructed_volume.shape}")
        
        # Save the reconstructed volume if output_path is provided
        if output_path:
            self.save(output_path)
            
        return self.reconstructed_volume
    
    def save(self, output_path='reconstructed_volume.npy'):
        """
        Save the reconstructed volume to a file
        
        Parameters:
        -----------
        output_path : str
            Path to save the reconstructed volume
        """
        if self.reconstructed_volume is None:
            raise ValueError("No reconstructed volume to save. Run reconstruct() first.")
            
        np.save(output_path, self.reconstructed_volume)
        print(f"Saved reconstructed volume to {output_path}")
    
    def visualize(self, slice_idx=None, output_dir='reconstruction_results', show=True):
        """
        Visualize slices from the reconstructed volume
        
        Parameters:
        -----------
        slice_idx : int or list of int, optional
            Index or indices of slices to visualize. If None, uses the middle slice.
        output_dir : str, optional
            Directory to save visualizations
        show : bool, optional
            Whether to display the plot (useful in Jupyter notebooks)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object for further customization
        """
        if self.reconstructed_volume is None:
            raise ValueError("No reconstructed volume to visualize. Run reconstruct() first.")
            
        try:
            import matplotlib.pyplot as plt
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # If slice_idx is None, use the middle slice
            if slice_idx is None:
                slice_idx = self.reconstructed_volume.shape[0] // 2
                
            # Convert to list if not already
            if not isinstance(slice_idx, (list, tuple)):
                slice_idx = [slice_idx]
                
            figs = []
            for idx in slice_idx:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(self.reconstructed_volume[idx], cmap='gray')
                plt.title(f'Reconstructed Slice {idx}')
                plt.colorbar(label='Intensity')
                
                # Save the figure
                plt.savefig(os.path.join(output_dir, f'slice_{idx}.png'))
                print(f"Saved visualization of slice {idx} to {output_dir}/slice_{idx}.png")
                
                figs.append(fig)
                
            if show:
                plt.show()
                
            return figs[0] if len(figs) == 1 else figs
                
        except ImportError:
            print("Matplotlib not available for slice visualization")
            return None

