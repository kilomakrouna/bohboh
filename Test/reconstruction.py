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
        self.use_cone_beam = False  # Changed to False to start with parallel beam (more reliable)
        self.reconstructed_volume = None
        
        # Check available algorithms
        self.available_algorithms = []
        try:
            self.available_algorithms = astra.astra.algorithm_list()
            print(f"Available ASTRA algorithms: {self.available_algorithms}")
        except:
            print("Could not retrieve algorithm list from ASTRA")
        
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
    
    def manual_backprojection(self, sinogram, angles):
        """
        Perform a simple manual backprojection without relying on ASTRA's algorithms
        
        Parameters:
        -----------
        sinogram : ndarray
            2D sinogram array
        angles : ndarray
            Array of angles in radians
            
        Returns:
        --------
        reconstruction : ndarray
            Reconstructed 2D image
        """
        num_angles, num_detector_pixels = sinogram.shape
        
        # Create a square reconstruction grid
        size = num_detector_pixels
        reconstruction = np.zeros((size, size), dtype=np.float32)
        
        # Center coordinates
        center_x = size // 2
        center_y = size // 2
        
        # Create meshgrid of coordinates
        x = np.arange(size) - center_x
        y = np.arange(size) - center_y
        X, Y = np.meshgrid(x, y)
        
        # For each angle, backproject
        for i, theta in enumerate(angles):
            # Convert to radians
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Calculate projection coordinates
            t = X * cos_theta + Y * sin_theta
            
            # Convert to detector pixel indices
            t = t + num_detector_pixels // 2
            
            # Clip to valid indices
            t = np.clip(t.astype(np.int32), 0, num_detector_pixels - 1)
            
            # Backproject
            reconstruction += sinogram[i, t]
        
        # Normalize
        reconstruction /= num_angles
        
        return reconstruction
    
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
            self.angles_rad = np.linspace(0, np.pi, num_projections, endpoint=False)
        
        # Try ASTRA methods first
        try:
            print("Trying ASTRA reconstruction methods...")
            # Define the method to use (try several if needed)
            reconstruction_methods = ['SIRT', 'CGLS', 'BP', 'FBP', 'SART']
            success = False
            
            # Find a working reconstruction method
            working_method = None
            for method in reconstruction_methods:
                print(f"Trying reconstruction with {method} method...")
                try:
                    # Test with one slice
                    sinogram = projections[:, 0, :]
                    angles = np.linspace(0, np.pi, num_projections, endpoint=False)
                    
                    # Create geometry
                    proj_geom = astra.create_proj_geom('parallel', 1.0, num_cols, angles)
                    vol_geom = astra.create_vol_geom(num_cols, num_cols)
                    
                    # Create data objects
                    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
                    rec_id = astra.data2d.create('-vol', vol_geom)
                    
                    # Try to create the algorithm
                    if method == 'BP':
                        # Backprojection
                        cfg = astra.astra_dict('BP')
                    elif method == 'FBP':
                        # Filtered backprojection
                        cfg = astra.astra_dict('FBP')
                    elif method == 'SIRT':
                        # SIRT iterative method
                        cfg = astra.astra_dict('SIRT')
                        cfg['option'] = {'ProjectionOrder': 'random'}
                        cfg['option']['MinConstraint'] = 0
                        cfg['option']['MaxConstraint'] = 255
                    elif method == 'SART':
                        # SART iterative method
                        cfg = astra.astra_dict('SART')
                        cfg['option'] = {}
                        cfg['option']['MinConstraint'] = 0
                    elif method == 'CGLS':
                        # CGLS iterative method
                        cfg = astra.astra_dict('CGLS')
                        cfg['option'] = {}
                    
                    cfg['ProjectionDataId'] = sinogram_id
                    cfg['ReconstructionDataId'] = rec_id
                    
                    # Try to create the algorithm
                    alg_id = astra.algorithm.create(cfg)
                    
                    # If we get here, the algorithm was created successfully
                    astra.algorithm.run(alg_id, 20)  # 20 iterations for iterative methods
                    working_method = method
                    success = True
                    
                    # Clean up
                    astra.algorithm.delete(alg_id)
                    astra.data2d.delete([sinogram_id, rec_id])
                    break
                    
                except Exception as e:
                    print(f"Method {method} failed: {e}")
                    # Try to clean up if objects were created
                    try:
                        if 'alg_id' in locals():
                            astra.algorithm.delete(alg_id)
                        if 'sinogram_id' in locals() and 'rec_id' in locals():
                            astra.data2d.delete([sinogram_id, rec_id])
                    except:
                        pass
            
            # If we found a working method, use it for all slices
            if success:
                print(f"Using ASTRA {working_method} for reconstruction")
                reconstructed_slices = []
                
                # Now reconstruct all slices with the working method
                for i in tqdm(range(num_rows)):
                    # Extract the sinogram for slice i
                    sinogram = projections[:, i, :]
                    
                    # Define parallel beam geometry
                    angles = np.linspace(0, np.pi, num_projections, endpoint=False)
                    proj_geom = astra.create_proj_geom('parallel', 1.0, num_cols, angles)
                    vol_geom = astra.create_vol_geom(num_cols, num_cols)
                    
                    # Create data objects
                    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
                    rec_id = astra.data2d.create('-vol', vol_geom)
                    
                    # Configure the algorithm
                    if working_method == 'BP':
                        cfg = astra.astra_dict('BP')
                    elif working_method == 'FBP':
                        cfg = astra.astra_dict('FBP')
                    elif working_method == 'SIRT':
                        cfg = astra.astra_dict('SIRT')
                        cfg['option'] = {'ProjectionOrder': 'random'}
                    elif working_method == 'SART':
                        cfg = astra.astra_dict('SART')
                        cfg['option'] = {}
                    elif working_method == 'CGLS':
                        cfg = astra.astra_dict('CGLS')
                        cfg['option'] = {}
                    
                    cfg['ProjectionDataId'] = sinogram_id
                    cfg['ReconstructionDataId'] = rec_id
                    
                    # Create and run the algorithm
                    alg_id = astra.algorithm.create(cfg)
                    
                    # Run the algorithm (with iterations for iterative methods)
                    if working_method in ['SIRT', 'SART', 'CGLS']:
                        astra.algorithm.run(alg_id, 20)
                    else:
                        astra.algorithm.run(alg_id)
                    
                    # Get the reconstructed slice
                    rec = astra.data2d.get(rec_id)
                    reconstructed_slices.append(rec)
                    
                    # Clean up memory
                    astra.algorithm.delete(alg_id)
                    astra.data2d.delete([sinogram_id, rec_id])
            else:
                raise ValueError("No ASTRA reconstruction methods worked")
        
        except Exception as e:
            print(f"All ASTRA reconstruction methods failed: {e}")
            print("Falling back to manual backprojection (this will be slower but more reliable)")
            
            # Use manual backprojection instead
            reconstructed_slices = []
            angles = np.linspace(0, np.pi, num_projections, endpoint=False)
            
            for i in tqdm(range(num_rows)):
                # Extract the sinogram for slice i
                sinogram = projections[:, i, :]
                
                # Apply manual backprojection
                rec = self.manual_backprojection(sinogram, angles)
                reconstructed_slices.append(rec)
        
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

