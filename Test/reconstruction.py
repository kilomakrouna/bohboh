import os
import numpy as np
import pydicom
import astra
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile # Added for TIFF support


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
        
        # Check CUDA availability and setup
        self.cuda_available = self._check_cuda_availability()
        self.use_cuda = self.cuda_available  # Use CUDA if available
        
        # Check available algorithms
        self.available_algorithms = []
        try:
            self.available_algorithms = astra.astra.algorithm_list()
            print(f"Available ASTRA algorithms: {self.available_algorithms}")
        except:
            print("Could not retrieve algorithm list from ASTRA")
        
        print(f"CT Reconstructor initialized")
        print(f"CUDA available: {self.cuda_available}")
        print(f"Using CUDA: {self.use_cuda}")
        print(f"Using {'cone-beam' if self.use_cone_beam else 'parallel-beam'} geometry")
        print(f"Source-object distance: {self.distance_source_object} mm")
        print(f"Object-detector distance: {self.distance_object_detector} mm")
        print(f"Number of angles: {len(self.angles_rad)}")
    
    def _check_cuda_availability(self):
        """
        Check if CUDA is available for ASTRA reconstruction
        
        Returns:
        --------
        bool : True if CUDA is available, False otherwise
        """
        try:
            # Try to create a simple CUDA algorithm to test availability
            proj_geom = astra.create_proj_geom('parallel', 1.0, 32, np.array([0]))
            vol_geom = astra.create_vol_geom(32, 32)
            
            # Create dummy data
            sinogram = np.ones((1, 32), dtype=np.float32)
            sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
            rec_id = astra.data2d.create('-vol', vol_geom)
            
            # Try to create a CUDA algorithm
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ProjectionDataId'] = sinogram_id
            cfg['ReconstructionDataId'] = rec_id
            
            alg_id = astra.algorithm.create(cfg)
            
            # If we get here, CUDA is available
            print("CUDA support detected in ASTRA")
            
            # Clean up
            astra.algorithm.delete(alg_id)
            astra.data2d.delete([sinogram_id, rec_id])
            
            return True
            
        except Exception as e:
            print(f"CUDA not available: {e}")
            print("Will use CPU algorithms instead")
            return False
    
    def set_cuda_usage(self, use_cuda):
        """
        Enable or disable CUDA usage
        
        Parameters:
        -----------
        use_cuda : bool
            Whether to use CUDA acceleration
        """
        if use_cuda and not self.cuda_available:
            print("Warning: CUDA requested but not available. Will use CPU instead.")
            self.use_cuda = False
        else:
            self.use_cuda = use_cuda
            print(f"CUDA usage set to: {self.use_cuda}")
    
    def get_system_info(self):
        """
        Get system information for debugging
        
        Returns:
        --------
        dict : System information including CUDA status
        """
        info = {
            'cuda_available': self.cuda_available,
            'cuda_enabled': self.use_cuda,
            'available_algorithms': self.available_algorithms,
            'astra_version': None,
            'numpy_version': np.__version__
        }
        
        try:
            info['astra_version'] = astra.__version__
        except:
            info['astra_version'] = "Unknown"
        
        # Try to get CUDA device info if available
        if self.cuda_available:
            try:
                # This is a simple way to check CUDA device count
                import subprocess
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split('\n'))
                    info['gpu_count'] = gpu_count
                else:
                    info['gpu_count'] = "Unknown"
            except:
                info['gpu_count'] = "Unknown"
        else:
            info['gpu_count'] = 0
        
        return info
    
    def print_system_info(self):
        """
        Print detailed system information for debugging
        """
        info = self.get_system_info()
        
        print("\n" + "="*50)
        print("SYSTEM INFORMATION")
        print("="*50)
        print(f"ASTRA Version: {info['astra_version']}")
        print(f"NumPy Version: {info['numpy_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
        print(f"CUDA Enabled: {info['cuda_enabled']}")
        print(f"GPU Count: {info['gpu_count']}")
        print(f"Available Algorithms: {len(info['available_algorithms'])}")
        if info['available_algorithms']:
            print("Algorithm List:")
            for i, alg in enumerate(info['available_algorithms'][:10]):  # Show first 10
                print(f"  - {alg}")
            if len(info['available_algorithms']) > 10:
                print(f"  ... and {len(info['available_algorithms']) - 10} more")
        print("="*50 + "\n")
    
    def load_projections(self, dcm_dir):
        """
        Load projection images (TIFF or DICOM) from a directory
        
        Parameters:
        -----------
        dcm_dir : str
            Path to directory containing image files
            
        Returns:
        --------
        projections : ndarray
            3D array of projection images
        """
        
        # Try loading TIFF files first
        tiff_files = sorted([os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.lower().endswith(('.tiff', '.tif'))])
        
        if tiff_files:
            print(f"Found {len(tiff_files)} TIFF files. Loading TIFF projection images...")
            try:
                projections = np.stack([tifffile.imread(f).astype(np.float32) for f in tqdm(tiff_files, desc="Loading TIFFs")], axis=0)
            except Exception as e:
                print(f"Error loading TIFF files: {e}")
                raise
        else:
            # Fallback to DICOM if no TIFFs found
            dcm_files = sorted([os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.lower().endswith('.dcm')])
            if not dcm_files:
                raise FileNotFoundError(f"No TIFF or DICOM files found in directory: {dcm_dir}")
            
            print(f"Found {len(dcm_files)} DICOM files. Loading DICOM projection images...")
            try:
                projections = np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in tqdm(dcm_files, desc="Loading DICOMs")], axis=0)
            except Exception as e:
                print(f"Error loading DICOM files: {e}")
                raise
        
        if projections.ndim == 2: # If it's a single 2D image, treat as a single slice projection series (e.g. stack of 2D tiffs)
            # This might need adjustment based on how multi-slice TIFFs are stored.
            # Assuming each file is a projection, and they form a stack.
            # If each TIFF file is a 3D stack itself, this logic needs to change.
            # For now, assuming each file is one projection angle.
             print(f"Warning: Loaded projections are 2D. Assuming a stack of 2D projection images.")
             projections = projections[np.newaxis, :, :] # Add a dummy slice dimension if needed for consistency
                                                        # Or handle it as [num_projections, height, width] directly

        if projections.ndim != 3:
            raise ValueError(f"Loaded projections must be a 3D array (num_projections, height, width). Got shape: {projections.shape}")

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
        
        # Define reconstruction methods based on CUDA availability
        if self.use_cuda:
            print("Using CUDA-accelerated reconstruction methods...")
            reconstruction_methods = [
                ('SIRT_CUDA', {'ProjectionOrder': 'random', 'MinConstraint': 0}),
                ('FBP_CUDA', {}),
                ('BP_CUDA', {}),
                ('CGLS_CUDA', {}),
                ('SART_CUDA', {'MinConstraint': 0})
            ]
        else:
            print("Using CPU reconstruction methods...")
            reconstruction_methods = [
                ('SIRT', {'ProjectionOrder': 'random', 'MinConstraint': 0}),
                ('FBP', {}),
                ('BP', {}),
                ('CGLS', {}),
                ('SART', {'MinConstraint': 0})
            ]
        
        # Try ASTRA methods first
        try:
            print("Testing ASTRA reconstruction methods...")
            success = False
            working_method = None
            working_options = None
            
            # Find a working reconstruction method
            for method, options in reconstruction_methods:
                print(f"Testing reconstruction with {method} method...")
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
                    
                    # Configure the algorithm
                    cfg = astra.astra_dict(method)
                    cfg['ProjectionDataId'] = sinogram_id
                    cfg['ReconstructionDataId'] = rec_id
                    
                    # Add options if any
                    if options:
                        cfg['option'] = options
                    
                    # Try to create and run the algorithm
                    alg_id = astra.algorithm.create(cfg)
                    
                    # Run a few iterations for testing
                    if method in ['SIRT_CUDA', 'SIRT', 'SART_CUDA', 'SART', 'CGLS_CUDA', 'CGLS']:
                        astra.algorithm.run(alg_id, 5)  # Just 5 iterations for testing
                    else:
                        astra.algorithm.run(alg_id)
                    
                    # If we get here, the algorithm worked
                    working_method = method
                    working_options = options
                    success = True
                    
                    print(f"✓ {method} method works successfully!")
                    
                    # Clean up
                    astra.algorithm.delete(alg_id)
                    astra.data2d.delete([sinogram_id, rec_id])
                    break
                    
                except Exception as e:
                    print(f"✗ Method {method} failed: {e}")
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
                print(f"\nUsing {working_method} for full reconstruction...")
                reconstructed_slices = []
                
                # Determine number of iterations for iterative methods
                num_iterations = 50  # Default for iterative methods
                if working_method in ['SIRT_CUDA', 'SIRT']:
                    num_iterations = 100  # SIRT benefits from more iterations
                elif working_method in ['CGLS_CUDA', 'CGLS']:
                    num_iterations = 30   # CGLS converges faster
                elif working_method in ['SART_CUDA', 'SART']:
                    num_iterations = 20   # SART can be unstable with too many iterations
                
                # Progress bar for reconstruction
                progress_bar = tqdm(range(num_rows), desc="Reconstructing slices")
                
                for i in progress_bar:
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
                    cfg = astra.astra_dict(working_method)
                    cfg['ProjectionDataId'] = sinogram_id
                    cfg['ReconstructionDataId'] = rec_id
                    
                    # Add options if any
                    if working_options:
                        cfg['option'] = working_options
                    
                    # Create and run the algorithm
                    alg_id = astra.algorithm.create(cfg)
                    
                    # Run the algorithm
                    if working_method in ['SIRT_CUDA', 'SIRT', 'SART_CUDA', 'SART', 'CGLS_CUDA', 'CGLS']:
                        astra.algorithm.run(alg_id, num_iterations)
                    else:
                        astra.algorithm.run(alg_id)
                    
                    # Get the reconstructed slice
                    rec = astra.data2d.get(rec_id)
                    reconstructed_slices.append(rec)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Method': working_method,
                        'CUDA': 'Yes' if 'CUDA' in working_method else 'No'
                    })
                    
                    # Clean up memory
                    astra.algorithm.delete(alg_id)
                    astra.data2d.delete([sinogram_id, rec_id])
                    
                progress_bar.close()
                
            else:
                raise ValueError("No ASTRA reconstruction methods worked")
        
        except Exception as e:
            print(f"\nAll ASTRA reconstruction methods failed: {e}")
            print("Falling back to manual backprojection (this will be slower but more reliable)")
            
            # Use manual backprojection instead
            reconstructed_slices = []
            angles = np.linspace(0, np.pi, num_projections, endpoint=False)
            
            for i in tqdm(range(num_rows), desc="Manual backprojection"):
                # Extract the sinogram for slice i
                sinogram = projections[:, i, :]
                
                # Apply manual backprojection
                rec = self.manual_backprojection(sinogram, angles)
                reconstructed_slices.append(rec)
        
        # Convert list of 2D slices to 3D volume
        self.reconstructed_volume = np.stack(reconstructed_slices, axis=0)
        
        print(f"\nReconstruction completed successfully!")
        print(f"Volume shape: {self.reconstructed_volume.shape}")
        print(f"Volume data type: {self.reconstructed_volume.dtype}")
        print(f"Value range: [{self.reconstructed_volume.min():.3f}, {self.reconstructed_volume.max():.3f}]")
        
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
            
    def visualize_3d(self, threshold=0.5, output_dir='3d_visualization', show=True):
        """
        Create a 3D visualization of the reconstructed volume using matplotlib
        
        Parameters:
        -----------
        threshold : float
            Threshold value for isosurface extraction (0-1)
        output_dir : str
            Directory to save visualizations
        show : bool
            Whether to display the plot (useful in Jupyter notebooks)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure for 3D visualization
        """
        if self.reconstructed_volume is None:
            raise ValueError("No reconstructed volume to visualize. Run reconstruct() first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize volume
        volume = self.reconstructed_volume
        if volume.max() > 1.0:
            volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
        else:
            volume_normalized = volume
        
        # Create the 3D visualization using matplotlib
        fig = self._create_3d_plot(volume_normalized, threshold, output_dir)
        
        if show:
            plt.show()
            
        return fig
    
    def _create_3d_plot(self, volume, threshold=0.5, output_dir=None):
        """
        Helper method to create 3D plot using matplotlib
        
        Parameters:
        -----------
        volume : ndarray
            3D volume data (normalized)
        threshold : float
            Threshold for binary visualization
        output_dir : str, optional
            Directory to save output
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The 3D figure
        """
        # Threshold the volume to create a binary mask
        binary_volume = volume > threshold
        
        # Get coordinates of non-zero voxels
        z_indices, y_indices, x_indices = np.where(binary_volume)
        
        # Subsample if too many points (for performance)
        max_points = 50000
        if len(z_indices) > max_points:
            step = len(z_indices) // max_points
            z_indices = z_indices[::step]
            y_indices = y_indices[::step]
            x_indices = x_indices[::step]
        
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the corresponding intensity values
        intensities = volume[z_indices, y_indices, x_indices]
        
        # Create scatter plot with color based on intensity
        scatter = ax.scatter(
            x_indices, y_indices, z_indices,
            c=intensities,
            cmap='viridis',
            s=1,
            alpha=0.1,
            marker='.'
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Intensity')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Volume Visualization')
        
        # Save the figure if output directory is provided
        if output_dir:
            output_path = os.path.join(output_dir, '3d_volume.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D visualization to {output_path}")
        
        return fig
    
    def visualize_orthogonal_slices(self, output_dir='visualization_slices', show=True):
        """
        Visualize orthogonal slices through the volume
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        show : bool
            Whether to display the plot
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the orthogonal slices
        """
        if self.reconstructed_volume is None:
            raise ValueError("No reconstructed volume to visualize. Run reconstruct() first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize volume
        volume = self.reconstructed_volume
        if volume.max() > 1.0:
            volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
        else:
            volume_normalized = volume
        
        # Get dimensions
        z_dim, y_dim, x_dim = volume_normalized.shape
        
        # Create a figure with 3 subplots for orthogonal views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Axial view (XY plane)
        axial_slice = volume_normalized[z_dim//2, :, :]
        axes[0].imshow(axial_slice, cmap='gray')
        axes[0].set_title(f'Axial Slice (Z={z_dim//2})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Coronal view (XZ plane)
        coronal_slice = volume_normalized[:, y_dim//2, :]
        axes[1].imshow(coronal_slice, cmap='gray')
        axes[1].set_title(f'Coronal Slice (Y={y_dim//2})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        # Sagittal view (YZ plane)
        sagittal_slice = volume_normalized[:, :, x_dim//2]
        axes[2].imshow(sagittal_slice, cmap='gray')
        axes[2].set_title(f'Sagittal Slice (X={x_dim//2})')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, 'orthogonal_slices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved orthogonal slices to {output_path}")
        
        if show:
            plt.show()
            
        return fig
    
    def histogram_analysis(self, output_dir='histogram_analysis', show=True):
        """
        Perform histogram analysis on the reconstructed volume
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        show : bool
            Whether to display the plot
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the histogram
        """
        if self.reconstructed_volume is None:
            raise ValueError("No reconstructed volume to analyze. Run reconstruct() first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a histogram of the volume intensities
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate histogram
        hist, bins = np.histogram(self.reconstructed_volume.flatten(), bins=100)
        
        # Plot the histogram
        ax.bar(bins[:-1], hist, width=(bins[1]-bins[0]), alpha=0.7, color='blue')
        
        # Add labels and title
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Reconstructed Volume')
        
        # Add grid for better readability
        ax.grid(alpha=0.3)
        
        # Add statistics as text
        volume_min = np.min(self.reconstructed_volume)
        volume_max = np.max(self.reconstructed_volume)
        volume_mean = np.mean(self.reconstructed_volume)
        volume_median = np.median(self.reconstructed_volume)
        
        stats_text = (
            f"Min: {volume_min:.2f}\n"
            f"Max: {volume_max:.2f}\n"
            f"Mean: {volume_mean:.2f}\n"
            f"Median: {volume_median:.2f}"
        )
        
        # Add the text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram analysis to {output_path}")
        
        if show:
            plt.show()
            
        return fig
