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
    def __init__(self, config_path=None):
        """
        Initialize the CT Reconstructor with optional configuration file
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the CT configuration JSON file. If None, parameters will be detected automatically.
        """
        # Initialize default values
        self.config = None
        self.detector_pixels = None
        self.detector_size = None
        self.distance_source_object = None
        self.distance_object_detector = None
        self.bounding_box = None
        self.angles_rad = None
        self.auto_detected = False
        
        # Load configuration if provided
        if config_path is not None:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    
                # Extract geometry parameters from config
                self.detector_pixels = self.config['geometry']['detectorPixel']
                self.detector_size = self.config['geometry']['detectorSize']
                self.distance_source_object = self.config['geometry']['distanceSourceObject']
                self.distance_object_detector = self.config['geometry']['distanceObjectDetector']
                self.bounding_box = self.config['geometry']['objectBoundingBox']
                
                # Extract projection angles
                self.angles_rad = np.array([angle_data['angle'] for angle_data in self.config['geometry']['projectionAngles']])
                
                print(f"Loaded configuration from: {config_path}")
                
            except FileNotFoundError:
                print(f"Warning: Configuration file '{config_path}' not found. Will use automatic detection.")
                config_path = None
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Warning: Error reading configuration file '{config_path}': {e}")
                print("Will use automatic detection.")
                config_path = None
        
        # If no config provided or loading failed, we'll detect parameters automatically later
        if config_path is None:
            print("No configuration file provided. Parameters will be detected automatically from projection data.")
            self.auto_detected = True
        
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
        
        if not self.auto_detected:
            print(f"Source-object distance: {self.distance_source_object} mm")
            print(f"Object-detector distance: {self.distance_object_detector} mm")
            print(f"Number of angles: {len(self.angles_rad)}")
        else:
            print("Geometry parameters will be set automatically when loading projections.")
    
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
        tiff_files = sorted([os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.lower().endswith(('.tiff', '.tif', '.bmp', '.BMP'))])
        
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
        
        # Auto-detect parameters if no config was provided
        if self.auto_detected:
            self._auto_detect_parameters(num_projections, num_rows, num_cols)
        
        return projections
    
    def _auto_detect_parameters(self, num_projections, num_rows, num_cols):
        """
        Automatically detect CT parameters from projection data
        
        Parameters:
        -----------
        num_projections : int
            Number of projection images
        num_rows : int
            Height of each projection image
        num_cols : int
            Width of each projection image
        """
        print("\n" + "="*50)
        print("AUTO-DETECTING CT PARAMETERS")
        print("="*50)
        
        # Set detector parameters based on projection dimensions
        self.detector_pixels = [num_cols, num_rows]  # [width, height]
        
        # Assume 1mm pixel size as default (can be adjusted)
        pixel_size = 1.0  # mm
        self.detector_size = [num_cols * pixel_size, num_rows * pixel_size]
        
        # Set reasonable default distances for parallel beam geometry
        # These don't affect parallel beam reconstruction much but are needed for ASTRA
        self.distance_source_object = 1000.0  # mm
        self.distance_object_detector = 1000.0  # mm
        
        # Create bounding box that encompasses the reconstruction volume
        # Make it slightly smaller than detector to avoid edge artifacts
        margin = 0.1  # 10% margin
        box_size = min(num_cols, num_rows) * (1 - margin) * pixel_size
        self.bounding_box = [-box_size/2, box_size/2, -box_size/2, box_size/2, -box_size/2, box_size/2]
        
        # Generate evenly spaced angles from 0 to 2*pi (360 degrees)
        # This provides full angular coverage to avoid cylindrical artifacts
        self.angles_rad = np.linspace(0, 2*np.pi, num_projections, endpoint=False)
        
        print(f"Detector pixels: {self.detector_pixels}")
        print(f"Detector size: {self.detector_size} mm")
        print(f"Pixel size: {pixel_size} mm")
        print(f"Source-object distance: {self.distance_source_object} mm")
        print(f"Object-detector distance: {self.distance_object_detector} mm")
        print(f"Bounding box: {self.bounding_box}")
        print(f"Number of angles: {len(self.angles_rad)}")
        print(f"Angle range: 0 to {2*np.pi:.3f} radians (0 to 360 degrees)")
        print("="*50 + "\n")
        
        print("Note: These parameters have been automatically detected.")
        print("360-degree angular coverage is used by default to avoid cylindrical artifacts.")
        print("For better results, consider providing a configuration file with exact geometry parameters.")
        print("You can adjust these parameters manually if needed.\n")
    
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
            # Keep the same angular range but adjust to match number of projections
            start_angle = self.angles_rad[0]
            end_angle = self.angles_rad[-1] + (self.angles_rad[-1] - self.angles_rad[0]) / (len(self.angles_rad) - 1)
            self.angles_rad = np.linspace(start_angle, end_angle, num_projections, endpoint=False)
            print(f"Updated to {len(self.angles_rad)} angles from {np.degrees(start_angle):.1f}° to {np.degrees(end_angle):.1f}°")
        
        # Check if a preferred method is set
        preferred_method = getattr(self, 'preferred_method', 'auto')
        
        # Define reconstruction methods based on CUDA availability and preferred method
        if preferred_method != 'auto':
            if preferred_method == 'manual_backprojection':
                print(f"Using preferred method: {preferred_method}")
                # Skip ASTRA and go directly to manual backprojection
                reconstructed_slices = []
                angles = self.angles_rad  # Use the actual configured angles
                
                for i in tqdm(range(num_rows), desc="Manual backprojection"):
                    # Extract the sinogram for slice i
                    sinogram = projections[:, i, :]
                    
                    # Apply manual backprojection
                    rec = self.manual_backprojection(sinogram, angles)
                    reconstructed_slices.append(rec)
                
                # Convert list of 2D slices to 3D volume
                self.reconstructed_volume = np.stack(reconstructed_slices, axis=0)
                
                print(f"\nManual backprojection completed successfully!")
                print(f"Volume shape: {self.reconstructed_volume.shape}")
                print(f"Volume data type: {self.reconstructed_volume.dtype}")
                print(f"Value range: [{self.reconstructed_volume.min():.3f}, {self.reconstructed_volume.max():.3f}]")
                
                # Save the reconstructed volume if output_path is provided
                if output_path:
                    self.save(output_path)
                    
                return self.reconstructed_volume
            
            else:
                # Try the preferred method first
                if self.use_cuda:
                    if preferred_method.endswith('_CUDA'):
                        reconstruction_methods = [(preferred_method, self._get_method_options(preferred_method))]
                    else:
                        reconstruction_methods = [(preferred_method + '_CUDA', self._get_method_options(preferred_method + '_CUDA'))]
                else:
                    if preferred_method.endswith('_CUDA'):
                        method_name = preferred_method.replace('_CUDA', '')
                        reconstruction_methods = [(method_name, self._get_method_options(method_name))]
                    else:
                        reconstruction_methods = [(preferred_method, self._get_method_options(preferred_method))]
                
                print(f"Trying preferred reconstruction method: {reconstruction_methods[0][0]}")
        else:
            # Use automatic method selection
            if self.use_cuda:
                print("Using CUDA-accelerated reconstruction methods...")
                reconstruction_methods = [
                    ('FBP_CUDA', {}),  # Moved FBP to first position (better for most cases)
                    ('BP_CUDA', {}),
                    ('SIRT_CUDA', {'ProjectionOrder': 'random', 'MinConstraint': 0}),
                    ('CGLS_CUDA', {}),
                    ('SART_CUDA', {'MinConstraint': 0})
                ]
            else:
                print("Using CPU reconstruction methods...")
                reconstruction_methods = [
                    ('FBP', {}),  # Moved FBP to first position (better for most cases)
                    ('BP', {}),
                    ('SIRT', {'ProjectionOrder': 'random', 'MinConstraint': 0}),
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
                    angles = self.angles_rad  # Use the actual configured angles
                    
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
                    angles = self.angles_rad  # Use the actual configured angles
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
            angles = self.angles_rad  # Use the actual configured angles
            
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
    
    def set_pixel_size(self, pixel_size_mm):
        """
        Set the pixel size and update detector size accordingly
        
        Parameters:
        -----------
        pixel_size_mm : float
            Pixel size in millimeters
        """
        if self.detector_pixels is None:
            raise ValueError("Detector pixels not set. Load projections first.")
        
        self.detector_size = [self.detector_pixels[0] * pixel_size_mm, 
                             self.detector_pixels[1] * pixel_size_mm]
        
        # Update bounding box based on new pixel size
        margin = 0.1  # 10% margin
        box_size = min(self.detector_pixels) * (1 - margin) * pixel_size_mm
        self.bounding_box = [-box_size/2, box_size/2, -box_size/2, box_size/2, -box_size/2, box_size/2]
        
        print(f"Updated pixel size to {pixel_size_mm} mm")
        print(f"New detector size: {self.detector_size} mm")
        print(f"Updated bounding box: {self.bounding_box}")
    
    def set_angular_range(self, start_angle=0, end_angle=np.pi, num_projections=None):
        """
        Set custom angular range for projections
        
        Parameters:
        -----------
        start_angle : float
            Starting angle in radians (default: 0)
        end_angle : float
            Ending angle in radians (default: pi for 180 degrees)
        num_projections : int, optional
            Number of projections. If None, uses current number of angles
        """
        if num_projections is None:
            if self.angles_rad is not None:
                num_projections = len(self.angles_rad)
            else:
                raise ValueError("Number of projections not set. Load projections first or specify num_projections.")
        
        self.angles_rad = np.linspace(start_angle, end_angle, num_projections, endpoint=False)
        
        print(f"Updated angular range: {start_angle:.3f} to {end_angle:.3f} radians")
        print(f"({np.degrees(start_angle):.1f} to {np.degrees(end_angle):.1f} degrees)")
        print(f"Number of angles: {len(self.angles_rad)}")
    
    def set_geometry_distances(self, source_object_distance, object_detector_distance):
        """
        Set the source-object and object-detector distances
        
        Parameters:
        -----------
        source_object_distance : float
            Distance from source to object in millimeters
        object_detector_distance : float
            Distance from object to detector in millimeters
        """
        self.distance_source_object = source_object_distance
        self.distance_object_detector = object_detector_distance
        
        print(f"Updated geometry distances:")
        print(f"Source-object distance: {self.distance_source_object} mm")
        print(f"Object-detector distance: {self.distance_object_detector} mm")
    
    def print_current_parameters(self):
        """
        Print the current CT parameters
        """
        print("\n" + "="*50)
        print("CURRENT CT PARAMETERS")
        print("="*50)
        
        if self.detector_pixels is not None:
            print(f"Detector pixels: {self.detector_pixels}")
        else:
            print("Detector pixels: Not set")
            
        if self.detector_size is not None:
            print(f"Detector size: {self.detector_size} mm")
            pixel_size = self.detector_size[0] / self.detector_pixels[0] if self.detector_pixels else None
            if pixel_size:
                print(f"Pixel size: {pixel_size:.3f} mm")
        else:
            print("Detector size: Not set")
            
        if self.distance_source_object is not None:
            print(f"Source-object distance: {self.distance_source_object} mm")
        else:
            print("Source-object distance: Not set")
            
        if self.distance_object_detector is not None:
            print(f"Object-detector distance: {self.distance_object_detector} mm")
        else:
            print("Object-detector distance: Not set")
            
        if self.bounding_box is not None:
            print(f"Bounding box: {self.bounding_box}")
        else:
            print("Bounding box: Not set")
            
        if self.angles_rad is not None:
            print(f"Number of angles: {len(self.angles_rad)}")
            print(f"Angle range: {self.angles_rad[0]:.3f} to {self.angles_rad[-1]:.3f} radians")
            print(f"({np.degrees(self.angles_rad[0]):.1f} to {np.degrees(self.angles_rad[-1]):.1f} degrees)")
        else:
            print("Angles: Not set")
            
        print(f"Configuration source: {'Auto-detected' if self.auto_detected else 'Config file'}")
        print("="*50 + "\n")
    
    def set_reconstruction_method(self, method_name):
        """
        Set a specific reconstruction method to use
        
        Parameters:
        -----------
        method_name : str
            Name of the reconstruction method to use
            Options: 'FBP', 'FBP_CUDA', 'SIRT', 'SIRT_CUDA', 'BP', 'BP_CUDA', 
                    'SART', 'SART_CUDA', 'CGLS', 'CGLS_CUDA', 'manual_backprojection', 'auto'
        
        Returns:
        --------
        bool : True if method was set successfully, False otherwise
        """
        valid_methods = [
            'FBP', 'FBP_CUDA', 'SIRT', 'SIRT_CUDA', 'BP', 'BP_CUDA',
            'SART', 'SART_CUDA', 'CGLS', 'CGLS_CUDA', 'manual_backprojection', 'auto'
        ]
        
        if method_name not in valid_methods:
            print(f"Error: Invalid method '{method_name}'. Valid options are: {valid_methods}")
            return False
        
        # Check if CUDA method is requested but not available
        if '_CUDA' in method_name and not self.cuda_available:
            print(f"Warning: {method_name} requested but CUDA not available. Use {method_name.replace('_CUDA', '')} instead.")
            return False
        
        self.preferred_method = method_name
        print(f"Reconstruction method set to: {method_name}")
        return True
    
    def get_current_method(self):
        """
        Get the currently set reconstruction method
        
        Returns:
        --------
        str : Current reconstruction method
        """
        return getattr(self, 'preferred_method', 'auto')
    
    def list_available_methods(self):
        """
        List all available reconstruction methods
        """
        print("Available reconstruction methods:")
        print("="*40)
        
        methods = [
            ('FBP', 'Filtered Backprojection (CPU)', True),
            ('FBP_CUDA', 'Filtered Backprojection (GPU)', self.cuda_available),
            ('BP', 'Simple Backprojection (CPU)', True),
            ('BP_CUDA', 'Simple Backprojection (GPU)', self.cuda_available),
            ('SIRT', 'Simultaneous Iterative Reconstruction (CPU)', True),
            ('SIRT_CUDA', 'Simultaneous Iterative Reconstruction (GPU)', self.cuda_available),
            ('SART', 'Simultaneous Algebraic Reconstruction (CPU)', True),
            ('SART_CUDA', 'Simultaneous Algebraic Reconstruction (GPU)', self.cuda_available),
            ('CGLS', 'Conjugate Gradient Least Squares (CPU)', True),
            ('CGLS_CUDA', 'Conjugate Gradient Least Squares (GPU)', self.cuda_available),
            ('manual_backprojection', 'Manual Backprojection (Fallback)', True),
            ('auto', 'Automatic selection (Default)', True)
        ]
        
        for method, description, available in methods:
            status = "✓" if available else "✗"
            print(f"{status} {method:<20} - {description}")
        
        print("="*40)
        print("Recommended methods:")
        print("- FBP/FBP_CUDA: Fast, good for well-sampled data")
        print("- SIRT/SIRT_CUDA: Better for limited-angle or noisy data")
        print("- BP/BP_CUDA: Simple but may have artifacts")
    
    def check_angular_coverage(self):
        """
        Check the angular coverage and suggest improvements if needed
        
        Returns:
        --------
        dict : Analysis of angular coverage
        """
        if self.angles_rad is None:
            return {"error": "No angles loaded yet"}
        
        analysis = {}
        
        # Calculate angular range
        angle_range = self.angles_rad[-1] - self.angles_rad[0]
        angle_step = np.mean(np.diff(self.angles_rad)) if len(self.angles_rad) > 1 else 0
        
        analysis['num_angles'] = len(self.angles_rad)
        analysis['start_angle_deg'] = np.degrees(self.angles_rad[0])
        analysis['end_angle_deg'] = np.degrees(self.angles_rad[-1])
        analysis['total_range_deg'] = np.degrees(angle_range)
        analysis['average_step_deg'] = np.degrees(angle_step)
        
        # Check coverage
        if angle_range < 0.9 * np.pi:  # Less than ~162 degrees
            analysis['coverage_warning'] = "Limited angular coverage detected. This may cause cylindrical artifacts."
            analysis['recommendation'] = "Consider using 360-degree scan or SIRT/SART algorithms for better reconstruction."
        elif angle_range < 1.1 * np.pi:  # About 180 degrees
            analysis['coverage_info'] = "Standard 180-degree coverage detected."
            analysis['recommendation'] = "Good for most objects. FBP should work well."
        else:  # More than 180 degrees
            analysis['coverage_info'] = "Extended angular coverage detected."
            analysis['recommendation'] = "Excellent coverage. All algorithms should work well."
        
        # Check angular sampling
        if len(self.angles_rad) < 180:
            analysis['sampling_warning'] = "Sparse angular sampling detected. This may cause streak artifacts."
            analysis['sampling_recommendation'] = "Consider using iterative algorithms (SIRT/SART) to reduce artifacts."
        
        return analysis
    
    def print_angular_analysis(self):
        """
        Print analysis of angular coverage
        """
        analysis = self.check_angular_coverage()
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print("\n" + "="*50)
        print("ANGULAR COVERAGE ANALYSIS")
        print("="*50)
        print(f"Number of angles: {analysis['num_angles']}")
        print(f"Start angle: {analysis['start_angle_deg']:.1f}°")
        print(f"End angle: {analysis['end_angle_deg']:.1f}°")
        print(f"Total coverage: {analysis['total_range_deg']:.1f}°")
        print(f"Average step: {analysis['average_step_deg']:.2f}°")
        
        if 'coverage_warning' in analysis:
            print(f"\n⚠️  WARNING: {analysis['coverage_warning']}")
        elif 'coverage_info' in analysis:
            print(f"\nℹ️  INFO: {analysis['coverage_info']}")
        
        if 'sampling_warning' in analysis:
            print(f"\n⚠️  WARNING: {analysis['sampling_warning']}")
        
        if 'recommendation' in analysis:
            print(f"\n💡 RECOMMENDATION: {analysis['recommendation']}")
        
        if 'sampling_recommendation' in analysis:
            print(f"\n💡 SAMPLING: {analysis['sampling_recommendation']}")
        
        print("="*50 + "\n")
    
    def use_360_degree_scan(self, num_projections=None):
        """
        Set angular range to 360 degrees to avoid cylindrical artifacts
        
        Parameters:
        -----------
        num_projections : int, optional
            Number of projections. If None, uses current number
        """
        if num_projections is None:
            if self.angles_rad is not None:
                num_projections = len(self.angles_rad)
            else:
                raise ValueError("Number of projections not set. Load projections first or specify num_projections.")
        
        # Set 360-degree coverage
        self.angles_rad = np.linspace(0, 2*np.pi, num_projections, endpoint=False)
        
        print(f"Updated to 360-degree scan with {num_projections} projections")
        print("This should reduce cylindrical artifacts significantly.")
        
        # Also update cone beam if the total coverage suggests it
        if num_projections >= 360:
            print("With this many projections, consider enabling cone-beam geometry for better results.")
            print("Use: reconstructor.use_cone_beam = True")
    
    def _get_method_options(self, method_name):
        """
        Get default options for a reconstruction method
        
        Parameters:
        -----------
        method_name : str
            Name of the reconstruction method
            
        Returns:
        --------
        dict : Options for the method
        """
        if 'SIRT' in method_name:
            return {'ProjectionOrder': 'random', 'MinConstraint': 0}
        elif 'SART' in method_name:
            return {'MinConstraint': 0}
        else:
            return {}
