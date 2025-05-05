"""
ASTRA Toolbox-based reconstruction algorithms for tomographic reconstruction.
This provides high-performance GPU-accelerated implementations using the ASTRA Toolbox.
"""
import numpy as np

try:
    import astra
    ASTRA_AVAILABLE = True
    print("ASTRA Toolbox is available. GPU-accelerated reconstruction enabled.")
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA Toolbox not available. Install with 'conda install -c astra-toolbox astra-toolbox'")

def filtered_backprojection_astra(projections, angles, volume_shape=None, filter_name='ram-lak'):
    """
    ASTRA-based Filtered backprojection algorithm.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple, optional): Shape of the output volume (height, width, slices).
        filter_name (str): Filter to use (ram-lak, shepp-logan, cosine, hamming, hann, etc.).
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Check if projections are too large for memory - raise early warning
    if np.prod(projections.shape) > 1e9:  # More than ~1GB of projection data
        print(f"Warning: Large projection data size {projections.shape}. Consider downsampling.")
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    if volume_shape is None:
        size = projections.shape[2]  # Use width as size
        volume_shape = (size, size, projections.shape[1])
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    print(f"ASTRA: Processing {projections.shape[1]} slices with projection data shape {projections.shape}")
    
    # Determine orientation of the projections (which dimension is angles)
    angles_dim = None
    if projections.shape[0] == len(angles):
        angles_dim = 0
    else:
        raise ValueError(f"Could not find angles dimension in projections with shape {projections.shape}")
    
    # For each slice along the rotation axis
    for slice_idx in range(projections.shape[1]):
        # Extract sinogram for current slice
        sinogram = projections[:, slice_idx, :]
        
        # Create ASTRA volume geometry
        vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
        
        # Create projection vectors manually
        # This avoids issues with the projection geometry
        det_count = sinogram.shape[1]  # Number of detector pixels
        vectors = np.zeros((len(angles_rad), 6))
        
        for i, angle in enumerate(angles_rad):
            # Ray direction
            vectors[i, 0] = np.cos(angle)
            vectors[i, 1] = np.sin(angle)
            # Center of detector
            vectors[i, 2] = -np.sin(angle) * det_count/2
            vectors[i, 3] = np.cos(angle) * det_count/2
            # Vector from detector pixel 0 to 1
            vectors[i, 4] = -np.sin(angle)
            vectors[i, 5] = np.cos(angle)
        
        # Create parallel beam projection geometry using vectors
        proj_geom = astra.create_proj_geom('parallel_vec', det_count, vectors)
        
        # Debug info
        print(f"Slice {slice_idx}: Created projection geometry with {det_count} detectors and {len(angles_rad)} angles")
        
        # Create ASTRA sinogram object - no transpose needed with vector geometry
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        
        # Create reconstruction object
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # Create configuration for FBP
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['FilterType'] = filter_name
        
        # Create and run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get the result
        volume[:, :, slice_idx] = astra.data2d.get(rec_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
    
    return volume

def sirt_reconstruction_astra(projections, angles, volume_shape, iterations=100):
    """
    ASTRA-based SIRT reconstruction algorithm.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume (height, width, slices).
        iterations (int): Number of iterations.
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # For each slice along the rotation axis
    for slice_idx in range(projections.shape[1]):
        # Extract sinogram for current slice
        sinogram = projections[:, slice_idx, :]
        
        # Transpose sinogram to match ASTRA's expected format (detectors × angles)
        sinogram = sinogram.transpose()
        
        # Create ASTRA volume geometry
        vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
        
        # Create ASTRA projection geometry (assuming parallel beam)
        proj_geom = astra.create_proj_geom('parallel', 1.0, sinogram.shape[0], angles_rad)
        
        # Create sinogram object
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        
        # Create reconstruction object
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # Create configuration for SIRT
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {'MinConstraint': 0}  # Enforce non-negativity
        
        # Create and run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        
        # Get the result
        volume[:, :, slice_idx] = astra.data2d.get(rec_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
    
    return volume

def fdk_reconstruction_astra(projections, geometry, volume_shape):
    """
    ASTRA-based FDK reconstruction algorithm for cone-beam geometry.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        geometry (dict): Projection geometry with source_origin_dist and origin_detector_dist.
        volume_shape (tuple): Shape of the output volume (height, width, depth).
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Extract geometry parameters
    angles_deg = geometry['angles']
    source_origin_dist = float(geometry['source_origin_dist'])
    origin_detector_dist = float(geometry['origin_detector_dist'])
    detector_width = projections.shape[2]
    detector_height = projections.shape[1]
    
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Create volume geometry (centered at origin)
    vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1], volume_shape[2])
    
    # Create vectors for cone beam geometry
    vectors = np.zeros((len(angles_rad), 12))
    for i, angle in enumerate(angles_rad):
        # Source position
        vectors[i, 0] = np.sin(angle) * source_origin_dist  # x
        vectors[i, 1] = -np.cos(angle) * source_origin_dist  # y
        vectors[i, 2] = 0  # z
        
        # Detector center
        vectors[i, 3] = -np.sin(angle) * origin_detector_dist  # x
        vectors[i, 4] = np.cos(angle) * origin_detector_dist  # y
        vectors[i, 5] = 0  # z
        
        # Detector u direction (columns)
        vectors[i, 6] = np.cos(angle)  # x
        vectors[i, 7] = np.sin(angle)  # y
        vectors[i, 8] = 0  # z
        
        # Detector v direction (rows)
        vectors[i, 9] = 0  # x
        vectors[i, 10] = 0  # y
        vectors[i, 11] = 1  # z
    
    # Create cone beam projection geometry
    proj_geom = astra.create_proj_geom('cone_vec', detector_height, detector_width, vectors)
    
    # Create 3D projections data
    projections_astra = np.transpose(projections, (1, 0, 2))
    projections_id = astra.data3d.create('-proj3d', proj_geom, projections_astra)
    
    # Create 3D volume data
    volume_id = astra.data3d.create('-vol', vol_geom)
    
    # Configure reconstruction
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = volume_id
    cfg['ProjectionDataId'] = projections_id
    
    # Create and run algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # Get volume data
    volume = astra.data3d.get(volume_id)
    
    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(volume_id)
    astra.data3d.delete(projections_id)
    
    return volume

def create_astra_geometric_parameters(geometry_dict):
    """
    Create ASTRA-compatible geometric parameters from our geometry dictionary.
    
    Args:
        geometry_dict (dict): Our geometry dictionary.
        
    Returns:
        dict: ASTRA-compatible geometry parameters.
    """
    # Extract parameters
    angles = geometry_dict['angles']
    detector_shape = geometry_dict['detector_shape']
    source_origin_dist = geometry_dict['source_origin_dist']
    origin_detector_dist = geometry_dict['origin_detector_dist']
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    # Create parameter dictionary
    astra_params = {
        'angles': angles_rad,
        'detector_shape': detector_shape,
        'source_origin_dist': source_origin_dist,
        'origin_detector_dist': origin_detector_dist
    }
    
    return astra_params
