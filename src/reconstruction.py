"""
Reconstruction algorithms for tomographic reconstruction.
"""
import numpy as np
from scipy import ndimage
from tqdm import tqdm

def generate_projector_matrix(geometry, volume_shape):
    """
    Generate a simplified projector matrix for tomographic reconstruction.
    This is a basic implementation and would be replaced by more efficient
    methods in a production-ready code.
    
    Args:
        geometry (dict): Projection geometry.
        volume_shape (tuple): Shape of the volume to reconstruct.
        
    Returns:
        list: List of projection matrices for each angle.
    """
    # This is a placeholder for a real projector matrix generation
    # In practice, you would use more sophisticated methods or libraries
    angles = geometry['angles']
    projectors = []
    
    for angle in angles:
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create a simple rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        projectors.append(rot_matrix)
    
    return projectors

def filtered_backprojection(projections, angles, volume_shape=None, filter_name='ramp'):
    """
    Filtered backprojection algorithm for parallel beam geometry.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple, optional): Shape of the output volume.
        filter_name (str): Filter to use ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann').
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    if volume_shape is None:
        size = max(projections.shape[1], projections.shape[2])
        volume_shape = (size, size, projections.shape[1])
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # For each slice along the rotation axis
    for slice_idx in tqdm(range(projections.shape[1]), desc="FBP Reconstruction"):
        # Extract sinogram for current slice
        sinogram = projections[:, slice_idx, :]
        
        # Apply filter in frequency domain
        sinogram_filtered = apply_filter(sinogram, filter_name)
        
        # Backproject the filtered sinogram
        slice_recon = backproject(sinogram_filtered, angles_rad, (volume_shape[0], volume_shape[1]))
        
        # Add to volume
        volume[:, :, slice_idx] = slice_recon
    
    return volume

def apply_filter(sinogram, filter_name):
    """
    Apply filter to sinogram in frequency domain.
    
    Args:
        sinogram (np.ndarray): Sinogram data.
        filter_name (str): Name of the filter.
        
    Returns:
        np.ndarray: Filtered sinogram.
    """
    # Get dimensions
    n_angles, n_detector = sinogram.shape
    
    # Prepare filter
    filter_len = max(64, 2**int(np.log2(n_detector) + 1))
    
    # Create ramp filter
    freq = np.fft.fftfreq(filter_len).reshape(-1, 1)
    omega = 2 * np.pi * freq
    
    # Choose filter based on name
    if filter_name == 'ramp':
        filt = np.abs(omega)
    elif filter_name == 'shepp-logan':
        filt = np.abs(omega) * np.sinc(omega / (2 * np.pi))
    elif filter_name == 'cosine':
        filt = np.abs(omega) * np.cos(omega)
    elif filter_name == 'hamming':
        filt = np.abs(omega) * (0.54 + 0.46 * np.cos(omega / 2))
    elif filter_name == 'hann':
        filt = np.abs(omega) * (0.5 + 0.5 * np.cos(omega / 2))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    # Zero out the DC component
    filt[0] = 0
    
    # Create filtered sinogram
    filtered_sinogram = np.zeros_like(sinogram)
    
    # Apply filter to each projection
    for i in range(n_angles):
        # Pad projection
        padded_projection = np.zeros(filter_len)
        padded_projection[:n_detector] = sinogram[i]
        
        # FFT
        projection_fft = np.fft.fft(padded_projection)
        
        # Apply filter
        filtered_projection_fft = projection_fft * filt.ravel()
        
        # IFFT and take real part
        filtered_projection = np.real(np.fft.ifft(filtered_projection_fft))
        
        # Crop and store
        filtered_sinogram[i] = filtered_projection[:n_detector]
    
    return filtered_sinogram

def backproject(sinogram, angles, output_shape):
    """
    Backproject a filtered sinogram to create a 2D image.
    
    Args:
        sinogram (np.ndarray): Filtered sinogram.
        angles (np.ndarray): Projection angles in radians.
        output_shape (tuple): Shape of the output image.
        
    Returns:
        np.ndarray: Backprojected image.
    """
    # Create coordinate grid for the output image
    x = np.arange(output_shape[1]) - output_shape[1] // 2
    y = np.arange(output_shape[0]) - output_shape[0] // 2
    X, Y = np.meshgrid(x, y)
    
    # Convert to float32 for better performance
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Initialize output
    output = np.zeros(output_shape, dtype=np.float32)
    detector_center = sinogram.shape[1] // 2
    detector_size = sinogram.shape[1]
    
    # For each angle
    for i, theta in enumerate(angles):
        # Calculate detector coordinates for each pixel
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        t = X * cos_theta + Y * sin_theta
        
        # Convert to detector pixel coordinates
        t_idx = np.round(t + detector_center).astype(np.int32)
        
        # Apply bounds (vectorized)
        valid = np.logical_and(t_idx >= 0, t_idx < detector_size)
        
        # Extract valid indices and process in chunks to avoid memory issues
        y_indices, x_indices = np.where(valid)
        chunk_size = 1000000  # Process in chunks to avoid memory issues
        
        for chunk_start in range(0, len(y_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(y_indices))
            chunk_y = y_indices[chunk_start:chunk_end]
            chunk_x = x_indices[chunk_start:chunk_end]
            chunk_t = t_idx[chunk_y, chunk_x]
            
            # Update output directly
            output[chunk_y, chunk_x] += sinogram[i, chunk_t]
    
    # Normalize by the number of angles
    return output * np.pi / len(angles)

def art_reconstruction(projections, angles, volume_shape, iterations=10, relaxation=0.1):
    """
    Algebraic Reconstruction Technique (ART).
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume.
        iterations (int): Number of iterations.
        relaxation (float): Relaxation parameter.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # For each slice
    for slice_idx in tqdm(range(projections.shape[1]), desc="ART Reconstruction"):
        # Extract sinogram for current slice
        sinogram = projections[:, slice_idx, :]
        
        # Initialize slice
        recon_slice = np.zeros((volume_shape[0], volume_shape[1]), dtype=np.float32)
        
        # ART iterations
        for _ in range(iterations):
            for i, angle in enumerate(angles):
                # Forward projection
                forward_proj = forward_project(recon_slice, angle)
                
                # Compute error
                error = sinogram[i] - forward_proj
                
                # Backproject error and update
                recon_slice += relaxation * backproject_single(error, angle, (volume_shape[0], volume_shape[1]))
        
        # Add to volume
        volume[:, :, slice_idx] = recon_slice
    
    return volume

def forward_project(image, angle_deg):
    """
    Forward project an image at given angle.
    
    Args:
        image (np.ndarray): 2D image.
        angle_deg (float): Projection angle in degrees.
        
    Returns:
        np.ndarray: 1D projection.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Rotate image
    rotated = ndimage.rotate(image, -angle_deg, reshape=False, order=1)
    
    # Sum along columns
    projection = np.sum(rotated, axis=0)
    
    return projection

def backproject_single(projection, angle_deg, output_shape):
    """
    Backproject a single projection to create a 2D image.
    
    Args:
        projection (np.ndarray): 1D projection.
        angle_deg (float): Projection angle in degrees.
        output_shape (tuple): Shape of the output image.
        
    Returns:
        np.ndarray: Backprojected image.
    """
    # Create empty image
    bp = np.zeros(output_shape, dtype=np.float32)
    
    # Fill with projection value
    for i in range(output_shape[0]):
        bp[i, :] = projection
    
    # Rotate back
    bp = ndimage.rotate(bp, angle_deg, reshape=False, order=1)
    
    return bp

def sirt_reconstruction(projections, angles, volume_shape, iterations=10):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume.
        iterations (int): Number of iterations.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # For each slice
    for slice_idx in tqdm(range(projections.shape[1]), desc="SIRT Reconstruction"):
        # Extract sinogram for current slice
        sinogram = projections[:, slice_idx, :]
        
        # Initialize slice
        recon_slice = np.zeros((volume_shape[0], volume_shape[1]), dtype=np.float32)
        
        # SIRT iterations
        for _ in range(iterations):
            # Initialize correction term
            correction = np.zeros_like(recon_slice)
            
            # For each angle
            for i, angle in enumerate(angles):
                # Forward projection
                forward_proj = forward_project(recon_slice, angle)
                
                # Compute error
                error = sinogram[i] - forward_proj
                
                # Backproject error
                correction += backproject_single(error, angle, (volume_shape[0], volume_shape[1]))
            
            # Update slice with average correction
            recon_slice += correction / len(angles)
        
        # Add to volume
        volume[:, :, slice_idx] = recon_slice
    
    return volume

def fdk_reconstruction(projections, geometry, volume_shape):
    """
    Feldkamp-Davis-Kress (FDK) algorithm for cone-beam CT.
    This is a simplified version - a complete implementation would be more complex.
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        geometry (dict): Projection geometry.
        volume_shape (tuple): Shape of the output volume.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Extract parameters
    angles = geometry['angles']
    source_origin_dist = geometry['source_origin_dist']
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Weight projections by distance
    weighted_projections = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        # Calculate weighting factor based on cone angle
        det_center = projections.shape[2] // 2
        det_pixels = np.arange(projections.shape[2]) - det_center
        weights = source_origin_dist / np.sqrt(source_origin_dist**2 + det_pixels**2)
        
        # Apply weighting
        for j in range(projections.shape[1]):
            weighted_projections[i, j, :] = projections[i, j, :] * weights
    
    # Apply filtered backprojection with weighted projections
    volume = filtered_backprojection(weighted_projections, angles, volume_shape)
    
    return volume
