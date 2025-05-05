"""
GPU-accelerated reconstruction algorithms for tomographic reconstruction.
"""
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. GPU acceleration will not be used.")

def filtered_backprojection_gpu(projections, angles, volume_shape=None, filter_name='ramp'):
    """
    GPU-accelerated Filtered backprojection algorithm for parallel beam geometry.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple, optional): Shape of the output volume.
        filter_name (str): Filter to use ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann').
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration. Please install with: pip install cupy-cuda12x")
    
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    if volume_shape is None:
        size = max(projections.shape[1], projections.shape[2])
        volume_shape = (size, size, projections.shape[1])
    
    # Transfer data to GPU
    projections_gpu = cp.asarray(projections)
    angles_rad_gpu = cp.asarray(angles_rad)
    
    # Initialize volume on GPU
    volume_gpu = cp.zeros(volume_shape, dtype=cp.float32)
    
    # For each slice along the rotation axis
    for slice_idx in range(projections.shape[1]):
        # Extract sinogram for current slice
        sinogram = projections_gpu[:, slice_idx, :]
        
        # Apply filter in frequency domain
        sinogram_filtered = apply_filter_gpu(sinogram, filter_name)
        
        # Backproject the filtered sinogram
        slice_recon = backproject_gpu(sinogram_filtered, angles_rad_gpu, (volume_shape[0], volume_shape[1]))
        
        # Add to volume
        volume_gpu[:, :, slice_idx] = slice_recon
    
    # Transfer result back to CPU
    volume = cp.asnumpy(volume_gpu)
    
    return volume

def apply_filter_gpu(sinogram, filter_name):
    """
    Apply filter to sinogram in frequency domain using GPU.
    
    Args:
        sinogram (cp.ndarray): Sinogram data on GPU.
        filter_name (str): Name of the filter.
        
    Returns:
        cp.ndarray: Filtered sinogram on GPU.
    """
    # Get dimensions
    n_angles, n_detector = sinogram.shape
    
    # Prepare filter
    filter_len = max(64, 2**int(np.log2(n_detector) + 1))
    
    # Create ramp filter
    freq = cp.fft.fftfreq(filter_len).reshape(-1, 1)
    omega = 2 * cp.pi * freq
    
    # Choose filter based on name
    if filter_name == 'ramp':
        filt = cp.abs(omega)
    elif filter_name == 'shepp-logan':
        filt = cp.abs(omega) * cp.sinc(omega / (2 * cp.pi))
    elif filter_name == 'cosine':
        filt = cp.abs(omega) * cp.cos(omega)
    elif filter_name == 'hamming':
        filt = cp.abs(omega) * (0.54 + 0.46 * cp.cos(omega / 2))
    elif filter_name == 'hann':
        filt = cp.abs(omega) * (0.5 + 0.5 * cp.cos(omega / 2))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    # Zero out the DC component
    filt[0] = 0
    
    # Create filtered sinogram
    filtered_sinogram = cp.zeros_like(sinogram)
    
    # Apply filter to each projection
    for i in range(n_angles):
        # Pad projection
        padded_projection = cp.zeros(filter_len)
        padded_projection[:n_detector] = sinogram[i]
        
        # FFT
        projection_fft = cp.fft.fft(padded_projection)
        
        # Apply filter
        filtered_projection_fft = projection_fft * filt.ravel()
        
        # IFFT and take real part
        filtered_projection = cp.real(cp.fft.ifft(filtered_projection_fft))
        
        # Crop and store
        filtered_sinogram[i] = filtered_projection[:n_detector]
    
    return filtered_sinogram

def backproject_gpu(sinogram, angles, output_shape):
    """
    Backproject a filtered sinogram to create a 2D image using GPU.
    
    Args:
        sinogram (cp.ndarray): Filtered sinogram on GPU.
        angles (cp.ndarray): Projection angles in radians on GPU.
        output_shape (tuple): Shape of the output image.
        
    Returns:
        cp.ndarray: Backprojected image on GPU.
    """
    # Create coordinate grid for the output image
    x = cp.arange(output_shape[1], dtype=cp.float32) - output_shape[1] // 2
    y = cp.arange(output_shape[0], dtype=cp.float32) - output_shape[0] // 2
    X, Y = cp.meshgrid(x, y)
    
    # Initialize output
    output = cp.zeros(output_shape, dtype=cp.float32)
    detector_center = sinogram.shape[1] // 2
    
    # Define a GPU kernel for faster backprojection
    # This is where the real GPU acceleration happens
    backproject_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void backproject_kernel(const float* sinogram, const float* angles,
                            float* output, const int n_angles, 
                            const int width, const int height, 
                            const int detector_size, const int detector_center) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= width || y >= height) return;
        
        float x_centered = x - width / 2.0f;
        float y_centered = y - height / 2.0f;
        float pixel_value = 0.0f;
        
        for (int i = 0; i < n_angles; i++) {
            float theta = angles[i];
            float t = x_centered * cos(theta) + y_centered * sin(theta);
            int t_idx = round(t + detector_center);
            
            if (t_idx >= 0 && t_idx < detector_size) {
                pixel_value += sinogram[i * detector_size + t_idx];
            }
        }
        
        output[y * width + x] = pixel_value * 3.14159f / n_angles;
    }
    ''', 'backproject_kernel')
    
    # Prepare data for the kernel
    sinogram_flat = sinogram.ravel()
    output_flat = output.ravel()
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (output_shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
        (output_shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    )
    
    # Launch the kernel
    backproject_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            sinogram_flat,
            angles,
            output_flat,
            angles.size,
            output_shape[1],
            output_shape[0],
            sinogram.shape[1],
            detector_center
        )
    )
    
    return output

def fdk_reconstruction_gpu(projections, geometry, volume_shape):
    """
    GPU-accelerated Feldkamp-Davis-Kress (FDK) algorithm for cone-beam CT.
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        geometry (dict): Projection geometry.
        volume_shape (tuple): Shape of the output volume.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration. Please install with: pip install cupy-cuda12x")
    
    # Extract parameters
    angles = geometry['angles']
    source_origin_dist = geometry['source_origin_dist']
    
    # Transfer to GPU
    projections_gpu = cp.asarray(projections)
    angles_gpu = cp.asarray(angles)
    
    # Weight projections by distance
    weighted_projections = cp.zeros_like(projections_gpu)
    for i in range(projections.shape[0]):
        # Calculate weighting factor based on cone angle
        det_center = projections.shape[2] // 2
        det_pixels = cp.arange(projections.shape[2]) - det_center
        weights = source_origin_dist / cp.sqrt(source_origin_dist**2 + det_pixels**2)
        
        # Apply weighting
        for j in range(projections.shape[1]):
            weighted_projections[i, j, :] = projections_gpu[i, j, :] * weights
    
    # Apply filtered backprojection with weighted projections
    volume = filtered_backprojection_gpu(cp.asnumpy(weighted_projections), cp.asnumpy(angles_gpu), volume_shape)
    
    return volume

def sirt_reconstruction_gpu(projections, angles, volume_shape, iterations=10):
    """
    GPU-accelerated Simultaneous Iterative Reconstruction Technique (SIRT).
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume.
        iterations (int): Number of iterations.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration. Please install with: pip install cupy-cuda12x")
    
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Transfer data to GPU
    projections_gpu = cp.asarray(projections)
    angles_rad_gpu = cp.asarray(angles_rad)
    
    # Initialize volume on GPU
    volume_gpu = cp.zeros(volume_shape, dtype=cp.float32)
    
    # For each slice
    for slice_idx in range(projections.shape[1]):
        # Extract sinogram for current slice
        sinogram = projections_gpu[:, slice_idx, :]
        
        # Initialize slice
        recon_slice = cp.zeros((volume_shape[0], volume_shape[1]), dtype=cp.float32)
        
        # SIRT iterations
        for _ in range(iterations):
            # Initialize correction term
            correction = cp.zeros_like(recon_slice)
            
            # For each angle
            for i, angle in enumerate(angles_rad_gpu):
                # Forward projection (simplified for demo)
                forward_proj = forward_project_gpu(recon_slice, angle)
                
                # Compute error
                error = sinogram[i] - forward_proj
                
                # Backproject error
                backproj_error = backproject_single_gpu(error, angle, (volume_shape[0], volume_shape[1]))
                
                # Accumulate correction
                correction += backproj_error
            
            # Update slice with average correction
            recon_slice += correction / len(angles)
        
        # Add to volume
        volume_gpu[:, :, slice_idx] = recon_slice
    
    # Transfer result back to CPU
    volume = cp.asnumpy(volume_gpu)
    
    return volume

def forward_project_gpu(image, angle):
    """
    Forward project an image at given angle using GPU.
    
    Args:
        image (cp.ndarray): 2D image on GPU.
        angle (float): Projection angle in radians.
        
    Returns:
        cp.ndarray: 1D projection on GPU.
    """
    # Get image dimensions
    height, width = image.shape
    
    # Create rotation matrix
    cos_theta = cp.cos(angle)
    sin_theta = cp.sin(angle)
    
    # Define rotation kernel
    rot_kernel = cp.ElementwiseKernel(
        'T img, int32 height, int32 width, float32 cos_theta, float32 sin_theta',
        'T rotated',
        '''
        int y = i / width;
        int x = i % width;
        float x_center = x - width/2.0f;
        float y_center = y - height/2.0f;
        float x_rot = x_center * cos_theta - y_center * sin_theta + width/2.0f;
        float y_rot = x_center * sin_theta + y_center * cos_theta + height/2.0f;
        
        if (x_rot >= 0 && x_rot < width-1 && y_rot >= 0 && y_rot < height-1) {
            int x1 = (int)floor(x_rot);
            int y1 = (int)floor(y_rot);
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            float dx = x_rot - x1;
            float dy = y_rot - y1;
            
            rotated = img[y1 * width + x1] * (1-dx) * (1-dy) +
                      img[y1 * width + x2] * dx * (1-dy) +
                      img[y2 * width + x1] * (1-dx) * dy +
                      img[y2 * width + x2] * dx * dy;
        } else {
            rotated = 0;
        }
        ''',
        'rotation_kernel'
    )
    
    # Rotate image
    rotated = cp.zeros_like(image)
    rot_kernel(image.ravel(), height, width, cos_theta, sin_theta, rotated.ravel())
    
    # Sum along columns to get projection
    projection = cp.sum(rotated, axis=0)
    
    return projection

def backproject_single_gpu(projection, angle, output_shape):
    """
    Backproject a single projection to create a 2D image using GPU.
    
    Args:
        projection (cp.ndarray): 1D projection on GPU.
        angle (float): Projection angle in radians.
        output_shape (tuple): Shape of the output image.
        
    Returns:
        cp.ndarray: Backprojected image on GPU.
    """
    # Create empty image
    bp = cp.zeros(output_shape, dtype=cp.float32)
    
    # Fill with projection value
    for i in range(output_shape[0]):
        bp[i, :] = projection
    
    # Rotate back
    cos_theta = cp.cos(-angle)
    sin_theta = cp.sin(-angle)
    
    # Define rotation kernel
    rot_kernel = cp.ElementwiseKernel(
        'T img, int32 height, int32 width, float32 cos_theta, float32 sin_theta',
        'T rotated',
        '''
        int y = i / width;
        int x = i % width;
        float x_center = x - width/2.0f;
        float y_center = y - height/2.0f;
        float x_rot = x_center * cos_theta - y_center * sin_theta + width/2.0f;
        float y_rot = x_center * sin_theta + y_center * cos_theta + height/2.0f;
        
        if (x_rot >= 0 && x_rot < width-1 && y_rot >= 0 && y_rot < height-1) {
            int x1 = (int)floor(x_rot);
            int y1 = (int)floor(y_rot);
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            float dx = x_rot - x1;
            float dy = y_rot - y1;
            
            rotated = img[y1 * width + x1] * (1-dx) * (1-dy) +
                      img[y1 * width + x2] * dx * (1-dy) +
                      img[y2 * width + x1] * (1-dx) * dy +
                      img[y2 * width + x2] * dx * dy;
        } else {
            rotated = 0;
        }
        ''',
        'rotation_kernel'
    )
    
    # Rotate image
    rotated = cp.zeros_like(bp)
    rot_kernel(bp.ravel(), output_shape[0], output_shape[1], cos_theta, sin_theta, rotated.ravel())
    
    return rotated
