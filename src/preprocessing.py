"""
Preprocessing functions for tomographic reconstruction.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, transform, exposure, restoration

def normalize_projections(projections):
    """
    Normalize projection images to range [0, 1].
    
    Args:
        projections (np.ndarray): Stack of projection images.
        
    Returns:
        np.ndarray: Normalized projections.
    """
    # Check if projections need normalization
    if projections.min() >= 0 and projections.max() <= 1:
        return projections
    
    p_min = projections.min()
    p_max = projections.max()
    
    if p_min == p_max:
        return np.zeros_like(projections)
    
    return (projections - p_min) / (p_max - p_min)

def apply_log_transform(projections):
    """
    Apply log transform to convert attenuation projections to line integrals.
    
    Args:
        projections (np.ndarray): Normalized stack of projection images.
        
    Returns:
        np.ndarray: Log-transformed projections.
    """
    # Ensure values are positive and not zero
    epsilon = 1e-5
    proj_positive = np.clip(projections, epsilon, 1.0)
    
    # Apply negative log transform (I/I0 -> -ln(I/I0))
    return -np.log(proj_positive)

def correct_beam_hardening(projections, polynomial_order=2):
    """
    Apply polynomial correction for beam hardening artifacts.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        polynomial_order (int): Order of polynomial correction.
        
    Returns:
        np.ndarray: Corrected projections.
    """
    # Get flattened intensities
    flat_proj = projections.flatten()
    
    # Create polynomial features
    poly_features = np.column_stack([flat_proj**i for i in range(1, polynomial_order+1)])
    
    # Fit polynomial correction (simple approach - more sophisticated methods exist)
    coeffs = np.polyfit(flat_proj, np.linspace(0, 1, len(flat_proj)), polynomial_order)
    
    # Apply correction
    corrected = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        corrected[i] = np.polyval(coeffs, projections[i])
    
    return corrected

def denoise_projections(projections, method='gaussian', **kwargs):
    """
    Apply denoising to projection images.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        method (str): Denoising method ('gaussian', 'median', 'bilateral', 'tv_chambolle').
        **kwargs: Additional parameters for the chosen method.
        
    Returns:
        np.ndarray: Denoised projections.
    """
    denoised = np.zeros_like(projections)
    
    for i in range(projections.shape[0]):
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            denoised[i] = filters.gaussian(projections[i], sigma=sigma)
        
        elif method == 'median':
            size = kwargs.get('size', 3)
            denoised[i] = filters.median(projections[i], footprint=np.ones((size, size)))
        
        elif method == 'bilateral':
            sigma_spatial = kwargs.get('sigma_spatial', 1.0)
            sigma_range = kwargs.get('sigma_range', 0.1)
            denoised[i] = restoration.denoise_bilateral(projections[i], 
                                                       sigma_spatial=sigma_spatial,
                                                       sigma_range=sigma_range)
        
        elif method == 'tv_chambolle':
            weight = kwargs.get('weight', 0.1)
            denoised[i] = restoration.denoise_tv_chambolle(projections[i], weight=weight)
            
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    return denoised

def correct_center_of_rotation(projections, angles):
    """
    Attempt to find and correct the center of rotation.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        angles (np.ndarray): Projection angles in degrees.
        
    Returns:
        np.ndarray: Projections with adjusted center of rotation.
    """
    # Find projections that are approximately 180 degrees apart
    angle_diff = np.abs(angles[:, np.newaxis] - (angles[np.newaxis, :] - 180))
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    min_diff_idx = np.argmin(angle_diff, axis=1)
    
    # Use the middle projection for estimation
    mid_idx = len(projections) // 2
    opposite_idx = min_diff_idx[mid_idx]
    
    if np.abs(angles[mid_idx] - angles[opposite_idx] - 180) > 5:
        # No good opposite projection found
        return projections
    
    p1 = projections[mid_idx]
    p2 = np.flip(projections[opposite_idx], axis=1)  # Flip horizontally
    
    # Find shift that maximizes correlation
    corr = np.zeros(p1.shape[1])
    for i in range(p1.shape[1]):
        shifted = np.roll(p2, i, axis=1)
        corr[i] = np.corrcoef(p1.flatten(), shifted.flatten())[0, 1]
    
    shift = np.argmax(corr) - p1.shape[1] // 2
    
    # Apply shift to all projections
    if shift != 0:
        corrected = np.zeros_like(projections)
        for i in range(len(projections)):
            corrected[i] = np.roll(projections[i], shift // 2, axis=1)
        return corrected
    
    return projections

def remove_ring_artifacts(projections):
    """
    Remove ring artifacts using Fourier-Wavelet based method.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        
    Returns:
        np.ndarray: Corrected projections.
    """
    # Simple ring artifact removal
    # Convert to sinogram space (projection image with rows=detector rows, cols=angles)
    sinograms = np.transpose(projections, (1, 0, 2))
    
    corrected_sinograms = np.zeros_like(sinograms)
    
    for i in range(sinograms.shape[0]):
        sino = sinograms[i]
        
        # Apply median filter along the angle dimension
        sino_filtered = ndimage.median_filter(sino, size=(1, 3))
        
        # Compute the difference
        residual = sino - sino_filtered
        
        # Apply a threshold to the residual to identify ring artifacts
        threshold = 0.1 * np.std(residual)
        mask = np.abs(residual) > threshold
        
        # Correct the sinogram
        corrected = sino.copy()
        corrected[mask] = sino_filtered[mask]
        
        corrected_sinograms[i] = corrected
    
    # Convert back to original space
    return np.transpose(corrected_sinograms, (1, 0, 2))

def preprocess_projections(projections, angles=None, normalize=True, denoise=True, 
                         remove_rings=True, correct_rotation=True, log_transform=True):
    """
    Apply a complete preprocessing pipeline to projection images.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        angles (np.ndarray, optional): Projection angles in degrees.
        normalize (bool): Apply normalization.
        denoise (bool): Apply denoising.
        remove_rings (bool): Apply ring artifact removal.
        correct_rotation (bool): Correct center of rotation.
        log_transform (bool): Apply log transform.
        
    Returns:
        np.ndarray: Preprocessed projections.
    """
    # Create a copy to avoid modifying original data
    processed = projections.copy()
    
    # Apply preprocessing steps
    if normalize:
        processed = normalize_projections(processed)
        
    if denoise:
        processed = denoise_projections(processed)
        
    if log_transform:
        processed = apply_log_transform(processed)
    
    if remove_rings:
        processed = remove_ring_artifacts(processed)
    
    if correct_rotation and angles is not None:
        processed = correct_center_of_rotation(processed, angles)
    
    return processed
