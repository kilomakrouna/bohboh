import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import json
from skimage import measure, filters, morphology, segmentation
import matplotlib.colors as colors
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def analyze_volume_statistics(volume):
    """
    Analyze volume statistics to help with automatic parameter selection
    
    Parameters:
    -----------
    volume : ndarray
        3D volume data
        
    Returns:
    --------
    stats : dict
        Dictionary containing volume statistics
    """
    stats = {
        'shape': volume.shape,
        'min': float(volume.min()),
        'max': float(volume.max()),
        'mean': float(volume.mean()),
        'std': float(volume.std()),
        'median': float(np.median(volume)),
        'percentiles': {
            '5': float(np.percentile(volume, 5)),
            '10': float(np.percentile(volume, 10)),
            '25': float(np.percentile(volume, 25)),
            '75': float(np.percentile(volume, 75)),
            '85': float(np.percentile(volume, 85)),
            '90': float(np.percentile(volume, 90)),
            '95': float(np.percentile(volume, 95)),
            '99': float(np.percentile(volume, 99))
        }
    }
    
    # Estimate optimal thresholds
    # Method 1: Otsu's method for automatic thresholding
    try:
        from skimage.filters import threshold_otsu
        otsu_thresh = threshold_otsu(volume)
        stats['otsu_threshold'] = float(otsu_thresh)
    except:
        stats['otsu_threshold'] = stats['mean'] + 0.5 * stats['std']
    
    # Method 2: Multiple threshold suggestions
    stats['suggested_thresholds'] = {
        'conservative': stats['percentiles']['75'],
        'moderate': stats['percentiles']['85'],
        'aggressive': stats['percentiles']['95']
    }
    
    print(f"Volume Analysis:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"  Otsu threshold: {stats['otsu_threshold']:.3f}")
    print(f"  Suggested thresholds: {stats['suggested_thresholds']}")
    
    return stats

def normalize_volume(volume, method='minmax', clip_percentile=1):
    """
    Normalize volume with different methods
    
    Parameters:
    -----------
    volume : ndarray
        Input volume
    method : str
        Normalization method ('minmax', 'zscore', 'robust')
    clip_percentile : float
        Percentile for robust normalization
        
    Returns:
    --------
    normalized_volume : ndarray
        Normalized volume
    """
    if method == 'minmax':
        min_val = volume.min()
        max_val = volume.max()
        return (volume - min_val) / (max_val - min_val + 1e-10)
    
    elif method == 'zscore':
        mean_val = volume.mean()
        std_val = volume.std()
        normalized = (volume - mean_val) / (std_val + 1e-10)
        # Clip to reasonable range and rescale to [0,1]
        normalized = np.clip(normalized, -3, 3)
        return (normalized + 3) / 6
    
    elif method == 'robust':
        # Use percentile-based normalization to handle outliers
        p_low = np.percentile(volume, clip_percentile)
        p_high = np.percentile(volume, 100 - clip_percentile)
        clipped = np.clip(volume, p_low, p_high)
        return (clipped - p_low) / (p_high - p_low + 1e-10)
    
    return volume

def detect_voxel_spacing(volume, ct_params=None):
    """
    Detect or estimate voxel spacing
    
    Parameters:
    -----------
    volume : ndarray
        3D volume data
    ct_params : dict, optional
        CT parameters
        
    Returns:
    --------
    voxel_spacing : list
        Estimated voxel spacing [x, y, z]
    """
    # Default isotropic voxels
    voxel_spacing = [1.0, 1.0, 1.0]
    
    if ct_params:
        # Try to extract voxel spacing from CT parameters
        try:
            if 'vgParam' in ct_params and 'ResultVoxelPixelSize' in ct_params['vgParam']:
                spacing = ct_params['vgParam']['ResultVoxelPixelSize']
                if isinstance(spacing, list) and len(spacing) == 3:
                    # Normalize spacing
                    max_spacing = max(spacing)
                    voxel_spacing = [s/max_spacing for s in spacing]
                    print(f"Extracted voxel spacing: {voxel_spacing}")
                    return voxel_spacing
        except Exception as e:
            print(f"Could not extract voxel spacing from CT params: {e}")
    
    # Estimate voxel spacing from volume shape if very anisotropic
    shape = volume.shape
    if max(shape) / min(shape) > 2:
        # Assume the largest dimension corresponds to the rotation axis
        # and should have finer spacing
        max_dim = max(shape)
        voxel_spacing = [max_dim/shape[2], max_dim/shape[1], max_dim/shape[0]]
        # Normalize
        max_spacing = max(voxel_spacing)
        voxel_spacing = [s/max_spacing for s in voxel_spacing]
        print(f"Estimated voxel spacing from shape: {voxel_spacing}")
    
    return voxel_spacing

def preprocess_volume(volume, remove_noise=True, fill_holes=True, smooth=True):
    """
    Preprocess volume to reduce reconstruction artifacts
    
    Parameters:
    -----------
    volume : ndarray
        Input volume
    remove_noise : bool
        Apply noise reduction
    fill_holes : bool
        Fill small holes
    smooth : bool
        Apply smoothing
        
    Returns:
    --------
    processed_volume : ndarray
        Processed volume
    """
    processed = volume.copy()
    
    if remove_noise:
        # Apply median filter to reduce noise
        processed = ndimage.median_filter(processed, size=3)
    
    if smooth:
        # Apply Gaussian smoothing
        processed = ndimage.gaussian_filter(processed, sigma=0.8)
    
    if fill_holes:
        # Binary closing to fill small holes (only if volume appears binary-like)
        if len(np.unique(processed)) < 20:  # Likely binary or few-valued
            # Create binary mask and apply morphological operations
            threshold = np.percentile(processed, 75)
            binary_mask = processed > threshold
            binary_mask = morphology.binary_closing(binary_mask, morphology.ball(2))
            processed = np.where(binary_mask, processed, 0)
    
    return processed

def agnostic_3d_visualizer(volume, ct_params=None, output_dir='agnostic_visualization',
                          auto_analyze=True, preprocessing=True):
    """
    Agnostic 3D volume visualizer that adapts to different data types
    
    Parameters:
    -----------
    volume : ndarray
        3D volume data
    ct_params : dict, optional
        CT parameters (can be None)
    output_dir : str
        Directory to save visualizations
    auto_analyze : bool
        Automatically analyze volume and suggest parameters
    preprocessing : bool
        Apply preprocessing to reduce artifacts
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Interactive visualization figure
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze volume statistics
    if auto_analyze:
        stats = analyze_volume_statistics(volume)
        initial_threshold = stats['suggested_thresholds']['moderate']
        # Normalize based on statistics
        if stats['max'] - stats['min'] > 10:  # Large dynamic range
            volume_norm = normalize_volume(volume, method='robust')
        else:
            volume_norm = normalize_volume(volume, method='minmax')
    else:
        volume_norm = normalize_volume(volume)
        initial_threshold = 0.5
    
    # Apply preprocessing if requested
    if preprocessing:
        print("Applying preprocessing to reduce artifacts...")
        volume_norm = preprocess_volume(volume_norm)
    
    # Detect voxel spacing
    voxel_spacing = detect_voxel_spacing(volume_norm, ct_params)
    
    # Set up the interactive figure
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax_main = fig.add_subplot(221, projection='3d')
    
    # Histogram plot
    ax_hist = fig.add_subplot(222)
    hist_counts, hist_bins, _ = ax_hist.hist(volume_norm.flatten(), bins=100, alpha=0.7)
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Intensity Distribution')
    ax_hist.set_yscale('log')
    
    # Slice views
    ax_slice_xy = fig.add_subplot(223)
    ax_slice_xz = fig.add_subplot(224)
    
    # Current visualization parameters
    current_params = {
        'threshold': initial_threshold,
        'method': 'surface',
        'colormap': 'viridis',
        'normalization': 'robust' if auto_analyze and stats['max'] - stats['min'] > 10 else 'minmax',
        'preprocessing': preprocessing
    }
    
    # Threshold line on histogram
    threshold_line = ax_hist.axvline(current_params['threshold'], color='red', 
                                   linestyle='--', label='Threshold')
    ax_hist.legend()
    
    def update_visualization():
        """Update all visualization components"""
        # Clear main axis
        ax_main.clear()
        
        threshold = current_params['threshold']
        method = current_params['method']
        colormap = current_params['colormap']
        
        # Update threshold line
        threshold_line.set_xdata([threshold, threshold])
        
        try:
            if method == 'surface':
                # Marching cubes with error handling
                try:
                    verts, faces, normals, values = measure.marching_cubes(
                        volume_norm, 
                        level=threshold,
                        spacing=voxel_spacing,
                        step_size=1
                    )
                    
                    if len(verts) > 0:
                        surf = ax_main.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            cmap=colormap,
                            alpha=0.8,
                            shade=True,
                            edgecolor='none'
                        )
                    else:
                        ax_main.text(0.5, 0.5, 0.5, 'No surface found\nTry lower threshold', 
                                   transform=ax_main.transAxes, ha='center', va='center')
                        
                except Exception as e:
                    print(f"Surface extraction failed: {e}")
                    # Fallback to points
                    current_params['method'] = 'points'
                    update_visualization()
                    return
            
            elif method == 'points':
                # Point cloud visualization
                binary_mask = volume_norm > threshold
                z_indices, y_indices, x_indices = np.where(binary_mask)
                
                if len(z_indices) > 0:
                    # Subsample for performance
                    max_points = 50000
                    if len(z_indices) > max_points:
                        step = len(z_indices) // max_points
                        z_indices = z_indices[::step]
                        y_indices = y_indices[::step]
                        x_indices = x_indices[::step]
                    
                    intensities = volume_norm[z_indices, y_indices, x_indices]
                    
                    # Apply voxel spacing
                    x_scaled = x_indices * voxel_spacing[0]
                    y_scaled = y_indices * voxel_spacing[1]
                    z_scaled = z_indices * voxel_spacing[2]
                    
                    scatter = ax_main.scatter(
                        x_scaled, y_scaled, z_scaled,
                        c=intensities,
                        cmap=colormap,
                        s=0.5,
                        alpha=0.6
                    )
                else:
                    ax_main.text(0.5, 0.5, 0.5, 'No points above threshold', 
                               transform=ax_main.transAxes, ha='center', va='center')
            
            elif method == 'contour':
                # Multi-level contour visualization
                levels = np.linspace(threshold, volume_norm.max(), 5)
                try:
                    for level in levels:
                        verts, faces, _, _ = measure.marching_cubes(
                            volume_norm, level=level, spacing=voxel_spacing, step_size=2
                        )
                        if len(verts) > 0:
                            ax_main.plot_trisurf(
                                verts[:, 0], verts[:, 1], verts[:, 2],
                                triangles=faces,
                                alpha=0.3,
                                color=plt.cm.get_cmap(colormap)(level)
                            )
                except:
                    ax_main.text(0.5, 0.5, 0.5, 'Contour extraction failed', 
                               transform=ax_main.transAxes, ha='center', va='center')
            
            # Update slice views
            z_mid = volume_norm.shape[0] // 2
            x_mid = volume_norm.shape[2] // 2
            
            ax_slice_xy.clear()
            im1 = ax_slice_xy.imshow(volume_norm[z_mid], cmap=colormap, aspect='equal')
            ax_slice_xy.set_title(f'Slice Z={z_mid}')
            ax_slice_xy.axhline(y=threshold*volume_norm.shape[1], color='red', alpha=0.5)
            
            ax_slice_xz.clear()
            im2 = ax_slice_xz.imshow(volume_norm[:, :, x_mid], cmap=colormap, aspect='equal')
            ax_slice_xz.set_title(f'Slice X={x_mid}')
            ax_slice_xz.axhline(y=threshold*volume_norm.shape[0], color='red', alpha=0.5)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            ax_main.text(0.5, 0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                       transform=ax_main.transAxes, ha='center', va='center')
        
        # Set labels and title
        ax_main.set_xlabel('X')
        ax_main.set_ylabel('Y')
        ax_main.set_zlabel('Z')
        ax_main.set_title(f'3D Visualization ({method}, t={threshold:.3f})')
        
        # Set axis limits
        max_dim = max(d*s for d, s in zip(volume_norm.shape, voxel_spacing))
        ax_main.set_xlim(0, volume_norm.shape[2] * voxel_spacing[0])
        ax_main.set_ylim(0, volume_norm.shape[1] * voxel_spacing[1])
        ax_main.set_zlim(0, volume_norm.shape[0] * voxel_spacing[2])
        
        fig.canvas.draw_idle()
    
    # Create interactive controls
    plt.subplots_adjust(bottom=0.25, right=0.85)
    
    # Threshold slider
    ax_threshold = plt.axes([0.1, 0.15, 0.6, 0.03])
    slider_threshold = Slider(ax_threshold, 'Threshold', 0.0, 1.0, 
                             valinit=current_params['threshold'])
    
    # Method selection
    ax_method = plt.axes([0.85, 0.6, 0.13, 0.15])
    radio_method = RadioButtons(ax_method, ('surface', 'points', 'contour'))
    
    # Colormap selection
    ax_colormap = plt.axes([0.85, 0.4, 0.13, 0.15])
    radio_colormap = RadioButtons(ax_colormap, ('viridis', 'bone', 'plasma', 'hot'))
    
    # Control buttons
    ax_save = plt.axes([0.85, 0.25, 0.13, 0.04])
    button_save = Button(ax_save, 'Save Image')
    
    ax_reset = plt.axes([0.85, 0.2, 0.13, 0.04])
    button_reset = Button(ax_reset, 'Reset View')
    
    ax_reprocess = plt.axes([0.85, 0.15, 0.13, 0.04])
    button_reprocess = Button(ax_reprocess, 'Reprocess')
    
    # Callback functions
    def update_threshold(val):
        current_params['threshold'] = val
        update_visualization()
    
    def update_method(label):
        current_params['method'] = label
        update_visualization()
    
    def update_colormap(label):
        current_params['colormap'] = label
        update_visualization()
    
    def save_figure(event):
        filename = f"agnostic_viz_{current_params['method']}_t{current_params['threshold']:.3f}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    def reset_view(event):
        ax_main.view_init(elev=30, azim=45)
        fig.canvas.draw_idle()
    
    def reprocess_data(event):
        nonlocal volume_norm
        print("Reprocessing volume...")
        # Try different normalization
        if current_params['normalization'] == 'minmax':
            volume_norm = normalize_volume(volume, method='robust')
            current_params['normalization'] = 'robust'
        else:
            volume_norm = normalize_volume(volume, method='minmax')
            current_params['normalization'] = 'minmax'
        
        # Reapply preprocessing
        if current_params['preprocessing']:
            volume_norm = preprocess_volume(volume_norm)
        
        print(f"Switched to {current_params['normalization']} normalization")
        update_visualization()
    
    # Connect callbacks
    slider_threshold.on_changed(update_threshold)
    radio_method.on_clicked(update_method)
    radio_colormap.on_clicked(update_colormap)
    button_save.on_clicked(save_figure)
    button_reset.on_clicked(reset_view)
    button_reprocess.on_clicked(reprocess_data)
    
    # Initial visualization
    update_visualization()
    
    # Add help text
    help_text = """
    Agnostic 3D Volume Visualizer
    
    Controls:
    • Drag to rotate 3D view
    • Scroll to zoom
    • Adjust threshold slider
    • Try different methods if surface fails
    • Use 'Reprocess' to try different normalization
    • Red line in histogram shows current threshold
    
    Tips:
    • If you see cylindrical artifacts, try:
      1. Lower threshold values
      2. Different normalization (Reprocess)
      3. Points method instead of surface
    • Check slice views for data quality
    """
    
    fig.text(0.02, 0.02, help_text, fontsize=8, 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8))
    
    return fig

def visualize_from_file_agnostic(volume_path, json_path=None, 
                                output_dir='agnostic_visualization'):
    """
    Load and visualize volume with agnostic approach
    
    Parameters:
    -----------
    volume_path : str
        Path to .npy volume file
    json_path : str, optional
        Path to CT parameters (can be None)
    output_dir : str
        Output directory
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Visualization figure
    """
    # Load volume
    try:
        volume = np.load(volume_path)
        print(f"Loaded volume: {volume.shape}, dtype: {volume.dtype}")
    except Exception as e:
        print(f"Error loading volume: {e}")
        return None
    
    # Load CT parameters if provided
    ct_params = None
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                ct_params = json.load(f)
            print(f"Loaded CT parameters from {json_path}")
        except Exception as e:
            print(f"Could not load CT parameters: {e}")
    else:
        print("No CT parameters provided - using agnostic approach")
    
    # Create visualization
    return agnostic_3d_visualizer(volume, ct_params=ct_params, output_dir=output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Agnostic 3D Volume Visualizer')
    parser.add_argument('volume_path', help='Path to .npy volume file')
    parser.add_argument('--json', '-j', help='Path to CT parameters JSON (optional)')
    parser.add_argument('--output', '-o', default='agnostic_visualization',
                        help='Output directory')
    parser.add_argument('--no-preprocessing', action='store_true',
                        help='Disable preprocessing')
    
    args = parser.parse_args()
    
    # Load and visualize
    fig = visualize_from_file_agnostic(
        args.volume_path, 
        json_path=args.json,
        output_dir=args.output
    )
    
    if fig:
        plt.show()
    else:
        print("Visualization failed")