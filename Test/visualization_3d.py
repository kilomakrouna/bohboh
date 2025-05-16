import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib.colors as colors

def load_volume(file_path):
    """
    Load a 3D volume from .npy file
    
    Parameters:
    -----------
    file_path : str
        Path to the .npy file containing the 3D volume
        
    Returns:
    --------
    volume : ndarray
        3D numpy array containing the volume data
    """
    try:
        volume = np.load(file_path)
        print(f"Loaded volume with shape: {volume.shape}")
        return volume
    except Exception as e:
        print(f"Error loading volume: {e}")
        return None

def normalize_volume(volume):
    """
    Normalize volume to range [0,1]
    
    Parameters:
    -----------
    volume : ndarray
        Input volume
        
    Returns:
    --------
    normalized_volume : ndarray
        Normalized volume in range [0,1]
    """
    min_val = volume.min()
    max_val = volume.max()
    return (volume - min_val) / (max_val - min_val + 1e-10)

def interactive_3d_plot(volume, threshold=0.5, point_size=2, alpha=0.3, 
                         subsample_factor=None, cmap='viridis', output_dir=None):
    """
    Create an interactive 3D visualization of a volume
    
    Parameters:
    -----------
    volume : ndarray
        3D numpy array containing the volume data
    threshold : float
        Initial threshold value (0-1)
    point_size : float
        Size of points in scatter plot
    alpha : float
        Transparency of points (0-1)
    subsample_factor : int, optional
        Subsampling factor for large volumes (if None, automatically calculated)
    cmap : str
        Colormap to use
    output_dir : str, optional
        Directory to save screenshots
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The interactive figure
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Normalize volume to [0,1]
    if volume.max() > 1.0:
        volume_normalized = normalize_volume(volume)
    else:
        volume_normalized = volume
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set initial threshold
    current_threshold = threshold
    
    # Create initial plot
    scatter = update_plot(ax, volume_normalized, current_threshold, 
                         point_size, alpha, subsample_factor, cmap)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1, label='Intensity')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Volume Visualization')
    
    # Add sliders for interactive control
    threshold_ax = plt.axes([0.25, 0.02, 0.55, 0.03])
    threshold_slider = Slider(
        ax=threshold_ax, 
        label='Threshold',
        valmin=0.0,
        valmax=1.0,
        valinit=current_threshold,
    )
    
    alpha_ax = plt.axes([0.25, 0.06, 0.55, 0.03])
    alpha_slider = Slider(
        ax=alpha_ax, 
        label='Opacity',
        valmin=0.0,
        valmax=1.0,
        valinit=alpha,
    )
    
    size_ax = plt.axes([0.25, 0.10, 0.55, 0.03])
    size_slider = Slider(
        ax=size_ax, 
        label='Point Size',
        valmin=0.1,
        valmax=10.0,
        valinit=point_size,
    )
    
    # Add buttons for different views
    view_ax = plt.axes([0.02, 0.4, 0.1, 0.15])
    views = RadioButtons(
        view_ax, 
        ('Free', 'Top', 'Front', 'Side'),
        active=0
    )
    
    # Add checkbox for coordinate axes
    axes_ax = plt.axes([0.02, 0.25, 0.1, 0.1])
    show_axes = CheckButtons(
        axes_ax, 
        ['Show Axes'],
        [True]
    )
    
    # Add button for saving screenshot
    save_ax = plt.axes([0.02, 0.15, 0.1, 0.05])
    save_button = Button(save_ax, 'Save')
    
    # Add reset button
    reset_ax = plt.axes([0.02, 0.07, 0.1, 0.05])
    reset_button = Button(reset_ax, 'Reset')
    
    # Create text display for statistics
    stats_ax = plt.axes([0.75, 0.8, 0.2, 0.15])
    stats_ax.axis('off')
    
    # Update statistics display
    def update_stats():
        # Create binary mask based on current threshold
        binary_mask = volume_normalized > current_threshold
        # Count points above threshold
        num_points = np.sum(binary_mask)
        # Calculate percentage of total volume
        percent = 100 * num_points / volume_normalized.size
        
        # Display statistics
        stats_text = (
            f"Volume: {volume.shape}\n"
            f"Min: {volume.min():.3f}\n"
            f"Max: {volume.max():.3f}\n"
            f"Mean: {volume.mean():.3f}\n"
            f"Points: {num_points}\n"
            f"% of volume: {percent:.2f}%"
        )
        
        # Clear previous text
        stats_ax.clear()
        stats_ax.axis('off')
        
        # Add the text with background
        stats_ax.text(0, 1, stats_text, transform=stats_ax.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Initial stats update
    update_stats()
    
    # Define callback functions for interactive elements
    def update_threshold(val):
        nonlocal current_threshold
        current_threshold = val
        scatter = update_plot(ax, volume_normalized, current_threshold, 
                             size_slider.val, alpha_slider.val, subsample_factor, cmap)
        update_stats()
        fig.canvas.draw_idle()
    
    def update_alpha(val):
        scatter = update_plot(ax, volume_normalized, current_threshold, 
                             size_slider.val, val, subsample_factor, cmap)
        fig.canvas.draw_idle()
    
    def update_size(val):
        scatter = update_plot(ax, volume_normalized, current_threshold, 
                             val, alpha_slider.val, subsample_factor, cmap)
        fig.canvas.draw_idle()
    
    def update_view(label):
        if label == 'Top':
            ax.view_init(90, 0)  # Top view (looking down z-axis)
        elif label == 'Front':
            ax.view_init(0, 0)   # Front view (looking at x-z plane)
        elif label == 'Side':
            ax.view_init(0, 90)  # Side view (looking at y-z plane)
        # For 'Free', do nothing (user can rotate freely)
        fig.canvas.draw_idle()
    
    def toggle_axes(label):
        visible = show_axes.get_status()[0]
        ax.set_axis_on() if visible else ax.set_axis_off()
        fig.canvas.draw_idle()
    
    def save_screenshot(event):
        if output_dir:
            output_path = os.path.join(output_dir, f'volume_3d_t{current_threshold:.2f}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D visualization to {output_path}")
    
    def reset(event):
        threshold_slider.reset()
        alpha_slider.reset()
        size_slider.reset()
        # Reset view
        ax.view_init(30, 30)
        fig.canvas.draw_idle()
    
    # Connect callbacks
    threshold_slider.on_changed(update_threshold)
    alpha_slider.on_changed(update_alpha)
    size_slider.on_changed(update_size)
    views.on_clicked(update_view)
    show_axes.on_clicked(toggle_axes)
    save_button.on_clicked(save_screenshot)
    reset_button.on_clicked(reset)
    
    # Enable interactive rotation
    plt.tight_layout()
    
    # Return the figure for further manipulation if needed
    return fig

def update_plot(ax, volume, threshold, point_size, alpha, subsample_factor, cmap):
    """
    Update the 3D scatter plot with new parameters
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The 3D axes to plot on
    volume : ndarray
        Normalized volume data
    threshold : float
        Threshold value (0-1)
    point_size : float
        Size of points in scatter plot
    alpha : float
        Transparency of points
    subsample_factor : int or None
        Subsampling factor (if None, automatically calculated)
    cmap : str
        Colormap to use
        
    Returns:
    --------
    scatter : matplotlib.collections.PathCollection
        The scatter plot object
    """
    # Clear previous plot
    ax.clear()
    
    # Create binary mask based on threshold
    binary_mask = volume > threshold
    
    # Get coordinates of points above threshold
    z_indices, y_indices, x_indices = np.where(binary_mask)
    
    # Get intensity values at these points
    intensities = volume[z_indices, y_indices, x_indices]
    
    # Calculate automatic subsampling if needed
    max_points = 100000  # Maximum number of points for interactive performance
    
    if subsample_factor is None:
        # Calculate automatic subsampling
        num_points = len(z_indices)
        if num_points > max_points:
            subsample_factor = int(np.ceil(num_points / max_points))
        else:
            subsample_factor = 1
    
    # Apply subsampling if factor > 1
    if subsample_factor > 1:
        z_indices = z_indices[::subsample_factor]
        y_indices = y_indices[::subsample_factor]
        x_indices = x_indices[::subsample_factor]
        intensities = intensities[::subsample_factor]
        print(f"Subsampling by factor {subsample_factor}: {len(z_indices)} points displayed")
    
    # Create scatter plot
    scatter = ax.scatter(
        x_indices, y_indices, z_indices,
        c=intensities,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        marker='.'
    )
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set axis limits to match volume dimensions
    ax.set_xlim(0, volume.shape[2])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[0])
    
    return scatter

def plot_orthogonal_slices(volume, output_dir=None):
    """
    Plot orthogonal slices through the center of the volume
    
    Parameters:
    -----------
    volume : ndarray
        3D numpy array containing the volume data
    output_dir : str, optional
        Directory to save the visualization
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the visualization
    """
    # Normalize volume if needed
    if volume.max() > 1.0:
        volume = normalize_volume(volume)
    
    # Get dimensions
    z_dim, y_dim, x_dim = volume.shape
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial slice (xy plane, z fixed)
    z_slice = z_dim // 2
    axes[0].imshow(volume[z_slice, :, :], cmap='gray')
    axes[0].set_title(f'Axial (Z={z_slice})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Coronal slice (xz plane, y fixed)
    y_slice = y_dim // 2
    axes[1].imshow(volume[:, y_slice, :], cmap='gray')
    axes[1].set_title(f'Coronal (Y={y_slice})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # Sagittal slice (yz plane, x fixed)
    x_slice = x_dim // 2
    axes[2].imshow(volume[:, :, x_slice], cmap='gray')
    axes[2].set_title(f'Sagittal (X={x_slice})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'orthogonal_slices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved orthogonal slices to {output_path}")
    
    return fig

def visualize_npy(file_path, output_dir='visualization_output', threshold=0.5, mode='3d'):
    """
    Main function to visualize .npy volume files
    
    Parameters:
    -----------
    file_path : str
        Path to the .npy file containing the 3D volume
    output_dir : str
        Directory to save visualizations
    threshold : float
        Initial threshold value for 3D visualization
    mode : str
        Visualization mode: '3d' or 'slices'
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The visualization figure
    """
    # Load volume
    volume = load_volume(file_path)
    if volume is None:
        return None
    
    # Print information about the volume
    print(f"Volume information:")
    print(f"  Shape: {volume.shape}")
    print(f"  Min value: {volume.min()}")
    print(f"  Max value: {volume.max()}")
    print(f"  Mean value: {volume.mean()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose visualization mode
    if mode == '3d':
        # Create interactive 3D visualization
        fig = interactive_3d_plot(volume, threshold=threshold, output_dir=output_dir)
    elif mode == 'slices':
        # Create orthogonal slices visualization
        fig = plot_orthogonal_slices(volume, output_dir=output_dir)
    else:
        print(f"Unknown visualization mode: {mode}")
        return None
    
    # Return the figure
    return fig

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D volume from .npy file')
    parser.add_argument('file_path', type=str, help='Path to .npy file')
    parser.add_argument('--output', '-o', type=str, default='visualization_output', 
                        help='Output directory for saved visualizations')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                        help='Initial threshold value for 3D visualization')
    parser.add_argument('--mode', '-m', type=str, choices=['3d', 'slices'], default='3d',
                        help='Visualization mode: 3d or slices')
    
    args = parser.parse_args()
    
    # Visualize the volume
    fig = visualize_npy(args.file_path, args.output, args.threshold, args.mode)
    
    # Show the visualization
    plt.show() 