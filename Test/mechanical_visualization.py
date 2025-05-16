import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import json
from skimage import measure
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

def load_ct_parameters(json_path='Test/CT4000.json'):
    """
    Load CT parameters from json file
    
    Parameters:
    -----------
    json_path : str
        Path to the json file containing CT parameters
        
    Returns:
    --------
    ct_params : dict
        Dictionary containing CT parameters
    """
    try:
        with open(json_path, 'r') as f:
            ct_params = json.load(f)
        print(f"Loaded CT parameters from {json_path}")
        return ct_params
    except Exception as e:
        print(f"Error loading CT parameters: {e}")
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

def visualize_mechanical_part(volume, ct_params=None, threshold=0.5, 
                              output_dir='mechanical_visualization', 
                              method='surface', colormap='bone'):
    """
    Visualize a mechanical part with appropriate settings
    
    Parameters:
    -----------
    volume : ndarray
        3D volume data
    ct_params : dict, optional
        CT parameters from json file
    threshold : float
        Threshold value for isosurface extraction
    output_dir : str
        Directory to save visualizations
    method : str
        Visualization method ('surface', 'wireframe', 'points', 'slices')
    colormap : str
        Colormap to use
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize volume
    if volume.max() > 1.0:
        volume_norm = normalize_volume(volume)
    else:
        volume_norm = volume
    
    # Set up figure and 3D axis for interactive visualization
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store the current threshold value
    current_threshold = threshold
    
    # Get voxel spacing if CT parameters are available
    voxel_spacing = [1.0, 1.0, 1.0]  # Default voxel spacing
    if ct_params:
        try:
            # Try to get voxel spacing from CT parameters
            if 'vgParam' in ct_params and 'ResultVoxelPixelSize' in ct_params['vgParam']:
                voxel_spacing = ct_params['vgParam']['ResultVoxelPixelSize']
                if isinstance(voxel_spacing, list) and len(voxel_spacing) == 3:
                    # Convert to relative voxel spacing
                    max_spacing = max(voxel_spacing)
                    voxel_spacing = [s/max_spacing for s in voxel_spacing]
                    print(f"Using relative voxel spacing: {voxel_spacing}")
                else:
                    voxel_spacing = [1.0, 1.0, 1.0]
            # Try to get bounding box
            if 'geometry' in ct_params and 'objectBoundingBox' in ct_params['geometry']:
                box_size = ct_params['geometry']['objectBoundingBox']['sizeXYZ']
                if isinstance(box_size, list) and len(box_size) == 3:
                    print(f"Object bounding box size: {box_size}")
        except Exception as e:
            print(f"Error extracting CT parameters: {e}, using default values")
            voxel_spacing = [1.0, 1.0, 1.0]
    
    # Set up colormap
    cmap = plt.cm.get_cmap(colormap)
    
    # Function to update the visualization
    def update_visualization(threshold_value, render_method):
        ax.clear()
        
        if render_method == 'surface':
            # Use marching cubes to extract isosurface
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    volume_norm, 
                    level=threshold_value,
                    spacing=voxel_spacing,
                    step_size=2  # Step size to reduce complexity
                )
                
                # Plot the surface
                surf = ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces,
                    cmap=colormap,
                    alpha=1.0,
                    shade=True,
                    edgecolor='none'
                )
                
                # Add colorbar
                if hasattr(fig, 'colorbar_ax'):
                    fig.colorbar_ax.remove()
                fig.colorbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
                fig.colorbar(surf, cax=fig.colorbar_ax)
                
            except Exception as e:
                print(f"Error creating isosurface: {e}")
                # Fall back to point cloud
                render_method = 'points'
        
        if render_method == 'wireframe':
            try:
                # Use marching cubes to extract isosurface
                verts, faces, normals, values = measure.marching_cubes(
                    volume_norm, 
                    level=threshold_value,
                    spacing=voxel_spacing,
                    step_size=2
                )
                
                # Plot wireframe by only showing edges
                for i, j, k in faces:
                    ax.plot([verts[i,0], verts[j,0], verts[k,0], verts[i,0]],
                            [verts[i,1], verts[j,1], verts[k,1], verts[i,1]],
                            [verts[i,2], verts[j,2], verts[k,2], verts[i,2]],
                            'k-', alpha=0.2, linewidth=0.2)
            
            except Exception as e:
                print(f"Error creating wireframe: {e}")
                render_method = 'points'
        
        if render_method == 'points':
            # Create a point cloud based on thresholding
            binary_mask = volume_norm > threshold_value
            z_indices, y_indices, x_indices = np.where(binary_mask)
            
            # Get intensity values at these points
            intensities = volume_norm[z_indices, y_indices, x_indices]
            
            # Subsample if necessary for performance
            max_points = 100000
            if len(z_indices) > max_points:
                step = len(z_indices) // max_points
                z_indices = z_indices[::step]
                y_indices = y_indices[::step]
                x_indices = x_indices[::step]
                intensities = intensities[::step]
            
            # Scale indices by voxel spacing
            x_indices = x_indices * voxel_spacing[0]
            y_indices = y_indices * voxel_spacing[1]
            z_indices = z_indices * voxel_spacing[2]
            
            # Create scatter plot
            scatter = ax.scatter(
                x_indices, y_indices, z_indices,
                c=intensities,
                cmap=colormap,
                s=1,
                alpha=0.3,
                marker='.'
            )
            
            # Add colorbar
            if hasattr(fig, 'colorbar_ax'):
                fig.colorbar_ax.remove()
            fig.colorbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
            fig.colorbar(scatter, cax=fig.colorbar_ax)
        
        if render_method == 'slices':
            # Show three orthogonal slices
            z_mid = volume_norm.shape[0] // 2
            y_mid = volume_norm.shape[1] // 2
            x_mid = volume_norm.shape[2] // 2
            
            # Create slices with appropriate scaling
            z_slice = ax.contourf(
                np.arange(0, volume_norm.shape[2]) * voxel_spacing[0],
                np.arange(0, volume_norm.shape[1]) * voxel_spacing[1],
                volume_norm[z_mid],
                zdir='z', offset=z_mid * voxel_spacing[2],
                levels=np.linspace(0, 1, 20),
                cmap=colormap, alpha=0.5
            )
            
            y_slice = ax.contourf(
                np.arange(0, volume_norm.shape[2]) * voxel_spacing[0],
                np.arange(0, volume_norm.shape[0]) * voxel_spacing[2],
                volume_norm[:,y_mid,:].T,
                zdir='y', offset=y_mid * voxel_spacing[1],
                levels=np.linspace(0, 1, 20),
                cmap=colormap, alpha=0.5
            )
            
            x_slice = ax.contourf(
                np.arange(0, volume_norm.shape[1]) * voxel_spacing[1],
                np.arange(0, volume_norm.shape[0]) * voxel_spacing[2],
                volume_norm[:,:,x_mid],
                zdir='x', offset=x_mid * voxel_spacing[0],
                levels=np.linspace(0, 1, 20),
                cmap=colormap, alpha=0.5
            )
            
            # Add colorbar
            if hasattr(fig, 'colorbar_ax'):
                fig.colorbar_ax.remove()
            fig.colorbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
            plt.colorbar(z_slice, cax=fig.colorbar_ax)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        ax.set_title(f'Mechanical Part Visualization ({render_method}, threshold={threshold_value:.2f})')
        
        # Set reasonable axis limits based on volume dimensions
        # and keep aspect ratio equal
        max_dim = max(d*s for d, s in zip(volume_norm.shape, voxel_spacing))
        ax.set_xlim(0, volume_norm.shape[2] * voxel_spacing[0])
        ax.set_ylim(0, volume_norm.shape[1] * voxel_spacing[1])
        ax.set_zlim(0, volume_norm.shape[0] * voxel_spacing[2])
        
        # Update the figure
        fig.canvas.draw_idle()
    
    # Initial visualization
    update_visualization(current_threshold, method)
    
    # Add interactive elements
    
    # Threshold slider
    threshold_ax = plt.axes([0.25, 0.05, 0.55, 0.03])
    threshold_slider = Slider(
        ax=threshold_ax,
        label='Threshold',
        valmin=0.0,
        valmax=1.0,
        valinit=current_threshold,
    )
    
    # Method radio buttons
    method_ax = plt.axes([0.025, 0.7, 0.15, 0.15])
    method_radio = RadioButtons(
        method_ax,
        ('surface', 'wireframe', 'points', 'slices'),
        active=('surface', 'wireframe', 'points', 'slices').index(method),
    )
    
    # View buttons
    view_ax = plt.axes([0.025, 0.5, 0.15, 0.15])
    view_radio = RadioButtons(
        view_ax,
        ('front', 'side', 'top', 'isometric'),
        active=3,  # Default to isometric
    )
    
    # Save button
    save_ax = plt.axes([0.025, 0.2, 0.15, 0.04])
    save_button = Button(save_ax, 'Save Image')
    
    # Reset button
    reset_ax = plt.axes([0.025, 0.15, 0.15, 0.04])
    reset_button = Button(reset_ax, 'Reset View')
    
    # Functions for interactive controls
    def update_threshold(val):
        nonlocal current_threshold
        current_threshold = val
        update_visualization(current_threshold, method_radio.value_selected)
    
    def update_method(val):
        update_visualization(current_threshold, val)
    
    def update_view(val):
        if val == 'front':
            ax.view_init(elev=0, azim=0)
        elif val == 'side':
            ax.view_init(elev=0, azim=90)
        elif val == 'top':
            ax.view_init(elev=90, azim=0)
        elif val == 'isometric':
            ax.view_init(elev=30, azim=30)
        fig.canvas.draw_idle()
    
    def save_figure(event):
        # Generate filename based on current settings
        filename = f"mechanical_part_{method_radio.value_selected}_t{current_threshold:.2f}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filepath}")
    
    def reset_view(event):
        ax.view_init(elev=30, azim=30)
        fig.canvas.draw_idle()
    
    # Connect callbacks
    threshold_slider.on_changed(update_threshold)
    method_radio.on_clicked(update_method)
    view_radio.on_clicked(update_view)
    save_button.on_clicked(save_figure)
    reset_button.on_clicked(reset_view)
    
    # Add text about controls
    info_text = """
    Controls:
    - Drag with mouse to rotate
    - Scroll to zoom
    - Use threshold slider to adjust visibility
    - Choose rendering method and view
    - Save high-res image with save button
    """
    fig.text(0.02, 0.02, info_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    return fig

def visualize_from_file(volume_path, json_path='Test/CT4000.json',
                       output_dir='mechanical_visualization', threshold=0.3,
                       method='surface', colormap='bone'):
    """
    Visualize mechanical part from .npy file with CT parameters
    
    Parameters:
    -----------
    volume_path : str
        Path to .npy file with volume data
    json_path : str
        Path to json file with CT parameters
    output_dir : str
        Directory to save visualizations
    threshold : float
        Initial threshold value
    method : str
        Visualization method
    colormap : str
        Colormap to use
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The visualization figure
    """
    # Load volume
    volume = load_volume(volume_path)
    if volume is None:
        return None
    
    # Load CT parameters if available
    ct_params = load_ct_parameters(json_path)
    
    # Create visualization
    fig = visualize_mechanical_part(
        volume,
        ct_params=ct_params,
        threshold=threshold,
        output_dir=output_dir,
        method=method,
        colormap=colormap
    )
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize mechanical part from .npy file')
    parser.add_argument('volume_path', type=str, help='Path to .npy file with volume data')
    parser.add_argument('--json', '-j', type=str, default='Test/CT4000.json',
                        help='Path to json file with CT parameters')
    parser.add_argument('--output', '-o', type=str, default='mechanical_visualization',
                        help='Directory to save visualizations')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Initial threshold value')
    parser.add_argument('--method', '-m', type=str, default='surface',
                        choices=['surface', 'wireframe', 'points', 'slices'],
                        help='Visualization method')
    parser.add_argument('--colormap', '-c', type=str, default='bone',
                        help='Colormap to use')
    
    args = parser.parse_args()
    
    # Create visualization
    fig = visualize_from_file(
        args.volume_path,
        json_path=args.json,
        output_dir=args.output,
        threshold=args.threshold,
        method=args.method,
        colormap=args.colormap
    )
    
    plt.show() 