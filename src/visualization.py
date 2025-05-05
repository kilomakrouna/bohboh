"""
Visualization utilities for tomographic reconstruction.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

def show_projections(projections, indices=None, cmap='gray', figsize=(15, 10)):
    """
    Display a subset of projection images.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        indices (list, optional): Indices of projections to display.
        cmap (str): Colormap to use.
        figsize (tuple): Figure size.
    """
    if indices is None:
        # Select a few evenly spaced projections
        n_projections = len(projections)
        n_display = min(4, n_projections)
        indices = np.linspace(0, n_projections-1, n_display, dtype=int)
    
    n_display = len(indices)
    fig, axes = plt.subplots(1, n_display, figsize=figsize)
    
    if n_display == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        im = axes[i].imshow(projections[idx], cmap=cmap)
        axes[i].set_title(f"Projection {idx}")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def show_sinogram(projections, row_idx=None, cmap='gray', figsize=(10, 6)):
    """
    Display a sinogram from the projection data.
    
    Args:
        projections (np.ndarray): Stack of projection images.
        row_idx (int, optional): Row index to display. Defaults to middle row.
        cmap (str): Colormap to use.
        figsize (tuple): Figure size.
    """
    if row_idx is None:
        row_idx = projections.shape[1] // 2
    
    sinogram = projections[:, row_idx, :]
    
    plt.figure(figsize=figsize)
    plt.imshow(sinogram, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(f"Sinogram at row {row_idx}")
    plt.xlabel("Detector Column")
    plt.ylabel("Projection Number")
    plt.show()

def show_volume_slices(volume, indices=None, axis=0, cmap='gray', figsize=(15, 10)):
    """
    Display slices from a 3D volume.
    
    Args:
        volume (np.ndarray): 3D volume data.
        indices (list, optional): Indices of slices to display.
        axis (int): Axis along which to take slices (0, 1, or 2).
        cmap (str): Colormap to use.
        figsize (tuple): Figure size.
    """
    if indices is None:
        # Select a few evenly spaced slices
        n_slices = volume.shape[axis]
        n_display = min(4, n_slices)
        indices = np.linspace(0, n_slices-1, n_display, dtype=int)
    
    n_display = len(indices)
    fig, axes = plt.subplots(1, n_display, figsize=figsize)
    
    if n_display == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        if axis == 0:
            slice_data = volume[idx, :, :]
        elif axis == 1:
            slice_data = volume[:, idx, :]
        else:  # axis == 2
            slice_data = volume[:, :, idx]
        
        im = axes[i].imshow(slice_data, cmap=cmap)
        axes[i].set_title(f"Slice {idx} along axis {axis}")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def show_volume_3slice(volume, cmap='gray', figsize=(15, 5)):
    """
    Display three orthogonal slices from a 3D volume.
    
    Args:
        volume (np.ndarray): 3D volume data.
        cmap (str): Colormap to use.
        figsize (tuple): Figure size.
    """
    # Get middle slices
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # XY plane (Z fixed)
    im0 = axes[0].imshow(volume[:, :, z_mid], cmap=cmap)
    axes[0].set_title(f"XY Plane (Z={z_mid})")
    fig.colorbar(im0, ax=axes[0])
    
    # XZ plane (Y fixed)
    im1 = axes[1].imshow(volume[:, y_mid, :], cmap=cmap)
    axes[1].set_title(f"XZ Plane (Y={y_mid})")
    fig.colorbar(im1, ax=axes[1])
    
    # YZ plane (X fixed)
    im2 = axes[2].imshow(volume[x_mid, :, :], cmap=cmap)
    axes[2].set_title(f"YZ Plane (X={x_mid})")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def volume_histogram(volume, bins=256, figsize=(10, 6)):
    """
    Display histogram of voxel values in the volume.
    
    Args:
        volume (np.ndarray): 3D volume data.
        bins (int): Number of histogram bins.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.hist(volume.flatten(), bins=bins, alpha=0.7)
    plt.title("Volume Histogram")
    plt.xlabel("Voxel Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

def render_surface_plotly(volume, threshold=None, opacity=0.5, colorscale='Viridis'):
    """
    Render an isosurface of the volume using Plotly.
    
    Args:
        volume (np.ndarray): 3D volume data.
        threshold (float, optional): Isovalue for surface. Defaults to 0.5 * max.
        opacity (float): Surface opacity.
        colorscale (str): Colorscale for the surface.
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure object.
    """
    if threshold is None:
        threshold = 0.5 * volume.max()
    
    # Create coordinate arrays
    x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    # Create isosurface
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=volume.max(),
        opacity=opacity,
        surface_count=1,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return fig

def render_volume_pyvista(volume, threshold=None, opacity=0.5):
    """
    Render volume using PyVista (if available).
    
    Args:
        volume (np.ndarray): 3D volume data.
        threshold (float, optional): Isovalue for surface. Defaults to 0.5 * max.
        opacity (float): Surface opacity.
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Install with 'pip install pyvista'")
        return
    
    if threshold is None:
        threshold = 0.5 * volume.max()
    
    # Create uniform grid
    grid = pv.UniformGrid()
    
    # Set dimensions (number of points in each direction)
    grid.dimensions = np.array(volume.shape) + 1
    
    # Edit the points - center data
    grid.origin = (-volume.shape[0]/2, -volume.shape[1]/2, -volume.shape[2]/2)
    grid.spacing = (1, 1, 1)
    
    # Add data to the grid
    grid.point_data["values"] = volume.flatten(order="F")
    
    # Create isosurface
    surface = grid.contour([threshold])
    
    # Create a plotter
    p = pv.Plotter()
    p.add_mesh(surface, opacity=opacity)
    p.show_axes()
    
    # Show the plot
    p.show()
