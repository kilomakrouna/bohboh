"""
Utility functions for tomographic reconstruction.
"""
import os
import numpy as np
import tifffile
import vtk
from vtk.util import numpy_support
import re
import json
import configparser

def ensure_directory(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_tiff_stack(directory, pattern="*.tif*"):
    """
    Load a stack of TIFF images from a directory.
    
    Args:
        directory (str): Directory containing TIFF images.
        pattern (str): Glob pattern for selecting TIFF files.
        
    Returns:
        tuple: (projections, filenames)
            projections: 3D array containing all projections.
            filenames: List of filenames in order they were loaded.
    """
    import glob
    
    # Get sorted list of TIFF files
    tiff_files = sorted(glob.glob(os.path.join(directory, pattern)))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {directory} with pattern {pattern}")
    
    # Load first image to get dimensions
    img = tifffile.imread(tiff_files[0])
    
    # Allocate array for all projections
    projections = np.zeros((len(tiff_files), img.shape[0], img.shape[1]), dtype=img.dtype)
    
    # Load all projections
    for i, tiff_file in enumerate(tiff_files):
        projections[i] = tifffile.imread(tiff_file)
    
    return projections, tiff_files

def read_metadata_file(filepath):
    """
    Read metadata from a text file.
    
    Args:
        filepath (str): Path to metadata file.
        
    Returns:
        dict: Metadata dictionary.
    """
    metadata = {}
    
    # Check if file exists
    if not os.path.exists(filepath):
        return metadata
    
    # Try to determine file format and parse accordingly
    with open(filepath, 'r') as f:
        content = f.read().strip()
        
        # Check if it's an INI-style file with sections like [GENERAL], [GEOMETRY]
        if re.search(r'^\[(.*?)\]', content, re.MULTILINE):
            try:
                config = configparser.ConfigParser()
                config.read(filepath)
                
                # Convert ConfigParser object to dictionary
                for section in config.sections():
                    metadata[section] = {}
                    for key, value in config[section].items():
                        # Try to convert value to numeric if possible
                        try:
                            if '.' in value:
                                metadata[section][key] = float(value)
                            elif value.isdigit():
                                metadata[section][key] = int(value)
                            else:
                                metadata[section][key] = value
                        except ValueError:
                            metadata[section][key] = value
                
                # Extract key geometry information for easier access
                if 'GEOMETRY' in metadata:
                    geom = metadata['GEOMETRY']
                    metadata['geometry'] = {}
                    
                    # Get the distance values
                    if 'distancesourcedetector' in {k.lower(): v for k, v in geom.items()}:
                        source_detector_dist = next(v for k, v in geom.items() 
                                                  if k.lower() == 'distancesourcedetector')
                        metadata['source_detector_distance'] = source_detector_dist
                        
                    if 'distancesourceorigin' in {k.lower(): v for k, v in geom.items()}:
                        source_origin_dist = next(v for k, v in geom.items() 
                                                if k.lower() == 'distancesourceorigin')
                        metadata['source_object_distance'] = source_origin_dist
                        metadata['geometry']['source_origin_dist'] = source_origin_dist
                        
                        # Calculate origin to detector distance
                        if 'source_detector_distance' in metadata:
                            origin_detector_dist = metadata['source_detector_distance'] - source_origin_dist
                            metadata['geometry']['origin_detector_dist'] = origin_detector_dist
                
                # Extract acquisition angles
                if 'ACQUISITION' in metadata:
                    acq = metadata['ACQUISITION']
                    if all(k in {k.lower(): v for k, v in acq.items()} for k in 
                           ['anglefirst', 'angleinterval', 'anglelast']):
                        first = next(v for k, v in acq.items() if k.lower() == 'anglefirst')
                        interval = next(v for k, v in acq.items() if k.lower() == 'angleinterval')
                        last = next(v for k, v in acq.items() if k.lower() == 'anglelast')
                        
                        # Generate angles array
                        metadata['angles'] = np.arange(first, last + interval, interval)
                
                # Extract pixel size
                if 'DETECTOR' in metadata:
                    det = metadata['DETECTOR']
                    if 'pixelsize' in {k.lower(): v for k, v in det.items()}:
                        pixel_size = next(v for k, v in det.items() if k.lower() == 'pixelsize')
                        metadata['pixel_size'] = pixel_size
                
                return metadata
            except Exception as e:
                print(f"Error parsing INI file: {e}")
                # Continue to try other formats
                pass
        
        # Try JSON format
        try:
            metadata = json.loads(content)
            return metadata
        except json.JSONDecodeError:
            pass
        
        # Try key-value pairs format (e.g., key=value)
        try:
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert value to numeric if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                        
                    metadata[key] = value
            
            if metadata:  # If we found some key-value pairs
                return metadata
        except Exception:
            pass
            
        # Try to extract values with regular expressions
        try:
            # Look for common patterns in metadata files
            angle_match = re.search(r'angle[s]?\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
            if angle_match:
                metadata['angle'] = float(angle_match.group(1))
                
            exposure_match = re.search(r'exposure[_\s]time[s]?\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
            if exposure_match:
                metadata['exposure_time'] = float(exposure_match.group(1))
                
            src_det_match = re.search(r'source[_\s]detector[_\s]distance\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
            if src_det_match:
                metadata['source_detector_distance'] = float(src_det_match.group(1))
                
            src_obj_match = re.search(r'source[_\s]object[_\s]distance\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
            if src_obj_match:
                metadata['source_object_distance'] = float(src_obj_match.group(1))
        except Exception:
            pass
    
    return metadata

def find_metadata_for_tiff(tiff_filepath):
    """
    Find and read metadata file associated with a TIFF file.
    
    Args:
        tiff_filepath (str): Path to TIFF file.
        
    Returns:
        dict: Metadata dictionary.
    """
    base_path = os.path.splitext(tiff_filepath)[0]
    
    # Try common metadata file extensions
    for ext in ['.txt', '.meta', '.metadata', '.json', '.ini']:
        metadata_path = base_path + ext
        if os.path.exists(metadata_path):
            return read_metadata_file(metadata_path)
    
    # Try directory-level metadata file
    dir_path = os.path.dirname(tiff_filepath)
    for filename in ['metadata.txt', 'metadata.json', 'acquisition.txt', 'parameters.txt']:
        metadata_path = os.path.join(dir_path, filename)
        if os.path.exists(metadata_path):
            return read_metadata_file(metadata_path)
    
    # No metadata file found
    return {}

def extract_angles_from_metadata(tiff_files):
    """
    Extract rotation angles from metadata files associated with TIFF files.
    
    Args:
        tiff_files (list): List of TIFF file paths.
        
    Returns:
        np.ndarray: Array of angles in degrees.
    """
    angles = []
    
    # First check if directory-level metadata has angles information
    if tiff_files:
        dir_path = os.path.dirname(tiff_files[0])
        dir_metadata_files = [
            os.path.join(dir_path, 'metadata.txt'),
            os.path.join(dir_path, 'metadata.ini')
        ]
        
        for metadata_file in dir_metadata_files:
            if os.path.exists(metadata_file):
                metadata = read_metadata_file(metadata_file)
                if 'angles' in metadata:
                    return metadata['angles']
                elif 'ACQUISITION' in metadata:
                    acq = metadata['ACQUISITION']
                    if all(k in {k.lower(): v for k, v in acq.items()} for k in 
                           ['anglefirst', 'angleinterval', 'anglelast', 'numberimages']):
                        first = next(v for k, v in acq.items() if k.lower() == 'anglefirst')
                        interval = next(v for k, v in acq.items() if k.lower() == 'angleinterval')
                        last = next(v for k, v in acq.items() if k.lower() == 'anglelast')
                        n_angles = next(v for k, v in acq.items() if k.lower() == 'numberimages')
                        
                        # Generate angles array
                        if isinstance(n_angles, int) and n_angles > 0:
                            return np.linspace(first, last, n_angles)
    
    # If no directory-level metadata with angles, check individual files
    for tiff_file in tiff_files:
        metadata = find_metadata_for_tiff(tiff_file)
        
        if 'angle' in metadata:
            angles.append(metadata['angle'])
        else:
            angles.append(None)
    
    # If any angle is None, use evenly spaced angles instead
    if None in angles:
        print("Warning: Could not extract angles from all metadata files. Using evenly spaced angles.")
        angles = np.linspace(0, 360, len(tiff_files), endpoint=False)
    else:
        angles = np.array(angles)
        
    return angles

def extract_angles_from_filenames(filenames, pattern='_(\d+)deg'):
    """
    Extract rotation angles from filenames using regex pattern.
    
    Args:
        filenames (list): List of filenames.
        pattern (str): Regex pattern to extract angle.
        
    Returns:
        np.ndarray: Array of angles in degrees.
    """
    import re
    
    angles = []
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            angles.append(float(match.group(1)))
        else:
            # If no angle found, use filename index as angle
            angles.append(None)
    
    # If any angle is None, use evenly spaced angles instead
    if None in angles:
        print("Warning: Could not extract angles from all filenames. Using evenly spaced angles.")
        angles = np.linspace(0, 360, len(filenames), endpoint=False)
    else:
        angles = np.array(angles)
        
    return angles

def load_projections_with_metadata(directory, pattern="*.tif*", angle_pattern='_(\d+)deg'):
    """
    Load projections and associated metadata.
    
    Args:
        directory (str): Directory containing projection images.
        pattern (str): Glob pattern for selecting TIFF files.
        angle_pattern (str): Regex pattern to extract angle from filenames.
        
    Returns:
        tuple: (projections, angles, metadata)
            projections: 3D array containing all projections.
            angles: Array of projection angles in degrees.
            metadata: Dictionary of additional metadata.
    """
    # Load projection images
    projections, filenames = load_tiff_stack(directory, pattern)
    
    # Check for directory-level metadata first
    metadata = {}
    for metadata_filename in ['metadata.txt', 'metadata.ini', 'acquisition.txt', 'parameters.txt']:
        metadata_path = os.path.join(directory, metadata_filename)
        if os.path.exists(metadata_path):
            metadata = read_metadata_file(metadata_path)
            print(f"Found metadata file: {metadata_path}")
            break
    
    # Try to get angles from metadata first
    angles = extract_angles_from_metadata(filenames)
    
    # If all angles are None, try to extract from filenames
    if np.all(angles == None):
        angles = extract_angles_from_filenames(filenames, angle_pattern)
    
    # Get geometry information from metadata if available
    if 'geometry' not in metadata and 'source_detector_distance' in metadata and 'source_object_distance' in metadata:
        source_origin_dist = metadata.get('source_object_distance')
        origin_detector_dist = metadata.get('source_detector_distance') - source_origin_dist
        
        metadata['geometry'] = {
            'source_origin_dist': source_origin_dist,
            'origin_detector_dist': origin_detector_dist
        }
    
    return projections, angles, metadata

def create_projection_geometry(angles, detector_shape, source_origin_dist, origin_detector_dist=None):
    """
    Create a dictionary containing the projection geometry.
    
    Args:
        angles (np.ndarray): Array of projection angles in degrees.
        detector_shape (tuple): Shape of detector (height, width).
        source_origin_dist (float): Distance from source to rotation center.
        origin_detector_dist (float, optional): Distance from rotation center to detector.
            If None, equal to source_origin_dist (symmetric).
            
    Returns:
        dict: Projection geometry parameters.
    """
    if origin_detector_dist is None:
        origin_detector_dist = source_origin_dist
        
    return {
        'angles': angles,
        'detector_shape': detector_shape,
        'source_origin_dist': source_origin_dist,
        'origin_detector_dist': origin_detector_dist,
        'total_dist': source_origin_dist + origin_detector_dist
    }

def save_numpy_as_vtk(volume, filename, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Save a 3D NumPy array as a VTK file for 3D visualization.
    
    Args:
        volume (np.ndarray): 3D volume data.
        filename (str): Output filename (.vti extension recommended).
        spacing (tuple): Voxel spacing in (x, y, z).
        origin (tuple): Volume origin coordinates.
    """
    # Ensure the data is in float32 format
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)
    
    # Create VTK image data
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
    vtk_data.SetSpacing(spacing)
    vtk_data.SetOrigin(origin)
    
    # Convert NumPy array to VTK array
    flat_data = volume.ravel(order='F')
    vtk_array = numpy_support.numpy_to_vtk(flat_data)
    
    # Add array to image data
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    # Write VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    writer.Write()
    
def save_volume_as_tiff_stack(volume, output_dir, base_filename="slice"):
    """
    Save a 3D volume as a stack of TIFF images.
    
    Args:
        volume (np.ndarray): 3D volume data.
        output_dir (str): Output directory for TIFF stack.
        base_filename (str): Base filename for each slice.
    """
    ensure_directory(output_dir)
    
    # Normalize volume to 0-65535 for 16-bit TIFF
    if volume.dtype != np.uint16:
        v_min, v_max = volume.min(), volume.max()
        if v_min != v_max:  # Avoid division by zero
            volume_norm = ((volume - v_min) / (v_max - v_min) * 65535).astype(np.uint16)
        else:
            volume_norm = np.zeros_like(volume, dtype=np.uint16)
    else:
        volume_norm = volume
    
    # Save each slice as a TIFF file
    for i in range(volume.shape[0]):
        filename = os.path.join(output_dir, f"{base_filename}_{i:04d}.tiff")
        tifffile.imwrite(filename, volume_norm[i])
