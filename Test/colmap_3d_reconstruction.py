#!/usr/bin/env python3
"""
COLMAP-based 3D Reconstruction from 2D TIFF Images

This script implements a complete 3D reconstruction pipeline using COLMAP:
1. Feature extraction from 2D TIFF images
2. Feature matching
3. Structure-from-Motion (SfM)
4. Multi-View Stereo (MVS) dense reconstruction
5. 3D visualization

Requirements:
- pip install pycolmap
- pip install opencv-python
- pip install open3d
- pip install matplotlib
"""

import os
import sys
import shutil
import sqlite3
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False
    print("Warning: pycolmap not available. Install with: pip install pycolmap")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Install with: pip install open3d")


class COLMAP3DReconstructor:
    def __init__(self, workspace_dir="colmap_workspace"):
        """
        Initialize COLMAP 3D Reconstructor
        
        Parameters:
        -----------
        workspace_dir : str
            Directory to store COLMAP workspace and results
        """
        if not COLMAP_AVAILABLE:
            raise ImportError("pycolmap is required. Install with: pip install pycolmap")
        
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.database_path = self.workspace_dir / "database.db"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"
        
        # Create directories
        for dir_path in [self.workspace_dir, self.images_dir, self.sparse_dir, self.dense_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.reconstruction = None
        self.point_cloud = None
        
        print(f"COLMAP 3D Reconstructor initialized")
        print(f"Workspace: {self.workspace_dir}")
        print(f"COLMAP available: {COLMAP_AVAILABLE}")
        print(f"Open3D available: {OPEN3D_AVAILABLE}")
    
    def load_tiff_images(self, input_dir, copy_to_workspace=True):
        """
        Load TIFF images from input directory
        
        Parameters:
        -----------
        input_dir : str
            Directory containing TIFF images
        copy_to_workspace : bool
            Whether to copy images to workspace (recommended for COLMAP)
            
        Returns:
        --------
        list : List of image file paths
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find TIFF files
        tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        tiff_files = []
        for ext in tiff_extensions:
            tiff_files.extend(list(input_path.glob(f"*{ext}")))
        
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {input_dir}")
        
        print(f"Found {len(tiff_files)} TIFF images")
        
        if copy_to_workspace:
            # Copy images to workspace and convert to JPG if needed
            processed_files = []
            failed_files = []
            
            print("Processing images for COLMAP...")
            for i, tiff_file in enumerate(tqdm(tiff_files, desc="Processing images")):
                try:
                    # Try multiple ways to read the image
                    image = None
                    
                    # Method 1: OpenCV
                    image = cv2.imread(str(tiff_file), cv2.IMREAD_COLOR)
                    
                    # Method 2: If OpenCV fails, try with different flags
                    if image is None:
                        image = cv2.imread(str(tiff_file), cv2.IMREAD_UNCHANGED)
                        if image is not None and len(image.shape) == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Method 3: Try PIL if OpenCV fails
                    if image is None:
                        try:
                            from PIL import Image as PILImage
                            pil_image = PILImage.open(tiff_file)
                            # Convert PIL to OpenCV format
                            image = np.array(pil_image)
                            if len(image.shape) == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        except:
                            pass
                    
                    if image is None:
                        failed_files.append(tiff_file.name)
                        continue
                    
                    # Save as JPG in workspace (COLMAP works better with JPG)
                    output_name = f"image_{i:04d}.jpg"
                    output_path = self.images_dir / output_name
                    
                    # Convert to 8-bit if necessary
                    if image.dtype != np.uint8:
                        # Normalize to 0-255 range
                        if image.dtype == np.uint16:
                            # For 16-bit images, scale down
                            image = (image / 256).astype(np.uint8)
                        else:
                            # For other formats, normalize
                            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Ensure image is in correct format for JPEG
                    if len(image.shape) == 2:
                        # Grayscale to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif len(image.shape) == 3 and image.shape[2] == 4:
                        # RGBA to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    
                    success = cv2.imwrite(str(output_path), image)
                    if success:
                        processed_files.append(output_path)
                    else:
                        failed_files.append(tiff_file.name)
                        
                except Exception as e:
                    failed_files.append(f"{tiff_file.name} (error: {str(e)})")
                    continue
            
            print(f"Processed {len(processed_files)} images successfully")
            if failed_files:
                print(f"Failed to process {len(failed_files)} images:")
                for failed in failed_files[:10]:  # Show first 10 failures
                    print(f"  - {failed}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
                
                if len(processed_files) == 0:
                    raise RuntimeError("No images could be processed successfully")
            
            return processed_files
        else:
            return tiff_files
    
    def extract_features(self, camera_model="SIMPLE_RADIAL"):
        """
        Extract features from images using COLMAP
        
        Parameters:
        -----------
        camera_model : str
            Camera model to use (SIMPLE_RADIAL, PINHOLE, OPENCV, etc.)
        """
        print("Extracting features...")
        
        # Create database
        if self.database_path.exists():
            self.database_path.unlink()
        
        # Feature extraction options - updated for newer pycolmap API
        try:
            # Try newer API first
            feature_options = pycolmap.SiftExtractionOptions()
            feature_options.estimate_affine_shape = True
            feature_options.domain_size_pooling = True
            
            # Run feature extraction with newer API
            pycolmap.extract_features(
                database_path=self.database_path,
                image_path=self.images_dir,
                sift_options=feature_options,
                camera_model=camera_model  # Pass camera model directly
            )
            
        except (AttributeError, TypeError) as e:
            print(f"Trying alternative API due to: {e}")
            
            # Try alternative API
            try:
                feature_options = pycolmap.SiftExtractionOptions()
                
                pycolmap.extract_features(
                    database_path=self.database_path,
                    image_path=self.images_dir,
                    sift_options=feature_options
                )
                
            except Exception as e2:
                print(f"Trying basic API due to: {e2}")
                
                # Try most basic API
                pycolmap.extract_features(
                    database_path=str(self.database_path),
                    image_path=str(self.images_dir)
                )
        
        print(f"Features extracted and saved to {self.database_path}")
    
    def match_features(self, matching_method="exhaustive"):
        """
        Match features between images
        
        Parameters:
        -----------
        matching_method : str
            Matching method: 'exhaustive', 'sequential', 'vocab_tree'
        """
        print(f"Matching features using {matching_method} method...")
        
        try:
            # Try newer API first
            match_options = pycolmap.SiftMatchingOptions()
            match_options.guided_matching = True
            
            if matching_method == "exhaustive":
                # Try different function names for exhaustive matching
                try:
                    pycolmap.match_exhaustive_features(
                        database_path=self.database_path,
                        sift_options=match_options
                    )
                except AttributeError:
                    # Try alternative function name
                    pycolmap.exhaustive_matching(
                        database_path=self.database_path,
                        sift_options=match_options
                    )
                    
            elif matching_method == "sequential":
                # Try different function names for sequential matching
                try:
                    pycolmap.match_sequential_features(
                        database_path=self.database_path,
                        sift_options=match_options
                    )
                except AttributeError:
                    # Try alternative function name
                    pycolmap.sequential_matching(
                        database_path=self.database_path,
                        sift_options=match_options
                    )
            else:
                raise ValueError(f"Unsupported matching method: {matching_method}")
                
        except (AttributeError, TypeError) as e:
            print(f"Trying alternative matching API due to: {e}")
            
            # Try simpler API without options
            try:
                if matching_method == "exhaustive":
                    pycolmap.exhaustive_matching(database_path=str(self.database_path))
                elif matching_method == "sequential":
                    pycolmap.sequential_matching(database_path=str(self.database_path))
                    
            except Exception as e2:
                print(f"Trying basic matching API due to: {e2}")
                
                # Try most basic API - check what functions are actually available
                available_functions = [attr for attr in dir(pycolmap) if 'match' in attr.lower()]
                print(f"Available matching functions: {available_functions}")
                
                # Try the most basic approach
                if hasattr(pycolmap, 'match_features'):
                    pycolmap.match_features(database_path=str(self.database_path))
                elif hasattr(pycolmap, 'exhaustive_matching'):
                    pycolmap.exhaustive_matching(database_path=str(self.database_path))
                else:
                    raise RuntimeError(f"No compatible matching function found. Available functions: {available_functions}")
        
        print("Feature matching completed")
    
    def run_sfm(self):
        """
        Run Structure-from-Motion (SfM) reconstruction
        """
        print("Running Structure-from-Motion reconstruction...")
        
        # Create sparse reconstruction directory
        sparse_model_dir = self.sparse_dir / "0"
        sparse_model_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Try newer API first
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_num_matches = 15
            mapper_options.init_min_num_inliers = 100
            mapper_options.init_max_forward_motion = 0.95
            mapper_options.multiple_models = False
            
            # Run incremental mapping
            maps = pycolmap.incremental_mapping(
                database_path=self.database_path,
                image_path=self.images_dir,
                output_path=sparse_model_dir,
                options=mapper_options
            )
            
        except (AttributeError, TypeError) as e:
            print(f"Trying alternative SfM API due to: {e}")
            
            # Try alternative API
            try:
                maps = pycolmap.incremental_mapping(
                    database_path=str(self.database_path),
                    image_path=str(self.images_dir),
                    output_path=str(sparse_model_dir)
                )
                
            except Exception as e2:
                print(f"Trying basic SfM API due to: {e2}")
                
                # Check available functions
                available_functions = [attr for attr in dir(pycolmap) if 'map' in attr.lower() or 'reconstruct' in attr.lower()]
                print(f"Available SfM functions: {available_functions}")
                
                # Try basic mapping
                if hasattr(pycolmap, 'mapper'):
                    maps = pycolmap.mapper(
                        database_path=str(self.database_path),
                        image_path=str(self.images_dir),
                        output_path=str(sparse_model_dir)
                    )
                else:
                    raise RuntimeError(f"No compatible SfM function found. Available functions: {available_functions}")
        
        if not maps:
            raise RuntimeError("SfM reconstruction failed. No models generated.")
        
        print(f"SfM completed. Generated {len(maps)} model(s)")
        
        # Load the reconstruction
        self.reconstruction = pycolmap.Reconstruction(sparse_model_dir)
        
        print(f"Reconstruction summary:")
        print(f"  Images: {len(self.reconstruction.images)}")
        print(f"  Cameras: {len(self.reconstruction.cameras)}")
        print(f"  Points: {len(self.reconstruction.points3D)}")
        
        return self.reconstruction
    
    def run_mvs(self, max_image_size=3200):
        """
        Run Multi-View Stereo (MVS) dense reconstruction
        
        Parameters:
        -----------
        max_image_size : int
            Maximum image size for dense reconstruction
        """
        print("Running Multi-View Stereo dense reconstruction...")
        
        if self.reconstruction is None:
            raise ValueError("Run SfM first before MVS")
        
        # Dense reconstruction options
        mvs_options = pycolmap.MultiViewStereoOptions()
        mvs_options.max_image_size = max_image_size
        mvs_options.patch_match_options.gpu_index = "0"  # Use GPU if available
        mvs_options.patch_match_options.depth_min = 0.01
        mvs_options.patch_match_options.depth_max = 100.0
        
        # Create dense workspace
        dense_workspace = self.dense_dir / "0"
        dense_workspace.mkdir(exist_ok=True, parents=True)
        
        try:
            # Run dense reconstruction
            pycolmap.patch_match_stereo(
                workspace_path=dense_workspace,
                workspace_format="COLMAP",
                pmvs_option_name="option-all"
            )
            
            print("Dense reconstruction completed")
            
            # Try to load dense point cloud
            self._load_dense_point_cloud()
            
        except Exception as e:
            print(f"Dense reconstruction failed: {e}")
            print("Continuing with sparse reconstruction only")
    
    def _load_dense_point_cloud(self):
        """
        Load dense point cloud if available
        """
        # Look for dense point cloud files
        ply_files = list(self.dense_dir.glob("**/*.ply"))
        
        if ply_files and OPEN3D_AVAILABLE:
            try:
                self.point_cloud = o3d.io.read_point_cloud(str(ply_files[0]))
                print(f"Loaded dense point cloud: {len(self.point_cloud.points)} points")
            except Exception as e:
                print(f"Could not load dense point cloud: {e}")
    
    def get_sparse_point_cloud(self):
        """
        Extract sparse point cloud from SfM reconstruction
        
        Returns:
        --------
        tuple : (points, colors) arrays
        """
        if self.reconstruction is None:
            raise ValueError("No reconstruction available. Run SfM first.")
        
        points = []
        colors = []
        
        for point_id, point in self.reconstruction.points3D.items():
            points.append(point.xyz)
            colors.append(point.color / 255.0)  # Normalize to 0-1
        
        return np.array(points), np.array(colors)
    
    def visualize_sparse_3d(self, max_points=10000, save_path=None):
        """
        Visualize sparse 3D point cloud using matplotlib
        
        Parameters:
        -----------
        max_points : int
            Maximum number of points to display
        save_path : str, optional
            Path to save the visualization
        """
        if self.reconstruction is None:
            raise ValueError("No reconstruction available. Run SfM first.")
        
        points, colors = self.get_sparse_point_cloud()
        
        if len(points) == 0:
            print("No 3D points to visualize")
            return None
        
        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            s=1,
            alpha=0.6
        )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Sparse Reconstruction ({len(points)} points)')
        
        # Equal aspect ratio
        max_range = np.max(np.ptp(points, axis=0))
        center = np.mean(points, axis=0)
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D visualization to {save_path}")
        
        plt.show()
        return fig
    
    def visualize_camera_poses(self, save_path=None):
        """
        Visualize camera poses in 3D
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if self.reconstruction is None:
            raise ValueError("No reconstruction available. Run SfM first.")
        
        # Extract camera positions and orientations
        camera_positions = []
        camera_directions = []
        
        for image_id, image in self.reconstruction.images.items():
            if not image.registered:
                continue
            
            # Get camera center in world coordinates
            R = image.rotation_matrix()
            t = image.translation
            center = -R.T @ t
            camera_positions.append(center)
            
            # Get camera direction (negative Z axis in camera coordinates)
            direction = R.T @ np.array([0, 0, -1])
            camera_directions.append(direction)
        
        camera_positions = np.array(camera_positions)
        camera_directions = np.array(camera_directions)
        
        # Get sparse points for context
        points, colors = self.get_sparse_point_cloud()
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sparse points (subsampled)
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            points_sub = points[indices]
            colors_sub = colors[indices]
        else:
            points_sub = points
            colors_sub = colors
        
        ax.scatter(
            points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
            c=colors_sub,
            s=0.5,
            alpha=0.3,
            label='3D Points'
        )
        
        # Plot camera positions
        ax.scatter(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            c='red',
            s=50,
            marker='^',
            label='Cameras'
        )
        
        # Plot camera directions
        scale = np.mean(np.ptp(points, axis=0)) * 0.05
        ax.quiver(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            camera_directions[:, 0] * scale,
            camera_directions[:, 1] * scale,
            camera_directions[:, 2] * scale,
            color='blue',
            alpha=0.7,
            label='Camera Direction'
        )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Reconstruction with Camera Poses ({len(camera_positions)} cameras)')
        ax.legend()
        
        # Equal aspect ratio
        all_points = np.vstack([points, camera_positions])
        max_range = np.max(np.ptp(all_points, axis=0))
        center = np.mean(all_points, axis=0)
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved camera pose visualization to {save_path}")
        
        plt.show()
        return fig
    
    def visualize_with_open3d(self):
        """
        Visualize 3D reconstruction using Open3D (interactive)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Install with: pip install open3d")
            return
        
        if self.reconstruction is None:
            raise ValueError("No reconstruction available. Run SfM first.")
        
        # Create Open3D point cloud from sparse reconstruction
        points, colors = self.get_sparse_point_cloud()
        
        if len(points) == 0:
            print("No 3D points to visualize")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Add camera poses
        geometries = [pcd]
        
        # Create camera coordinate frames
        for image_id, image in self.reconstruction.images.items():
            if not image.registered:
                continue
            
            # Get camera pose
            R = image.rotation_matrix()
            t = image.translation
            center = -R.T @ t
            
            # Create coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # Transform frame to camera pose
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = center
            frame.transform(T)
            
            geometries.append(frame)
        
        # Visualize
        print("Opening Open3D viewer...")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Ctrl+Mouse: Pan")
        print("- Scroll: Zoom")
        print("- Press 'h' for help")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="COLMAP 3D Reconstruction",
            width=1200,
            height=800
        )
    
    def export_point_cloud(self, output_path):
        """
        Export point cloud to file
        
        Parameters:
        -----------
        output_path : str
            Output file path (.ply, .pcd, etc.)
        """
        if self.reconstruction is None:
            raise ValueError("No reconstruction available. Run SfM first.")
        
        points, colors = self.get_sparse_point_cloud()
        
        if len(points) == 0:
            print("No 3D points to export")
            return
        
        if OPEN3D_AVAILABLE:
            # Use Open3D for export
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Point cloud exported to {output_path}")
        else:
            # Manual PLY export
            if not output_path.endswith('.ply'):
                output_path += '.ply'
            
            with open(output_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for point, color in zip(points, colors):
                    r, g, b = (color * 255).astype(int)
                    f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
            
            print(f"Point cloud exported to {output_path}")
    
    def run_full_reconstruction(self, input_dir, camera_model="SIMPLE_RADIAL", 
                              matching_method="exhaustive", run_dense=False):
        """
        Run complete 3D reconstruction pipeline
        
        Parameters:
        -----------
        input_dir : str
            Directory containing TIFF images
        camera_model : str
            Camera model to use
        matching_method : str
            Feature matching method
        run_dense : bool
            Whether to run dense reconstruction (slower)
            
        Returns:
        --------
        pycolmap.Reconstruction : The reconstruction result
        """
        try:
            print("="*60)
            print("COLMAP 3D RECONSTRUCTION PIPELINE")
            print("="*60)
            
            # Step 1: Load images
            print("\n1. Loading and processing images...")
            self.load_tiff_images(input_dir)
            
            # Step 2: Extract features
            print("\n2. Extracting features...")
            self.extract_features(camera_model)
            
            # Step 3: Match features
            print("\n3. Matching features...")
            self.match_features(matching_method)
            
            # Step 4: Run SfM
            print("\n4. Running Structure-from-Motion...")
            reconstruction = self.run_sfm()
            
            # Step 5: Dense reconstruction (optional)
            if run_dense:
                print("\n5. Running dense reconstruction...")
                self.run_mvs()
            
            print("\n" + "="*60)
            print("RECONSTRUCTION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return reconstruction
            
        except Exception as e:
            print(f"\nReconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Example usage of COLMAP 3D reconstruction
    """
    print("COLMAP 3D Reconstruction Example")
    print("="*40)
    
    # Check dependencies
    if not COLMAP_AVAILABLE:
        print("Error: pycolmap not available. Install with:")
        print("pip install pycolmap")
        return
    
    # Example usage
    workspace = "colmap_output"
    reconstructor = COLMAP3DReconstructor(workspace)
    
    print("\nTo use this script:")
    print("1. Place your TIFF images in a directory")
    print("2. Run the reconstruction:")
    print("")
    print("   reconstructor = COLMAP3DReconstructor('workspace')")
    print("   reconstruction = reconstructor.run_full_reconstruction('path/to/tiff/images')")
    print("   reconstructor.visualize_sparse_3d()")
    print("   reconstructor.visualize_camera_poses()")
    print("   reconstructor.visualize_with_open3d()  # Interactive viewer")
    print("   reconstructor.export_point_cloud('result.ply')")
    print("")
    print("Requirements:")
    print("- Images should have sufficient overlap")
    print("- Images should be taken from different viewpoints")
    print("- Good lighting and texture for feature detection")


if __name__ == "__main__":
    main() 