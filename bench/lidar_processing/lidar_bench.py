"""
Autonomous Vehicle Sensor Data Processing Benchmark
Simulates real-time processing of LiDAR point cloud data
"""

import numpy as np
import time
import argparse
import os
from typing import List, Tuple
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class LiDARProcessor:
    """Simulates LiDAR point cloud data processing for autonomous vehicles."""
    
    def __init__(self, num_points: int = 100000, use_parallel: bool = True):
        """Initialize LiDAR processor.
        
        Args:
            num_points: Number of points in the point cloud
            use_parallel: Whether to use parallel processing
        """
        self.num_points = num_points
        self.points = None
        self.processed_points = None
        self.use_parallel = use_parallel
        self.num_workers = min(cpu_count(), 8) if use_parallel else 1
    
    def generate_point_cloud(self) -> np.ndarray:
        """Generate synthetic LiDAR point cloud data using vectorized operations.
        
        Returns:
            3D point cloud array (N x 3)
        """
        # Generate points in a realistic LiDAR pattern using vectorized operations
        # Simulate 360-degree horizontal scan with vertical layers
        n_h = int(np.sqrt(self.num_points))
        n_v = int(np.sqrt(self.num_points))
        
        # Create angle grids
        angles_h = np.linspace(0, 2 * np.pi, n_h)
        angles_v = np.linspace(-np.pi/6, np.pi/6, n_v)
        
        # Create meshgrids for vectorized computation
        h_grid, v_grid = np.meshgrid(angles_h, angles_v)
        
        # Generate distances with noise
        distances = np.random.uniform(1.0, 50.0, h_grid.shape)
        
        # Vectorized coordinate calculation
        x = distances * np.cos(v_grid) * np.cos(h_grid)
        y = distances * np.cos(v_grid) * np.sin(h_grid)
        z = distances * np.sin(v_grid)
        
        # Flatten and combine coordinates
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Add random points if needed
        if len(points) < self.num_points:
            remaining = self.num_points - len(points)
            random_points = np.random.uniform(
                [-50, -50, -5], [50, 50, 5], 
                size=(remaining, 3)
            )
            points = np.vstack([points, random_points])
        
        # Take exactly the requested number of points
        self.points = points[:self.num_points]
        return self.points
    
    def filter_ground_points(self, points: np.ndarray, ground_threshold: float = -1.5) -> np.ndarray:
        """Filter out ground points (simple ground removal).
        
        Args:
            points: Input point cloud
            ground_threshold: Z-coordinate threshold for ground points
            
        Returns:
            Filtered point cloud without ground points
        """
        return points[points[:, 2] > ground_threshold]
    
    def cluster_points(self, points: np.ndarray, cluster_radius: float = 0.5, min_samples: int = 5) -> List[List[int]]:
        """Efficient DBSCAN clustering algorithm for object detection.
        
        Args:
            points: Input point cloud
            cluster_radius: Maximum distance for clustering (eps parameter)
            min_samples: Minimum number of points to form a cluster
            
        Returns:
            List of clusters, each containing point indices
        """
        if len(points) == 0:
            return []
        
        # Use DBSCAN for efficient clustering
        clustering = DBSCAN(eps=cluster_radius, min_samples=min_samples, n_jobs=-1).fit(points)
        
        # Convert DBSCAN labels to cluster lists
        clusters = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            cluster_indices = np.where(clustering.labels_ == label)[0].tolist()
            if len(cluster_indices) >= min_samples:
                clusters.append(cluster_indices)
        
        return clusters
    
    def calculate_bounding_boxes(self, points: np.ndarray, clusters: List[List[int]]) -> List[Tuple[float, float, float, float, float, float]]:
        """Calculate bounding boxes for detected objects using parallel processing.
        
        Args:
            points: Point cloud data
            clusters: List of point clusters
            
        Returns:
            List of bounding boxes (min_x, max_x, min_y, max_y, min_z, max_z)
        """
        if not clusters:
            return []
        
        def calculate_single_bbox(cluster_indices):
            cluster_points = points[cluster_indices]
            return (
                np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]),
                np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1]),
                np.min(cluster_points[:, 2]), np.max(cluster_points[:, 2])
            )
        
        if self.use_parallel and len(clusters) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                bounding_boxes = list(executor.map(calculate_single_bbox, clusters))
        else:
            bounding_boxes = [calculate_single_bbox(cluster) for cluster in clusters]
        
        return bounding_boxes
    
    def process_point_cloud(self) -> dict:
        """Process the entire point cloud pipeline.
        
        Returns:
            Dictionary with processing results and timing
        """
        start_time = time.perf_counter()
        
        # Generate point cloud
        points = self.generate_point_cloud()
        generation_time = time.perf_counter()
        
        # Filter ground points
        filtered_points = self.filter_ground_points(points)
        filtering_time = time.perf_counter()
        
        # Cluster points
        clusters = self.cluster_points(filtered_points)
        clustering_time = time.perf_counter()
        
        # Calculate bounding boxes
        bounding_boxes = self.calculate_bounding_boxes(filtered_points, clusters)
        bbox_time = time.perf_counter()
        
        total_time = bbox_time - start_time
        
        return {
            'total_time_ms': total_time * 1000,
            'generation_time_ms': (generation_time - start_time) * 1000,
            'filtering_time_ms': (filtering_time - generation_time) * 1000,
            'clustering_time_ms': (clustering_time - filtering_time) * 1000,
            'bbox_time_ms': (bbox_time - clustering_time) * 1000,
            'num_points': len(points),
            'filtered_points': len(filtered_points),
            'num_clusters': len(clusters),
            'num_objects': len(bounding_boxes)
        }


def run_benchmark(num_points: int, iterations: int, use_parallel: bool = True) -> List[dict]:
    """Run LiDAR processing benchmark.
    
    Args:
        num_points: Number of points in point cloud
        iterations: Number of iterations to run
        use_parallel: Whether to use parallel processing
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for i in range(iterations):
        processor = LiDARProcessor(num_points, use_parallel)
        result = processor.process_point_cloud()
        results.append(result)
    
    return results


def main():
    """Main entry point for LiDAR benchmark."""
    parser = argparse.ArgumentParser(description="LiDAR Point Cloud Processing Benchmark")
    parser.add_argument("--points", type=int, default=100000,
                       help="Number of points in point cloud")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations to run")
    parser.add_argument("--output", default="lidar_results.csv",
                       help="Output CSV file")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output, only show CSV")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Use parallel processing (default: True)")
    
    args = parser.parse_args()
    
    print(f"Running LiDAR benchmark with {args.points} points, {args.iterations} iterations...")
    
    # Run benchmark
    results = run_benchmark(args.points, args.iterations, args.parallel)
    
    # Calculate statistics
    total_times = [r['total_time_ms'] for r in results]
    generation_times = [r['generation_time_ms'] for r in results]
    filtering_times = [r['filtering_time_ms'] for r in results]
    clustering_times = [r['clustering_time_ms'] for r in results]
    bbox_times = [r['bbox_time_ms'] for r in results]
    
    if not args.quiet:
        # Print results
        print(f"\nBenchmark Results:")
        print(f"Total Processing Time: {np.mean(total_times):.3f}ms ± {np.std(total_times):.3f}ms")
        print(f"Point Generation: {np.mean(generation_times):.3f}ms ± {np.std(generation_times):.3f}ms")
        print(f"Ground Filtering: {np.mean(filtering_times):.3f}ms ± {np.std(filtering_times):.3f}ms")
        print(f"Clustering: {np.mean(clustering_times):.3f}ms ± {np.std(clustering_times):.3f}ms")
        print(f"Bounding Boxes: {np.mean(bbox_times):.3f}ms ± {np.std(bbox_times):.3f}ms")
        
        # Save detailed results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
        
        # Print performance metrics in CSV format for integration
        print(f"\nCSV Output:")
        print(f"bench,points,wall_ms,iterations")
        for result in results:
            print(f"lidar_processing,{args.points},{result['total_time_ms']:.3f},{args.iterations}")
    else:
        # Quiet mode: only output CSV data (no header)
        for result in results:
            print(f"lidar_processing,{args.points},{result['total_time_ms']:.3f},{args.iterations}")


if __name__ == "__main__":
    main()
