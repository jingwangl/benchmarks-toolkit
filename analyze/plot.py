"""
Advanced visualization and chart generation for performance benchmarks.
Part of the Autonomous Vehicle Performance Analysis Toolkit.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add config directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "config"))
from config_manager import get_config

# Set professional matplotlib style
plt.style.use('seaborn-v0_8')
mplstyle.use(['seaborn-v0_8', 'seaborn-v0_8-paper'])


class PerformanceVisualizer:
    """Handles creation of professional performance visualizations."""
    
    def __init__(self, config_manager=None):
        """Initialize the performance visualizer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or get_config()
        self.figure_size = tuple(self.config.get_figure_size())
        self.dpi = self.config.get_dpi()
        
        # Professional color palette for autonomous vehicle themes
        self.colors = {
            'primary': '#2E86AB',      # Deep blue
            'secondary': '#A23B72',    # Deep pink
            'accent': '#F18F01',      # Orange
            'success': '#C73E1D',     # Red
            'neutral': '#6C757D',     # Gray
            'light': '#F8F9FA'        # Light gray
        }
        
        # Line styles for different configurations
        self.line_styles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    def setup_plot_style(self) -> None:
        """Configure matplotlib for professional appearance."""
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.2,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.9
        })
    
    def create_cpp_compute_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create visualization for C++ computational benchmarks.
        
        Args:
            df: DataFrame with cpp_compute data
            output_dir: Directory to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Group by optimization level
        optimization_levels = df['config'].unique()
        color_idx = 0
        
        for opt_level in optimization_levels:
            opt_data = df[df['config'] == opt_level]
            
            # Filter for p95 data only
            p95_data = opt_data[opt_data['level_4'] == 'p95']
            
            if p95_data.empty:
                continue
                
            # Sort by thread count for proper line plotting
            p95_data = p95_data.sort_values('threads')
            
            threads = p95_data['threads'].values
            p95_values = p95_data['wall_ms'].values
            
            color = list(self.colors.values())[color_idx % len(self.colors)]
            line_style = self.line_styles[color_idx % len(self.line_styles)]
            marker = self.markers[color_idx % len(self.markers)]
            
            ax.plot(threads, p95_values, 
                   color=color, linestyle=line_style, marker=marker,
                   linewidth=2.5, markersize=8, markerfacecolor=color,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label=f'{opt_level} Optimization')
            
            color_idx += 1
        
        # Customize plot appearance
        ax.set_xlabel('Number of Threads', fontweight='bold')
        ax.set_ylabel('P95 Latency (ms)', fontweight='bold')
        ax.set_title('C++ Computational Performance Analysis\nThread Scalability vs Optimization Level', 
                    fontweight='bold', pad=20)
        
        # Set axis properties
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        
        # Add performance insights as text
        p95_data_all = df[df['level_4'] == 'p95']
        if not p95_data_all.empty:
            min_latency = p95_data_all['wall_ms'].min()
            max_latency = p95_data_all['wall_ms'].max()
        else:
            min_latency = max_latency = 0
        improvement = ((max_latency - min_latency) / max_latency) * 100
        
        ax.text(0.02, 0.98, f'Performance Range: {min_latency:.3f}ms - {max_latency:.3f}ms\n'
                           f'Optimization Impact: {improvement:.1f}% improvement',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = 'cpp_compute_p95.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return filepath
    
    def create_py_io_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create visualization for Python I/O benchmarks.
        
        Args:
            df: DataFrame with py_io data
            output_dir: Directory to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Filter for p95 data only
        p95_data = df[df['level_4'] == 'p95']
        
        if p95_data.empty:
            print("Warning: No p95 data found for py_io benchmark")
            return None
            
        # Sort by block size for proper plotting
        p95_data = p95_data.sort_values('block_kb')
        
        block_sizes = p95_data['block_kb'].values
        p95_values = p95_data['wall_ms'].values
        
        # Create bar plot for I/O performance
        bars = ax.bar(range(len(block_sizes)), p95_values,
                     color=self.colors['primary'], alpha=0.8,
                     edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, p95_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_values)*0.01,
                   f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot appearance
        ax.set_xlabel('Block Size (KB)', fontweight='bold')
        ax.set_ylabel('P95 Latency (ms)', fontweight='bold')
        ax.set_title('Python I/O Performance Analysis\nBlock Size Impact on Latency', 
                    fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(range(len(block_sizes)))
        ax.set_xticklabels([f'{int(size)}KB' for size in block_sizes])
        
        # Set axis properties
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        
        # Add performance insights
        if not p95_data.empty:
            min_latency = p95_data['wall_ms'].min()
            max_latency = p95_data['wall_ms'].max()
        else:
            min_latency = max_latency = 0
        efficiency_ratio = min_latency / max_latency
        
        ax.text(0.02, 0.98, f'Latency Range: {min_latency:.1f}ms - {max_latency:.1f}ms\n'
                           f'Efficiency Ratio: {efficiency_ratio:.2f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = 'py_io_p95.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return filepath
    
    def create_comparison_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create a comparison plot showing both benchmarks.
        
        Args:
            df: Combined DataFrame with all benchmark data
            output_dir: Directory to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # C++ Compute subplot
        cpp_data = df[df['bench'] == 'cpp_compute']
        if not cpp_data.empty:
            cpp_p95_data = cpp_data[cpp_data['level_4'] == 'p95']
            if not cpp_p95_data.empty:
                optimization_levels = cpp_p95_data['config'].unique()
                color_idx = 0
                
                for opt_level in optimization_levels:
                    opt_data = cpp_p95_data[cpp_p95_data['config'] == opt_level].sort_values('threads')
                    color = list(self.colors.values())[color_idx % len(self.colors)]
                    
                    ax1.plot(opt_data['threads'], opt_data['wall_ms'], 
                            color=color, marker='o', linewidth=2, markersize=6,
                            label=f'{opt_level} Optimization')
                    color_idx += 1
            
            ax1.set_xlabel('Threads')
            ax1.set_ylabel('P95 Latency (ms)')
            ax1.set_title('C++ Computational Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Python I/O subplot
        py_data = df[df['bench'] == 'py_io']
        if not py_data.empty:
            py_p95_data = py_data[py_data['level_4'] == 'p95']
            if not py_p95_data.empty:
                py_p95_sorted = py_p95_data.sort_values('block_kb')
                bars = ax2.bar(range(len(py_p95_sorted)), py_p95_sorted['wall_ms'],
                              color=self.colors['secondary'], alpha=0.8)
                
                ax2.set_xlabel('Block Size (KB)')
                ax2.set_ylabel('P95 Latency (ms)')
                ax2.set_title('Python I/O Performance')
                ax2.set_xticks(range(len(py_p95_sorted)))
                ax2.set_xticklabels([f'{int(size)}KB' for size in py_p95_sorted['block_kb']])
                ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Autonomous Vehicle Performance Analysis\nBenchmark Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = 'performance_comparison.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return filepath
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """Generate all performance visualizations.
        
        Args:
            df: DataFrame with performance data
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plot files
        """
        self.setup_plot_style()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        generated_plots = []
        
        # Generate individual benchmark plots
        for bench in df['bench'].unique():
            bench_data = df[df['bench'] == bench]
            
            if bench == 'cpp_compute':
                plot_path = self.create_cpp_compute_plot(bench_data, output_dir)
                generated_plots.append(plot_path)
            elif bench == 'py_io':
                plot_path = self.create_py_io_plot(bench_data, output_dir)
                generated_plots.append(plot_path)
        
        # Generate comparison plot
        comparison_path = self.create_comparison_plot(df, output_dir)
        generated_plots.append(comparison_path)
        
        return generated_plots


def main():
    """Main entry point for the performance visualizer."""
    parser = argparse.ArgumentParser(
        description="Generate professional performance visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot.py metrics.csv -o report/figs
  python plot.py data.csv -o charts --config custom_config.json
        """
    )
    
    parser.add_argument("metrics_csv", help="CSV file with performance metrics")
    parser.add_argument("-o", "--outdir", required=True,
                       help="Output directory for generated plots")
    parser.add_argument("--config", default="config/config.json",
                       help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    config_manager = get_config()
    visualizer = PerformanceVisualizer(config_manager)
    
    try:
        # Load data
        df = pd.read_csv(args.metrics_csv)
        
        if args.verbose:
            print(f"Loaded {len(df)} records from {args.metrics_csv}")
            print(f"Benchmarks found: {df['bench'].unique()}")
        
        # Generate visualizations
        plot_paths = visualizer.generate_visualizations(df, args.outdir)
        
        print(f"Generated {len(plot_paths)} visualization(s):")
        for plot_path in plot_paths:
            print(f"  - {plot_path}")
        
        print(f"\nVisualizations saved to {args.outdir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
