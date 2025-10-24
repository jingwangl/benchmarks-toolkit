"""
Professional report generation for performance analysis results.
Part of the Autonomous Vehicle Performance Analysis Toolkit.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add config directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "config"))
from config_manager import get_config


class ReportGenerator:
    """Generates professional performance analysis reports."""
    
    def __init__(self, config_manager=None):
        """Initialize the report generator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or get_config()
        self.project_name = self.config.get("project.name", "Performance Analysis")
        self.project_version = self.config.get("project.version", "1.0.0")
    
    def load_environment_info(self, env_file: str) -> Dict[str, str]:
        """Load environment information from file.
        
        Args:
            env_file: Path to environment information file
            
        Returns:
            Dictionary with environment information
        """
        env_info = {}
        
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    env_info['raw_content'] = content
            except Exception as e:
                print(f"Warning: Could not read environment file {env_file}: {e}")
        
        return env_info
    
    def load_performance_data(self, metrics_file: str) -> Optional[pd.DataFrame]:
        """Load performance metrics data.
        
        Args:
            metrics_file: Path to metrics CSV file
            
        Returns:
            DataFrame with performance data or None if file doesn't exist
        """
        if os.path.exists(metrics_file):
            try:
                return pd.read_csv(metrics_file)
            except Exception as e:
                print(f"Warning: Could not load metrics file {metrics_file}: {e}")
        
        return None
    
    def generate_performance_summary(self, df: pd.DataFrame) -> str:
        """Generate performance summary section.
        
        Args:
            df: DataFrame with performance metrics
            
        Returns:
            Markdown formatted performance summary
        """
        if df is None or df.empty:
            return "No performance data available."
        
        summary = []
        summary.append("## Performance Summary\n")
        
        for bench in df['bench'].unique():
            bench_data = df[df['bench'] == bench]
            summary.append(f"### {bench.replace('_', ' ').title()} Benchmark\n")
            
            if bench == 'cpp_compute':
                summary.append("**Computational Performance Analysis:**\n")
                
                # Filter for p95 data only
                p95_data = bench_data[bench_data['level_4'] == 'p95']
                if not p95_data.empty:
                    # Find best performance
                    best_performance = p95_data.loc[p95_data['wall_ms'].idxmin()]
                    worst_performance = p95_data.loc[p95_data['wall_ms'].idxmax()]
                    
                    summary.append(f"- **Best Performance:** {best_performance['config']} optimization, "
                                  f"{best_performance['threads']} threads - {best_performance['wall_ms']:.3f}ms P95")
                    summary.append(f"- **Worst Performance:** {worst_performance['config']} optimization, "
                                  f"{worst_performance['threads']} threads - {worst_performance['wall_ms']:.3f}ms P95")
                    
                    # Calculate improvement
                    improvement = ((worst_performance['wall_ms'] - best_performance['wall_ms']) / 
                                  worst_performance['wall_ms']) * 100
                    summary.append(f"- **Optimization Impact:** {improvement:.1f}% performance improvement\n")
                else:
                    summary.append("- No P95 performance data available\n")
                
            elif bench == 'py_io':
                summary.append("**I/O Performance Analysis:**\n")
                
                # Filter for p95 data only
                p95_data = bench_data[bench_data['level_4'] == 'p95']
                if not p95_data.empty:
                    # Find most efficient block size
                    best_block = p95_data.loc[p95_data['wall_ms'].idxmin()]
                    worst_block = p95_data.loc[p95_data['wall_ms'].idxmax()]
                    
                    summary.append(f"- **Most Efficient:** {best_block['block_kb']:.0f}KB blocks - "
                                  f"{best_block['wall_ms']:.1f}ms P95")
                    summary.append(f"- **Least Efficient:** {worst_block['block_kb']:.0f}KB blocks - "
                                  f"{worst_block['wall_ms']:.1f}ms P95")
                    
                    # Calculate efficiency ratio
                    efficiency = best_block['wall_ms'] / worst_block['wall_ms']
                    summary.append(f"- **Efficiency Ratio:** {efficiency:.2f} (lower is better)\n")
                else:
                    summary.append("- No P95 performance data available\n")
        
        return "\n".join(summary)
    
    def generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate performance recommendations.
        
        Args:
            df: DataFrame with performance metrics
            
        Returns:
            Markdown formatted recommendations
        """
        if df is None or df.empty:
            return "No recommendations available without performance data."
        
        recommendations = []
        recommendations.append("## Performance Recommendations\n")
        
        for bench in df['bench'].unique():
            bench_data = df[df['bench'] == bench]
            
            if bench == 'cpp_compute':
                recommendations.append("### Computational Optimization\n")
                
                # Filter for p95 data only
                p95_data = bench_data[bench_data['level_4'] == 'p95']
                if not p95_data.empty:
                    # Analyze thread scaling
                    thread_scaling = p95_data.groupby('threads')['wall_ms'].mean()
                    if len(thread_scaling) > 1:
                        scaling_efficiency = thread_scaling.iloc[0] / thread_scaling.iloc[-1]
                        if scaling_efficiency > 0.8:
                            recommendations.append("- **Thread Scaling:** Good scalability observed across thread counts")
                        else:
                            recommendations.append("- **Thread Scaling:** Consider investigating thread contention or memory bandwidth limitations")
                    
                    # Analyze optimization levels
                    opt_performance = p95_data.groupby('config')['wall_ms'].mean()
                    if 'O3' in opt_performance.index and 'O2' in opt_performance.index:
                        o3_improvement = ((opt_performance['O2'] - opt_performance['O3']) / 
                                        opt_performance['O2']) * 100
                        if o3_improvement > 5:
                            recommendations.append(f"- **Compiler Optimization:** O3 provides {o3_improvement:.1f}% improvement over O2")
                        else:
                            recommendations.append("- **Compiler Optimization:** O3 shows minimal improvement over O2")
                else:
                    recommendations.append("- No P95 performance data available for analysis")
                
            elif bench == 'py_io':
                recommendations.append("### I/O Optimization\n")
                
                # Filter for p95 data only
                p95_data = bench_data[bench_data['level_4'] == 'p95']
                if not p95_data.empty:
                    # Find optimal block size
                    optimal_block = p95_data.loc[p95_data['wall_ms'].idxmin()]
                    recommendations.append(f"- **Optimal Block Size:** {optimal_block['block_kb']:.0f}KB provides best performance")
                    
                    # Analyze block size impact
                    block_range = p95_data['block_kb'].max() / p95_data['block_kb'].min()
                    if block_range > 10:
                        recommendations.append("- **Block Size Strategy:** Consider using adaptive block sizing based on data characteristics")
                    else:
                        recommendations.append("- **Block Size Strategy:** Current block size range is appropriate")
                else:
                    recommendations.append("- No P95 performance data available for analysis")
        
        recommendations.append("\n### General Recommendations\n")
        recommendations.append("- **Monitoring:** Implement continuous performance monitoring in production")
        recommendations.append("- **Baseline:** Establish performance baselines for regression testing")
        recommendations.append("- **Profiling:** Use detailed profiling tools for deeper performance analysis")
        
        return "\n".join(recommendations)
    
    def generate_markdown_report(self, 
                               metrics_file: str,
                               env_file: str,
                               output_file: str,
                               figures_dir: str) -> str:
        """Generate a comprehensive markdown report.
        
        Args:
            metrics_file: Path to performance metrics CSV
            env_file: Path to environment information file
            output_file: Path to save the report
            figures_dir: Directory containing generated figures
            
        Returns:
            Path to generated report file
        """
        # Load data
        df = self.load_performance_data(metrics_file)
        env_info = self.load_environment_info(env_file)
        
        # Generate report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = []
        
        # Header
        report_content.append(f"# {self.project_name} - Performance Analysis Report")
        report_content.append("")
        report_content.append(f"**Generated:** {timestamp}")
        report_content.append(f"**Version:** {self.project_version}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("## Executive Summary")
        report_content.append("")
        report_content.append("This report presents a comprehensive performance analysis of computational and I/O workloads ")
        report_content.append("relevant to autonomous vehicle systems. The analysis includes statistical performance metrics, ")
        report_content.append("visualizations, and actionable recommendations for system optimization.")
        report_content.append("")
        
        # Environment Information
        report_content.append("## System Environment")
        report_content.append("")
        if env_info.get('raw_content'):
            report_content.append("```")
            report_content.append(env_info['raw_content'])
            report_content.append("```")
        else:
            report_content.append("Environment information not available.")
        report_content.append("")
        
        # Performance Summary
        performance_summary = self.generate_performance_summary(df)
        report_content.append(performance_summary)
        
        # Visualizations
        report_content.append("## Performance Visualizations")
        report_content.append("")
        
        if os.path.exists(figures_dir):
            figure_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
            figure_files.sort()
            
            for fig_file in figure_files:
                fig_name = fig_file.replace('.png', '').replace('_', ' ').title()
                report_content.append(f"### {fig_name}")
                report_content.append("")
                report_content.append(f"![{fig_name}](figs/{fig_file})")
                report_content.append("")
        else:
            report_content.append("No visualizations available.")
        
        # Recommendations
        recommendations = self.generate_recommendations(df)
        report_content.append(recommendations)
        
        # Methodology
        report_content.append("## Methodology")
        report_content.append("")
        report_content.append("### Benchmark Design")
        report_content.append("- **Computational Benchmarks:** CPU-intensive workloads with varying optimization levels and thread counts")
        report_content.append("- **I/O Benchmarks:** File system operations with different block sizes")
        report_content.append("- **Statistical Analysis:** P50, P95, P99 percentiles and mean values")
        report_content.append("")
        report_content.append("### Data Collection")
        report_content.append("- Multiple iterations per configuration for statistical significance")
        report_content.append("- Wall-clock timing for accurate performance measurement")
        report_content.append("- Environment information capture for reproducibility")
        report_content.append("")
        
        # Conclusion
        report_content.append("## Conclusion")
        report_content.append("")
        report_content.append("This performance analysis provides insights into system behavior under different workloads ")
        report_content.append("and configurations. The results can be used to optimize system performance and establish ")
        report_content.append("baselines for future development and testing.")
        report_content.append("")
        report_content.append("For questions or additional analysis, please refer to the project documentation or ")
        report_content.append("contact the performance analysis team.")
        
        # Write report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        return output_file


def main():
    """Main entry point for report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate professional performance analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report_generator.py -m metrics.csv -e env.txt -o report/REPORT.md -f report/figs
  python report_generator.py --metrics out/metrics.csv --env out/env.txt --output report.md
        """
    )
    
    parser.add_argument("-m", "--metrics", required=True,
                       help="Path to performance metrics CSV file")
    parser.add_argument("-e", "--env", required=True,
                       help="Path to environment information file")
    parser.add_argument("-o", "--output", required=True,
                       help="Output path for the report")
    parser.add_argument("-f", "--figures", required=True,
                       help="Directory containing generated figures")
    parser.add_argument("--config", default="config/config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize report generator
    config_manager = get_config()
    generator = ReportGenerator(config_manager)
    
    try:
        # Generate report
        report_path = generator.generate_markdown_report(
            args.metrics, args.env, args.output, args.figures
        )
        
        print(f"Report generated successfully: {report_path}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
