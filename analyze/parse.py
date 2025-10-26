"""
性能基准测试的统计分析和数据聚合。
自动驾驶车辆性能分析工具包的一部分。
"""

import argparse
import pandas as pd
import numpy as np
import os
import glob
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add config directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "config"))
from config_manager import get_config


class PerformanceAnalyzer:
    """Handles statistical analysis of performance benchmark data."""
    
    def __init__(self, config_manager=None):
        """Initialize the performance analyzer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or get_config()
        self.statistical_metrics = self.config.get_statistical_metrics()
    
    def load_data(self, input_patterns: List[str]) -> pd.DataFrame:
        """Load and combine CSV data from multiple sources.
        
        Args:
            input_patterns: List of file patterns to load
            
        Returns:
            Combined DataFrame with all benchmark data
            
        Raises:
            SystemExit: If no valid CSV files are found
        """
        dataframes = []
        
        for pattern in input_patterns:
            for file_path in glob.glob(pattern):
                if file_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        print(f"Loaded {len(df)} records from {file_path}")
                        dataframes.append(df)
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
        
        if not dataframes:
            raise SystemExit("Error: No valid CSV files found")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total records")
        
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['bench', 'wall_ms']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in data")
                return False
        
        # Check for empty wall_ms values
        empty_wall_ms = df['wall_ms'].isna().sum()
        if empty_wall_ms > 0:
            print(f"Warning: {empty_wall_ms} records have empty wall_ms values")
        
        return True
    
    def calculate_statistics(self, data: pd.Series) -> Optional[pd.Series]:
        """Calculate statistical metrics for a data series.
        
        Args:
            data: Series of wall_ms values
            
        Returns:
            Series with calculated statistics or None if data is invalid
        """
        if data.empty or data.isna().all():
            return None
        
        # Remove NaN values
        clean_data = data.dropna()
        if clean_data.empty:
            return None
        
        statistics = {}
        
        for metric in self.statistical_metrics:
            if metric == 'p50':
                statistics[metric] = np.percentile(clean_data, 50)
            elif metric == 'p95':
                statistics[metric] = np.percentile(clean_data, 95)
            elif metric == 'p99':
                statistics[metric] = np.percentile(clean_data, 99)
            elif metric == 'mean':
                statistics[metric] = clean_data.mean()
            else:
                print(f"Warning: Unknown metric '{metric}'")
        
        return pd.Series(statistics)
    
    def identify_grouping_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns to use for grouping data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names to use for grouping
        """
        grouping_columns = []
        # Include common knobs across benches; add 'points' for LiDAR to split stats by cloud size
        potential_groups = ['config', 'threads', 'block_kb', 'optimization_level', 'points']
        
        for col in potential_groups:
            if col in df.columns:
                grouping_columns.append(col)
        
        return grouping_columns
    
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by benchmark and configuration parameters.
        
        Args:
            df: Input DataFrame with raw benchmark data
            
        Returns:
            Aggregated DataFrame with statistical summaries
        """
        grouping_columns = self.identify_grouping_columns(df)
        
        print(f"Grouping by: {['bench'] + grouping_columns}")
        
        # Group data and calculate statistics
        grouped = df.groupby(['bench'] + grouping_columns, dropna=False)
        
        # Apply statistical calculation
        aggregated = grouped['wall_ms'].apply(self.calculate_statistics).reset_index()
        
        # Remove rows where statistics couldn't be calculated
        if not aggregated.empty:
            # Check which statistical metrics are actually present
            available_metrics = [col for col in self.statistical_metrics if col in aggregated.columns]
            if available_metrics:
                aggregated = aggregated.dropna(subset=available_metrics)
        
        print(f"Aggregated data: {len(aggregated)} summary records")
        
        return aggregated
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save aggregated results to CSV file.
        
        Args:
            df: Aggregated DataFrame to save
            output_path: Path to save the CSV file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Saved aggregated results to {output_path}")
    
    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate a summary report of the analysis.
        
        Args:
            df: Aggregated DataFrame to summarize
        """
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        for bench in df['bench'].unique():
            bench_data = df[df['bench'] == bench]
            print(f"\n{bench.upper()} BENCHMARK:")
            print("-" * 40)
            
            for _, row in bench_data.iterrows():
                config_str = ", ".join([f"{col}={row[col]}" 
                                      for col in df.columns 
                                      if col not in ['bench'] + self.statistical_metrics 
                                      and pd.notna(row[col])])
                
                print(f"  {config_str}:")
                
                # Print available metrics with safe access
                metrics_to_print = ['p50', 'p95', 'p99', 'mean']
                for metric in metrics_to_print:
                    if metric in row and pd.notna(row[metric]):
                        print(f"    {metric.upper()}: {row[metric]:.3f}ms")
                
                print()


def main():
    """Main entry point for the performance analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze performance benchmark data and generate statistical summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse.py out/*_raw.csv -o out/metrics.csv
  python parse.py cpp_data.csv py_data.csv -o results.csv
        """
    )
    
    parser.add_argument("inputs", nargs="+", 
                       help="Input CSV files or patterns to analyze")
    parser.add_argument("-o", "--output", required=True,
                       help="Output CSV file for aggregated results")
    parser.add_argument("--config", default="config/config.json",
                       help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    config_manager = get_config()
    analyzer = PerformanceAnalyzer(config_manager)
    
    if args.verbose:
        print("Configuration loaded:")
        config_manager.print_config()
    
    try:
        # Load and validate data
        df = analyzer.load_data(args.inputs)
        
        if not analyzer.validate_data(df):
            print("Error: Data validation failed")
            sys.exit(1)
        
        # Perform analysis
        aggregated_df = analyzer.aggregate_data(df)
        
        # Save results
        analyzer.save_results(aggregated_df, args.output)
        
        # Generate summary report
        analyzer.generate_summary_report(aggregated_df)
        
        print(f"\nAnalysis complete! Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
