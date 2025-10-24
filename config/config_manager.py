"""
Configuration management for the Autonomous Vehicle Performance Analysis Toolkit.
Handles loading and validation of configuration files.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                print(f"Warning: Config file {self.config_path} not found, using defaults")
                self._set_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            self._set_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self._set_default_config()
    
    def _set_default_config(self) -> None:
        """Set default configuration values."""
        self.config = {
            "project": {
                "name": "Autonomous Vehicle Performance Analysis Toolkit",
                "version": "1.0.0"
            },
            "benchmarks": {
                "cpp_compute": {
                    "optimization_levels": ["O2", "O3"],
                    "thread_counts": [1, 4, 8],
                    "iterations": 2000000
                },
                "py_io": {
                    "block_sizes_kb": [4, 64, 512],
                    "operation_count": 100,
                    "iterations": 3
                }
            },
            "analysis": {
                "statistical_metrics": ["p50", "p95", "p99", "mean"],
                "figure_size": [10, 6],
                "dpi": 100
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_benchmark_config(self, benchmark_name: str) -> Dict[str, Any]:
        """Get configuration for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            
        Returns:
            Benchmark configuration dictionary
        """
        return self.get(f"benchmarks.{benchmark_name}", {})
    
    def get_optimization_levels(self) -> List[str]:
        """Get C++ optimization levels."""
        return self.get("benchmarks.cpp_compute.optimization_levels", ["O2", "O3"])
    
    def get_thread_counts(self) -> List[int]:
        """Get thread count configurations."""
        return self.get("benchmarks.cpp_compute.thread_counts", [1, 4, 8])
    
    def get_block_sizes(self) -> List[int]:
        """Get I/O block sizes in KB."""
        return self.get("benchmarks.py_io.block_sizes_kb", [4, 64, 512])
    
    def get_statistical_metrics(self) -> List[str]:
        """Get statistical metrics to calculate."""
        return self.get("analysis.statistical_metrics", ["p50", "p95", "p99", "mean"])
    
    def get_figure_size(self) -> List[int]:
        """Get matplotlib figure size."""
        return self.get("analysis.figure_size", [10, 6])
    
    def get_dpi(self) -> int:
        """Get matplotlib DPI setting."""
        return self.get("analysis.dpi", 100)
    
    def validate_config(self) -> bool:
        """Validate configuration file structure.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ["project", "benchmarks", "analysis"]
        
        for section in required_sections:
            if section not in self.config:
                print(f"Error: Missing required section '{section}' in config")
                return False
        
        # Validate benchmark configurations
        benchmarks = self.config.get("benchmarks", {})
        if "cpp_compute" not in benchmarks:
            print("Error: Missing cpp_compute benchmark configuration")
            return False
        
        if "py_io" not in benchmarks:
            print("Error: Missing py_io benchmark configuration")
            return False
        
        return True
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Optional path to save config, defaults to original path
        """
        path = Path(output_path) if output_path else self.config_path
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def print_config(self) -> None:
        """Print current configuration in a readable format."""
        print("Current Configuration:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance.
    
    Returns:
        Global ConfigManager instance
    """
    return config
