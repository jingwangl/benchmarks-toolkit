"""
Autonomous Vehicle Performance Analysis Toolkit - Demo Script
This script demonstrates the capabilities of the performance analysis toolkit
for autonomous vehicle systems.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class ToolkitDemo:
    """Demonstrates the capabilities of the performance analysis toolkit."""
    
    def __init__(self):
        """Initialize the demo."""
        self.root_dir = Path(__file__).parent
        self.demo_start_time = time.time()
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_step(self, step: str):
        """Print a step indicator."""
        print(f"\n[STEP] {step}")
        print("-" * 60)
    
    def run_command(self, command: str, description: str = "") -> bool:
        """Run a command and return success status."""
        if description:
            print(f"Running: {description}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[SUCCESS] {description}")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
                return True
            else:
                print(f"[FAILED] {description}")
                if result.stderr.strip():
                    print(f"Error: {result.stderr.strip()}")
                return False
        except Exception as e:
            print(f"[EXCEPTION] {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        self.print_step("Checking Prerequisites")
        
        prerequisites = [
            ("python --version", "Python installation"),
            ("pip --version", "pip package manager"),
            ("python -c \"import pandas, matplotlib, numpy\"", "Required Python packages")
        ]
        
        all_good = True
        for command, description in prerequisites:
            if not self.run_command(command, description):
                all_good = False
        
        return all_good
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        self.print_step("Installing Dependencies")
        
        return self.run_command(
            "pip install -r requirements.txt",
            "Installing Python dependencies"
        )
    
    def run_benchmarks(self) -> bool:
        """Run all benchmark tests."""
        self.print_step("Running Performance Benchmarks")
        
        # Check if we're on Windows or Linux
        if os.name == 'nt':  # Windows
            return self.run_command(
                "powershell -ExecutionPolicy Bypass -File run.ps1 all",
                "Running complete benchmark suite (PowerShell)"
            )
        else:  # Linux/macOS
            return self.run_command(
                "bash run.sh all",
                "Running complete benchmark suite (Bash)"
            )
    
    def show_results(self):
        """Display the generated results."""
        self.print_step("Generated Results")
        
        # Check output files
        output_files = [
            ("out/metrics.csv", "Performance metrics"),
            ("report/figs/cpp_compute_p95.png", "C++ computational performance chart"),
            ("report/figs/py_io_p95.png", "Python I/O performance chart"),
            ("report/figs/performance_comparison.png", "Performance comparison chart"),
            ("report/REPORT.md", "Comprehensive analysis report")
        ]
        
        print("Generated files:")
        for file_path, description in output_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"  [OK] {file_path} ({size:,} bytes) - {description}")
            else:
                print(f"  [MISSING] {file_path} - {description} (not found)")
    
    def show_capabilities(self):
        """Show the capabilities of the toolkit."""
        self.print_step("Toolkit Capabilities")
        
        capabilities = [
            "Autonomous Vehicle Focus: LiDAR point cloud processing simulation",
            "Multi-language Benchmarking: C++ computational and Python I/O workloads",
            "Professional Visualizations: Publication-ready charts and graphs",
            "Comprehensive Reporting: Automated analysis with recommendations",
            "Configurable Parameters: Thread counts, optimization levels, block sizes",
            "Statistical Analysis: P50, P95, P99 percentiles and trend analysis",
            "Cross-platform Support: Windows PowerShell and Linux Bash compatibility",
            "Structured Output: CSV data, PNG charts, and Markdown reports"
        ]
        
        for capability in capabilities:
            print(f"  {capability}")
    
    def show_use_cases(self):
        """Show potential use cases for autonomous vehicle development."""
        self.print_step("Autonomous Vehicle Use Cases")
        
        use_cases = [
            "Sensor Data Processing: Analyze computational requirements for LiDAR, camera, radar",
            "Real-time Performance: Ensure system meets real-time constraints",
            "Bottleneck Identification: Locate performance-critical components",
            "Scalability Analysis: Understand system behavior under different loads",
            "Regression Testing: Track performance changes across software versions",
            "Hardware Evaluation: Compare performance across different configurations",
            "Resource Planning: Plan hardware requirements based on performance data",
            "Optimization Guidance: Data-driven recommendations for system improvement"
        ]
        
        for use_case in use_cases:
            print(f"  {use_case}")
    
    def run_demo(self):
        """Run the complete demo."""
        self.print_header("Autonomous Vehicle Performance Analysis Toolkit - Demo")
        
        print("""
This demo showcases a comprehensive performance analysis toolkit designed
specifically for autonomous vehicle systems. The toolkit provides:

- Automated benchmarking of computational and I/O workloads
- Professional visualizations and statistical analysis
- LiDAR point cloud processing simulation
- Cross-platform compatibility and easy deployment
- Comprehensive reporting with actionable insights

Let's run through the complete workflow...
        """)
        
        # Run demo steps
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Installing dependencies", self.install_dependencies),
            ("Running benchmarks", self.run_benchmarks),
            ("Displaying results", self.show_results)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n[ERROR] Demo failed at step: {step_name}")
                return False
        
        # Show additional information
        self.show_capabilities()
        self.show_use_cases()
        
        # Demo completion
        demo_time = time.time() - self.demo_start_time
        self.print_header("Demo Complete!")
        
        print(f"""
Demo completed successfully in {demo_time:.1f} seconds!

Check the following directories for results:
   - out/ - Raw data and processed metrics
   - report/figs/ - Generated visualizations
   - report/REPORT.md - Comprehensive analysis report

This toolkit demonstrates professional performance analysis capabilities
suitable for autonomous vehicle development and optimization.

Next steps:
   - Review the generated report and visualizations
   - Modify configuration parameters in config/config.json
   - Add custom benchmarks for your specific use cases
   - Integrate with CI/CD pipelines for continuous performance monitoring
        """)
        
        return True


def main():
    """Main entry point for the demo."""
    demo = ToolkitDemo()
    
    try:
        success = demo.run_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
