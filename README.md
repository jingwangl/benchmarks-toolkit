# Autonomous Vehicle Performance Analysis Toolkit

A comprehensive benchmarking and performance analysis toolkit designed for autonomous vehicle systems. This toolkit provides automated performance testing, data collection, and visualization capabilities to help analyze computational bottlenecks and optimize system performance.

## 🚗 Features

### Core Capabilities
- **Multi-language Benchmarking**: C++ computational workloads and Python I/O operations
- **Automated Data Collection**: Wall-clock timing, statistical analysis (P50/P95/P99)
- **Environment Profiling**: CPU, memory, and system configuration capture
- **Professional Reporting**: Automated report generation with visualizations
- **Cross-platform Support**: Windows PowerShell and Linux Bash compatibility

### Performance Metrics
- **Computational Performance**: CPU-bound workloads with different optimization levels
- **I/O Performance**: File system operations with varying block sizes
- **Statistical Analysis**: Comprehensive percentile analysis and trend detection
- **Resource Utilization**: System resource monitoring and profiling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pandas and matplotlib
- C++ compiler (g++ or MSVC)
- PowerShell (Windows) or Bash (Linux/macOS)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd benchmarks-toolkit

# Install Python dependencies
pip install -r requirements.txt

# Run complete benchmark suite
powershell -ExecutionPolicy Bypass -File run.ps1 all  # Windows
# OR
bash run.sh all  # Linux/macOS
```

### Results
- **Raw Data**: `out/*.csv` - Detailed performance measurements
- **Analysis**: `out/metrics.csv` - Statistical summaries
- **Visualizations**: `report/figs/*.png` - Performance charts
- **Report**: `report/REPORT.md` - Comprehensive analysis report

## 📊 Benchmark Suites

### Computational Benchmarks (`bench/cpp_compute/`)
- **Purpose**: CPU-intensive workload analysis
- **Parameters**: 
  - Optimization levels: O2, O3
  - Thread counts: 1, 4, 8
  - Iterations: 200,000 (optimized for home computers)
- **Metrics**: Execution time, throughput, scalability

### I/O Benchmarks (`bench/py_io/`)
- **Purpose**: Storage subsystem performance analysis
- **Parameters**:
  - Block sizes: 4KB, 64KB, 512KB
  - Operations: 20 read/write cycles (optimized for home computers)
- **Metrics**: Latency, throughput, I/O efficiency

### LiDAR Processing (`bench/lidar_processing/`)
- **Purpose**: Real-time point cloud processing simulation
- **Parameters**:
  - Point cloud sizes: 5K, 10K, 20K, 50K points (optimized for home computers)
  - Processing pipeline stages
- **Metrics**: End-to-end processing time, stage-wise breakdown

## 🏗️ Architecture

```
benchmarks-toolkit/
├── bench/                    # Benchmark implementations
│   ├── cpp_compute/         # C++ computational workloads
│   └── py_io/               # Python I/O operations
├── analyze/                 # Data analysis and visualization
│   ├── parse.py            # Statistical analysis
│   └── plot.py             # Chart generation
├── collect/                # Data collection orchestration
├── out/                    # Output data and results
├── report/                 # Generated reports and visualizations
└── config/                 # Configuration management
```

## 🔧 Configuration

### Benchmark Parameters
- **Thread Configuration**: Adjustable thread counts for scalability testing
- **Block Sizes**: Configurable I/O block sizes for storage analysis
- **Optimization Levels**: Compiler optimization settings for performance comparison

### Analysis Settings
- **Statistical Metrics**: P50, P95, P99 percentiles and mean values
- **Visualization Options**: Chart types, colors, and formatting
- **Report Templates**: Customizable report generation

## 📈 Use Cases

### Autonomous Vehicle Development
- **Sensor Data Processing**: Analyze computational requirements for LiDAR, camera, and radar data
- **Real-time Performance**: Ensure system meets real-time constraints
- **Resource Optimization**: Identify bottlenecks in computational pipelines

### Performance Regression Testing
- **Continuous Integration**: Automated performance validation
- **Version Comparison**: Track performance changes across software versions
- **Hardware Evaluation**: Compare performance across different hardware configurations

### System Profiling
- **Bottleneck Identification**: Locate performance-critical components
- **Scalability Analysis**: Understand system behavior under different loads
- **Resource Planning**: Plan hardware requirements based on performance data

## 🛠️ Development

### Adding New Benchmarks
1. Create benchmark implementation in `bench/` directory
2. Implement data collection script (CSV output format)
3. Add configuration parameters
4. Update analysis scripts if needed

### Extending Analysis
- **Custom Metrics**: Add new statistical measures
- **Visualization Types**: Create additional chart types
- **Report Formats**: Extend report generation capabilities

## 📋 Roadmap

### Phase 1: Core Enhancements
- [ ] Enhanced error handling and validation
- [ ] Configuration file management
- [ ] Improved cross-platform compatibility

### Phase 2: Advanced Features
- [ ] Real-time monitoring capabilities
- [ ] Advanced statistical analysis
- [ ] Performance regression detection

### Phase 3: Autonomous Vehicle Focus
- [ ] ROS2 integration for sensor data analysis
- [ ] Real-time system profiling
- [ ] Autonomous driving workload simulation

## 🤝 Contributing

This project is designed to demonstrate performance analysis capabilities for autonomous vehicle systems. Contributions are welcome for:
- New benchmark implementations
- Analysis algorithm improvements
- Visualization enhancements
- Documentation updates

## 📄 License

This project is created for demonstration purposes in autonomous vehicle performance analysis.

---

**Created for**: Autonomous Vehicle Performance Analyst Position  
**Focus**: Computational performance analysis, system profiling, and optimization