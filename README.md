# 自动驾驶车辆性能分析工具包

专为自动驾驶车辆系统设计的基准测试与性能分析工具包。提供自动化基准测试、数据聚合与可视化，以及可复用的报告生成流程。

## 🚗 功能特性

### 核心能力
- **多语言基准测试**：C++ 计算工作负载、Python I/O 操作、LiDAR 点云模拟
- **自动化数据收集**：墙钟时间；批量多次运行以支撑统计聚合
- **环境信息捕获（Linux）**：`envinfo.sh` 输出 `out/env.txt`
- **统计与可视化**：`analyze/parse.py` 统计聚合，`analyze/plot.py` 生成图表
- **报告生成**：`analyze/report_generator.py` 产出 `report/REPORT.md`
- **跨平台脚本**：Windows PowerShell 与 Linux Bash


## 🚀 快速开始

### 前置要求
- Python 3.8+（依赖见 `requirements.txt`）
- g++（Windows 可使用 MinGW/MSYS2，或修改为 MSVC）
- PowerShell（Windows）或 Bash（Linux/macOS）

### 安装
```bash
# 克隆仓库
git clone <repository-url>
cd benchmarks-toolkit

# 安装Python依赖
pip install -r requirements.txt

# 运行完整基准流程（构建→运行→分析→报告）
powershell -ExecutionPolicy Bypass -File run.ps1 all   # Windows
# 或
bash run.sh all                                        # Linux/macOS
```

### 输出产物
- 原始数据：`out/*_raw.csv`
- 聚合指标：`out/metrics.csv`
- 可视化：`report/figs/*.png`
- 报告：`report/REPORT.md`

## 📊 基准测试套件

### C++ 计算（`bench/cpp_compute/`）
- **目的**：CPU 密集型计算的线程扩展性与编译优化对比
- **参数**：
  - 优化级别：`O2`、`O3`
  - 线程数：`1, 4, 8`
  - 迭代次数：`90000000`
  - 运行次数：30 次
  - 输出：`out/cpp_compute_raw.csv`，列为 `bench,config,threads,wall_ms,iterations`

### Python I/O（`bench/py_io/`）
- **目的**：固定总数据量模式下不同块大小的 I/O 延迟差异
- **参数**：
  - 块大小：`4KB, 64KB, 512KB`
  - 模式：PowerShell 使用 `fixed_total`（`--total_mb` 固定，默认 4MB，脚本内注释为 32MB 但当前值为 4）
  - 循环：每种块大小 30 次
  - 输出：
    - PowerShell：`out/py_io_raw.csv`，列为 `bench,block_kb,wall_ms,total_mb,blocks`
    - Bash：`out/py_io_raw.csv`，列为 `bench,block_kb,wall_ms,count`

### LiDAR 处理（`bench/lidar_processing/`）
- **目的**：点云生成→地面过滤→DBSCAN 聚类→包围盒统计的端到端耗时
- **参数**：
  - 点数: `5k, 10k, 20k, 50k`，
  - 运行次数： 30 次
  - 输出：`out/lidar_processing_raw.csv`，列为 `bench,points,wall_ms,iterations`

## 🏗️ 仓库结构

```
benchmarks-toolkit/
├── bench/
│   ├── cpp_compute/
│   ├── py_io/
│   └── lidar_processing/
├── analyze/
│   ├── parse.py
│   ├── plot.py
│   └── report_generator.py
├── collect/
├── config/
├── out/
├── report/
└── run.ps1 / run.sh
```

## 🔧 配置（`config/config.json` 与 `config_manager.py`）
- 统计指标：`p50, p95, p99, mean`
- 可视化：默认 `figure_size=[10,6]`, `dpi=100`, 样式基于 seaborn
- 基准参数（供工具参考）：
  - C++：`optimization_levels=["O2","O3"], thread_counts=[1,4,8], iterations=200000`
  - I/O：`block_sizes_kb=[4,64,512], operation_count=20, iterations=3`
  - LiDAR：`point_cloud_sizes=[5000,10000,20000,50000], iterations=3`
说明：脚本实际运行次数/规模以 `bench/*/run.ps1|run.sh` 为准，配置文件用于分析与可视化的默认参数来源。

## 📈 运行与报告

### Windows（PowerShell）
```powershell
./run.ps1 build   # 构建 C++ 基准
./run.ps1 run     # 运行 C++、I/O、LiDAR 全部基准
./run.ps1 analyze # 统计聚合 + 生成图表
./run.ps1 report  # 生成 REPORT.md
./run.ps1 all     # 一键完成全部流程
```

### Linux/macOS（Bash）
```bash
./run.sh build    # 构建 C++ 基准
./run.sh run      # 运行采集（含 envinfo.sh）
./run.sh analyze  # 统计聚合 + 生成图表
./run.sh report   # 生成 REPORT.md
./run.sh all      # 一键完成全部流程
```

## 📊 分析与可视化说明
- `analyze/parse.py`：自动识别 `bench` 及关键参数列，按 `p50/p95/p99/mean` 聚合
- `analyze/plot.py`：输出三类图表（C++ 线程扩展性、I/O 块大小、LiDAR 点数），另含总览图 `performance_comparison.png`
- 图像默认保存到 `report/figs/`

## 📝 注意事项
- LiDAR 依赖 `scikit-learn` 的 DBSCAN；请确认已安装 `scikit-learn`
- `envinfo.sh` 当前仅在 Linux/macOS 下有效，Windows 环境信息由报告阶段直接读取现有 `out/env.txt`（如存在）
- I/O 基准的 PowerShell 与 Bash 脚本输出列名不同，`parse.py` 会按存在列聚合

## 📄 许可证与用途
该项目为自动驾驶性能分析演示用途，聚焦计算与 I/O 基准化流程与可视化呈现。