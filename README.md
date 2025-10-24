# benchmarks-toolkit (MVP)

一键跑基准 → 采集数据 → 生成图表与报告的轻量工具集。**用途：**做性能回归/对比实验、快速定位计算 vs I/O 短板，并把结果沉淀为可复现实验与文档。

## 快速开始
```bash
# 需要 g++、python3（pandas, matplotlib）
pip install -r requirements.txt
bash run.sh all
# 结果在 out/ 与 report/ 下（PNG 图与 REPORT.md）
```

## 含有哪些基准
- `bench/cpp_compute`：CPU 计算密集（O2/O3、线程数可配）
- `bench/py_io`：简单文件写读，块大小可配（4/64/512KB）

## 采集/分析的指标
- 程序级：wall clock（ms），重复运行后计算 P50/P95/P99、mean
- 环境信息：CPU/内核/内存（envinfo.sh）
- *可选扩展：*perf/pidstat/iostat 采样（MVP 未强依赖）

## 目录
见仓库结构；关键入口脚本：`run.sh`。

## 报告产物
- `out/*.csv`：原始数据与聚合后的 `metrics.csv`
- `report/figs/*.png`：图表
- `report/REPORT.md`：自动生成的 Markdown 报告（包含环境信息与图）

## 设计理念
- **可复现**：参数/环境/数据均落盘；记录 git hash 便于回溯
- **可对比**：配好参数矩阵（优化等级/线程数/块大小）即可生成对比图
- **可扩展**：新增基准只需提供 `run.sh` 输出 CSV 即可被统一分析

## 下一步（Roadmap）
- 增加 perf 事件采集（cycles/instructions/LLC miss）并纳入报告
- 加入 ROS/bag 简单统计脚本，贴近车载日志分析
- 提供 CI 工作流，自动校验分析脚本与样例数据
```
