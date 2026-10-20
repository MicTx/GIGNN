# 架构

GIGNN 是 Python 库包，不是托管服务。代码库围绕空间分类流水线组织：表格化空间单元数据从估计器或工具层进入，图结构被校验或构建，推导高斯边属性/边权，训练与 notebook 对齐的加权多跳 GIGNN，最后由报告/图表暴露决策与诊断。

## 仓库结构

```text
gignn/       运行时库包
tests/       回归与契约测试
examples/    可运行的冒烟示例
docs/        用户、开发、发布与代理文档
research/    论文与 notebook 证据（用于对齐审计）
.spec/       任务包、归档记录与可复用项目笔记
```

只有 `gignn/` 是运行时包代码。`research/` 是实现状态审计的证据，库不应导入它。

## 包结构

```text
gignn/
├── __init__.py              # 顶层公开导出与绘图惰性加载
├── auto.py                  # 高层自动工作流估计器
├── auto_archive.py          # 安全 ZIP/JSON/NPY 自动检查点容器
├── auto_persistence.py      # 自动工作流载荷模式与重建
├── splitting.py             # 训练/验证/测试与空间分组切分
├── ordinal.py               # 显式有序目标编码
├── estimators.py            # 直接 sklearn 风格分类器
├── validation.py            # 输入与边校验、标签编码
├── generalization.py        # 数据集标准化与预测诊断
├── reports.py               # 报告 dataclass
├── plotting.py              # Matplotlib/NetworkX 绘图层
├── limits.py                # 资源上限
├── geodetector/             # 离散化、q 统计、交互、选择器
├── graph/                   # 图构建与 PyG 数据转换
├── models/                  # notebook 对齐 GIGNN 与 GraphSAGE 兼容模型
├── training/                # 训练循环、类权重、种子
└── io/                      # 检查点持久化
```

## 公开入口

顶层 `gignn` 导出是主要稳定 API：

- `GIGNNAutoClassifier`
- `GIGNNClassifier`
- 泛化辅助
- 绘图辅助
- 包异常

`gignn.geodetector` 与 `gignn.graph` 也暴露有文档的工具 API。`gignn.training`、`gignn.models`、`gignn.io`、`gignn.validation` 对高级用户有用，但在 `1.0` 前应视为更低层的集成面。

## 数据流

```text
pandas DataFrame + 标签 + geometry/edge_index
        |
        v
校验与可选的空间数据集标准化
        |
        v
地理探测器离散化 -> q 评分 -> 特征选择
        |
        v
图候选项构建或显式边校验
        |
        v
带编码标签、edge_attr、edge_weight 的 PyG Data 转换
        |
        v
加权多跳 GIGNN 训练
        |
        v
预测、概率预测、报告、模拟、绘图
```

## 自动工作流

`GIGNNAutoClassifier.fit` 协调以下阶段：

1. 要求 `geometry` 或 `edge_index` 至少其一。
2. 校验搜索预算。
3. 重置特征/标签索引并校验显式有序目标契约。
4. 用随机、外部或空间分组指派解析不相交的训练/验证/测试切分。
5. 仅在训练行上拟合地理探测器选择器并冻结其离散化元数据。
6. 用冻结的选择器变换所有切分。
7. 在每个切分内独立构建几何图候选，或过滤并局部重索引显式边。
8. 在训练集上训练图/模型候选，仅按验证准确率选择；无边的失败候选保持可审计，不中止可行候选。
9. 在独立测试切分上一次性评估选定候选。
10. 在全图上重拟合部署估计器并暴露验证/测试报告。

关键设计点：特征发现只用训练行，然后把存储的离散化元数据应用到验证、测试、全量与未来预测帧。几何图也按切分独立构建，因此非训练几何无法改变训练拓扑。验证指标选择候选；独立测试指标从不回馈选择。

## 直接估计器工作流

`GIGNNClassifier.fit` 执行更窄的工作流：

1. 校验训练参数与资源上限。
2. 从显式 `edge_index` 或几何解析图边。
3. 从源/目标特征差计算高斯边属性并聚合成标量边权。
4. 把 `X`、`y`、边、`edge_attr`、`edge_weight` 转换为 PyG `Data`。
5. 用配置的隐藏层与 `num_neighbors_list` 构建 notebook 对齐的 `GIGNN`。
6. 用类加权交叉熵与 Adam 训练。
7. 存储模型、标签编码器、特征元数据、类别与训练损失。

`predict_proba` 校验特征列与训练列完全一致，解析预测图边，运行模型并返回 softmax 概率。

## 模块边界

| 层 | 模块 | 边界 |
| --- | --- | --- |
| 校验 | `gignn/validation.py` | 纯输入校验与标签编码。 |
| 特征发现 | `geodetector/*` | 离散化与地理探测器统计。 |
| 图构建 | `gignn/graph/builders.py` | 边校验与几何到边转换。 |
| PyG 桥 | `gignn/graph/data.py` | `torch` 与 `torch_geometric.data.Data` 的依赖边界。 |
| 模型 | `gignn/models/gignn.py`、`gignn/models/graphsage.py` | 仅 notebook 对齐加权多跳 GIGNN 与兼容 GraphSAGE 定义。 |
| 训练 | `training/*` | 优化器循环、类权重、种子控制。 |
| 估计 | `gignn/estimators.py` | 直接面向用户的分类器生命周期。 |
| 编排 | `gignn/auto.py`、`gignn/splitting.py`、`gignn/ordinal.py` | 三路切分、显式有序语义、图/模型搜索与报告。 |
| 诊断 | `gignn/generalization.py` | 数据集标准化与预测分析。 |
| 输出 | `gignn/plotting.py`、`gignn/reports.py` | 面向人的表格、报告与图。 |
| 持久化 | `gignn/io/persistence.py`、`gignn/auto_archive.py`、`gignn/auto_persistence.py` | 直接估计器 weights-only 检查点 + 有界 ZIP/JSON/NPY 自动工作流归档、模式/版本守卫与状态重建。 |

## 依赖边界

核心依赖声明在 `pyproject.toml`：

- NumPy 与 pandas 用于表格/数组。
- SciPy 与 scikit-learn 用于地理探测器 p 值、指标与切分。
- PyTorch 与 PyTorch Geometric 用于 GIGNN 训练与 PyG 数据。
- Matplotlib 与 NetworkX 用于绘图。

可选 GIS 依赖独立：

- GeoPandas
- Shapely

库不要求环境变量、数据库、HTTP 服务器、Docker 或后台 worker。

## 持久化边界

`GIGNNClassifier.save/load` 通过既有 PyTorch 检查点通道持久化低层模型，并在 CPU 上以 `torch.load(..., weights_only=True)` 加载。`GIGNNAutoClassifier.save/load` 使用独立的 `.gignn` ZIP 容器：元数据与表格/报告值是 JSON，数值数组与模型张量是以 `allow_pickle=False` 加载的 NPY 成员，每个数组绑定声明的 dtype、形状、元素数、字节长度与 SHA-256。自动归档还恢复冻结的选择器、有序类顺序、切分索引、选定图配置、验证/测试报告、训练损失与拟合的低层权重。

两条路径都在重建前强制文件尺寸、模式、模型资源与版本守卫。自动路径额外拒绝符号链接文件、重复或意外 ZIP 成员、不支持的压缩、过大的解压尺寸/成员数、非法 NPY 头、未被引用的数组与包主/次版本不兼容。

检查点路径是本地文件系统输入，通过服务暴露模型加载/保存的应用应约束它们。
