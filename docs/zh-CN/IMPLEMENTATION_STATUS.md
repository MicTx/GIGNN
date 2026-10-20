# GIGNN 实现状态

本文档将包与已发表论文 —— Ma, X., Zhang, Z., & Song, Y. (2025). Geographically informed graph neural networks. *Spatial Statistics*, 69, 100920. https://doi.org/10.1016/j.spasta.2025.100920 —— 以及参考 notebook `research/notebooks/space-prediction-generalization.ipynb`（保留了案例研究的推理阶段代码）逐项核对。

## 简短回答

包实现了论文所*描述*的 GIGNN 机制的完整链路：

- GeoDetector 因子探测器与交互探测器（式 (1)）、Jenks 自然断点，以及 §2.2.1 / 图 8 的交互网络因子选择。
- 作为节点特征的分层重量化（编码 1..n）（§2.2.2）。
- `sigma = 2` 的高斯核边属性，作为缩放邻居消息的边权进入聚合（式 (2)、式 (4)）。
- GraphSAGE 风格的 `NeighborAggregator` + `GCN` 层（式 (3)-(5)），`num_neighbors_list=(6, 12)`，128 维单隐层，训练时逐批重采样邻居，推理时确定性全邻居聚合。
- 直推式掩码训练（`train_mask`），以及 §5.2.2 的残差诊断：偏差 = 预测码 − 观测码（图 10 约定）与 Moran's I（与 `esda.Moran` 精确一致）。

按范围不实现：

- 论文指标数值的复现（2016 年 84.1% 准确率、2021 年 81.0% 泛化率、Moran's I = 0.089/0.014）：依赖未分发的 Greater Perth SA2/ABS/VIIRS 数据集。
- 仿真数据生成器（式 (6)-(8)）、GAT/GIN/ChebNet/MPNN/RF 基线，以及 k 折驱动器（`fit(..., train_mask=...)` 支持「五轮 × 1000 轮次」的协议，但没有打包驱动函数）。
- 原始数据摄取、托管服务、自动发布 —— 见[当前边界](#当前边界)。

## 论文-代码对照矩阵

| 论文要素 | 实现 | 证据 | 状态 / 注意 |
| --- | --- | --- | --- |
| §2.2.1 / 式 (1) GeoDetector q、因子探测器 | `geodetector/factor.py`：`geodetector_q`、`factor_detector`（ANOVA F 检验 p 值）；`target_to_numeric` 只接受数值型与*有序*分类目标 | `tests/test_geodetector.py` | 已实现 |
| 式 (1) Jenks 自然断点，Xs 分 5 层 | `discretization.py`：精确 Fisher–Jenks 动态规划（`natural_breaks`/`fisher_jenks`）；`maximum_breaks` 是独立的最宽间隙法。`GeoDetectorSelector` 默认 `discretizer="natural_breaks"`、`max_bins=5` | `tests/test_geodetector.py` | 已实现；DP 是确定性的 —— notebook 的 `mapclassify.NaturalBreaks` 用随机启发式优化同一目标，断点可能略有差异 |
| §2.2.2 重量化，编码 1..n | `spatial_discretization` 按区间位置升序标注 1..n；Jenks 分层按取值有序，因此对 Jenks 与论文「按组级统计量升序」的重标注等价 | `tests/test_geodetector.py` | 已实现 |
| §2.2.1 交互探测器 | `interaction_detector`：`q_interaction`、`p_value_interaction` 与带数值容差的标准五型分类 | `tests/test_geodetector.py` | 已实现 |
| §4.2 / 图 8 交互网络选择（「交互检测网络中高中心度与高 Q 值的前三个关键因子」；网络 = GD 前 10 条交互） | `GeoDetectorSelector(strategy="interaction", top_interactions=10, top_k=...)`：按因子在前 10 交互网络中的度排序，度相同按因子 q | `tests/test_geodetector.py` | 已实现；论文对「高中心度与高 Q 值」语焉不详 —— 实现为度降序再 q 降序；复现案例用 `top_k=3` |
| §2.2.2 / 式 (2) 高斯核，sigma=2，作用于重量化特征 | `calculate_gaussian_edge_attributes`（逐特征 `exp(-(Δx_k)²/2σ²)`）+ `calculate_edge_weights(mode="product")` 精确等于 `exp(-‖xi−xj‖²/2σ²)`；`mean`/`min`/`max` 为非论文备选。边权趋零时发出 `EdgeWeightWarning` | `tests/test_model_training.py::test_eq2_product_edge_weight_equals_vector_gaussian_kernel` | 已实现；要求分层编码特征 —— 应输入 `GeoDetectorSelector` 的输出而非原始量纲 |
| §4.2 步骤 1 拓扑（notebook：buffer(100)+intersects；仿真 §3：六边形 queen 邻接） | `build_spatial_edges`：`knn`（默认）、`distance`、`contiguity`（touches/intersects），或用户 `edge_index` | `tests/test_graph_builders.py` | 泛化实现：notebook 的缓冲邻接可用缓冲后的 `geometry`+`contiguity` 或显式 `edge_index` 复现；没有 `buffer` 便捷封装 |
| §2.2.3 / 式 (3)-(5) NA + GCN 层、k 阶聚合 | `models/gignn.py`：`NeighborAggregator`、`GCN`、`GIGNN`；`normalize_edge_weights=False` 时的非归一加权均值 `A(v) = (1/\|N\|)Σ w_uv h_u` | `tests/test_model_training.py::test_gignn_forward_uses_edge_weights`、`test_unnormalized_mean_limits_increment_of_dissimilar_neighbors` | 已实现 —— 且比 notebook 更严格：见下文「分歧」 |
| notebook 超参：`HIDDEN_DIM=[128, out]`、`NUM_NEIGHBORS_LIST=[6,12]`、LR 0.01、EPOCHS 1000、batch 64 | `GIGNNClassifier` 默认 `hidden_channels=(128,)`、`num_neighbors_list=(6, 12)`、`learning_rate=0.01`；`max_epochs` 默认 100（论文取值可经参数设置） | `tests/test_estimator.py` | 已实现；未打包逐节点小批量 —— 训练为全图 |
| §5.2.1 五折交叉验证协议 | 直推式 `fit(..., train_mask=...)`：所有节点留在图中，仅掩码内标签进入损失 | `tests/test_model_training.py::test_train_node_classifier_ignores_labels_outside_mask` | 机制已实现；未打包折驱动器。注意：在折外先对全部标签运行 `GeoDetectorSelector` 会把标签信息泄漏进每一折 —— 论文工作流正是如此；折内选择需自行编排 |
| §5.2.2 / 图 10 偏差分析 | `generalization_diagnostics`：偏差 = 预测码 − 观测码，正值表示高估 | `tests/test_generalization_standard.py` | 已实现；与 notebook 存在分歧 —— 见下文 |
| §5.2.2 残差 Moran's I | `morans_i(values, edge_index)`：行标准化邻接权重，正态与置换推断 | `tests/test_generalization_standard.py`（与 `esda.Moran` 2.9.0 参考值完全一致） | 已实现 |

## 论文、notebook 与包之间的分歧

以下是论文/notebook 证据内部被发现的不一致。包遵循*论文所述设计*，这些分歧均非照抄：

1. **边属性未进入已发表模型。** notebook 计算了 `edge_attr` 与 `edge_attr_diff` 并存入 `Data`，但 `NeighborAggregator.forward(neighbor_feature)` 没有权重参数 —— 聚合是无权均值。包按论文所述用式 (2) 权重缩放邻居消息。
2. **notebook 推理是随机的。** `multihop_sampling` 在 `torch.no_grad()` 内不带种子运行，因此报告的 84.1%/81.0% 是采样估计量的一次取值。包默认确定性全邻居推理（`inference="full"`），即该估计量的期望；`inference="sampled"` 复现 notebook 行为。
3. **偏差符号。** notebook 计算 `观测 − 预测` 却把正值标注为「Overestimated」，而论文图 10 注释定义正偏差为 `预测 − 观测`。按图 10 自身定义，报告的「25 个高估 / 9 个低估」应对调。包实现图 10 约定。
4. **训练循环未保留在 notebook 中。** 单元格 49–53 只*加载*检查点；优化器、损失函数与类权重进入损失的方式未保存在该产物中。包使用 Adam + 类加权交叉熵（权重按逆类别分布，同单元格 43 的计算）。
5. **基线图与目标图不一致。** 目标模型用缓冲相交邻接，对照模型用另一套 `edge_index`，使论文表格中的模型间比较存在混淆变量。

## 当前边界

已实现的核心适用于面向对齐空间单元指标表的本地研究工作流。用户仍需提供清洗后的特征表、标签、几何或图边。

当前包范围外：

- 原始影像摄取与远程数据获取
- 已发表的案例数据集与检查点文件
- 打包的 k 折驱动器与式 (6)-(8) 仿真生成器
- notebook 的确切文件系统路径
- 托管的推理或训练服务
- 自动包发布

## 回归命令

在变更 GIGNN 核心对齐相关的实现或文档时使用以下命令：

```bash
.venv/bin/python -m pytest tests/test_geodetector.py tests/test_graph_builders.py tests/test_model_training.py tests/test_estimator.py tests/test_auto.py tests/test_generalization_standard.py
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
.venv/bin/python -m black --check gignn tests examples
```
