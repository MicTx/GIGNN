# 公开 API 参考

本参考描述当前代码库中可见的公开与半公开 API。稳定用户入口是顶层 `gignn` 导出加有文档的 `gignn.geodetector` 与 `gignn.graph` 辅助。更低层的 `training`、`models`、`io`、`validation` API 对高级集成有用，但 `1.0` 前可能变化。

关于 notebook GIGNN 核心是否已实现的当前审计，见[实现状态](IMPLEMENTATION_STATUS.md)。

## 顶层 `gignn` 导出

| 名称 | 种类 | 用途 |
| --- | --- | --- |
| `GIGNNAutoClassifier` | Dataclass 估计器 | 搜索特征发现、图策略与模型配置的高层工作流。 |
| `GIGNNClassifier` | Dataclass 估计器 | 带显式图/模型控制的直接 notebook 对齐 GIGNN 节点分类器。 |
| `GIGNN` | Torch 模块 | 匹配 notebook 核心的加权多跳邻居聚合模型。 |
| `multihop_sampling` | 函数 | 从邻接字典与逐跳邻居数构建 notebook 风格采样节点列表。 |
| `standardize_spatial_dataset` | 函数 | 从表格帧构建标准化的空间单元数据集与报告。 |
| `generalization_diagnostics` | 函数 | 计算准确率、预测偏差表、分类报告与可选的偏差 Moran's I。 |
| `morans_i` | 函数 | 图上的全局 Moran's I（语义与 `esda.Moran` 兼容）。 |
| `cross_validate` | 函数 | 直推式 k 折评估，汇出折外预测与可选偏差 Moran's I（论文 §5.2 协议）。 |
| `model_accuracy_table` | 函数 | 把模型准确率值转换为对比表。 |
| `export_research_artifacts` | 函数 | 把拟合的自动工作流导出为 raw/preprocessing/process/results 研究产物包。 |
| `ResearchArtifactBundle` | 冻结 dataclass | 保存产物根、manifest 路径与逻辑文件映射。 |
| `NATURE_PALETTE` | 惰性加载常量 | Nature 风格语义绘图调色板。 |
| `NATURE_COLOR_CYCLE` | 惰性加载常量 | 默认发表绘图颜色循环。 |
| `apply_publication_style` | 惰性加载函数 | 无导入时副作用地应用发表级 Matplotlib rcParams。 |
| `infer_publication_layout` | 惰性加载函数 | 从数据复杂度推断图尺寸、字号、间距、图例位置与标记尺寸。 |
| `add_panel_label` | 惰性加载函数 | 为多面板图添加加粗的轴相对面板标签。 |
| `save_publication_figure` | 惰性加载函数 | 把图保存为 SVG/PNG/PDF 式发表输出。 |
| `plot_classification_map` | 惰性加载函数 | 绘制专题类或预测地图。 |
| `plot_prediction_comparison` | 惰性加载函数 | 并排绘制观测与预测地图。 |
| `plot_observed_prediction_bias` | 惰性加载函数 | 绘制观测、预测与偏差面板。 |
| `plot_accuracy_comparison` | 惰性加载函数 | 绘制模型准确率条形图。 |
| `plot_discretization_histogram` | 惰性加载函数 | 绘制特征直方图与可选离散化区间。 |
| `plot_training_curves` | 惰性加载函数 | 绘制训练损失与训练/测试准确率曲线。 |
| `plot_metric_boxplot` | 惰性加载函数 | 绘制指标分布，可选分组。 |
| `plot_interaction_network` | 惰性加载函数 | 绘制地理探测器因子交互网络。 |
| `StandardizedSpatialDataset` | 冻结 dataclass | 保存标准化帧、特征、目标、ID 与几何。 |
| `SpatialDatasetReport` | 冻结 dataclass | 报告选定列、行数与已填充缺失值。 |
| `GeneralizationDiagnostics` | 冻结 dataclass | 保存准确率、偏差帧、分类报告与可选 `bias_moran`。 |
| `MoranResult` | 冻结 dataclass | Moran's I 统计量、期望值、正态与置换推断。 |
| `CrossValidationReport` | 冻结 dataclass | 各折指标加汇合的折外标签/预测与 `bias_moran`。 |
| `GIGNNError` | 异常 | 包基础异常。 |
| `EdgeWeightWarning` | 警告 | 高斯边权坍缩，邻居不再起作用。 |
| `ValidationError` | 异常 | 输入契约或资源上限违例。 |
| `NotFittedError` | 异常 | 估计器或变换器在拟合前使用。 |

## `GIGNNAutoClassifier`

签名：

```python
GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [3]},
    graph_grid={"strategy": ["knn"], "k": [8]},
    model_grid={"hidden_channels": [(64,)], "learning_rate": [0.01], "max_epochs": [100]},
    top_k_features=None,
    random_state=None,
    validation_fraction=0.2,
    test_fraction=0.2,
    max_grid_candidates=100,
)
```

方法：

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `fit(X, y, *, geometry=None, edge_index=None, split_labels=None, spatial_groups=None, class_order=None, node_ids=None)` | `self` | 要求 `geometry` 或 `edge_index`；以不相交训练/验证/测试切分搜索探测器、图与模型候选。 |
| `predict(X, *, geometry=None, edge_index=None)` | `np.ndarray` | 返回原始标签空间的标签。 |
| `predict_proba(X, *, geometry=None, edge_index=None)` | `np.ndarray` | 返回形状 `(n_rows, n_classes)`。 |
| `simulate(X_scenario, *, baseline_X, geometry=None, edge_index=None, node_ids=None, scenario_edge_index=None)` | `SimulationReport` | 在锁定行同一性与拓扑下比较基线与情景。 |
| `save(path)` | `None` | 把拟合的完整自动工作流保存为含 JSON 元数据与 NPY 数组的有界 `.gignn` ZIP；不使用 pickle 加载器。 |
| `load(path)` | `GIGNNAutoClassifier` | 恢复前校验归档成员、尺寸、哈希、模式/版本、数组与模型状态；然后恢复选择器、有序模式、图/模型配置、报告、训练损失与拟合权重。 |
| `export_research_artifacts(output_dir, **kwargs)` | `ResearchArtifactBundle` | 为拟合的自动工作流写出可复现包。 |

学习属性：

| 属性 | 说明 |
| --- | --- |
| `selector_` | 拟合的 `GeoDetectorSelector`。 |
| `estimator_` | 拟合的低层 `GIGNNClassifier`。 |
| `selected_features_` | 地理探测器排序选出的特征名。 |
| `edge_index_` | 选定的图边索引。 |
| `discovery_report_` | `SpatialRelationshipReport`。 |
| `graph_report_` | `GraphSelectionReport`。 |
| `model_selection_report_` | `ModelSelectionReport`。 |
| `validation_report_` | 仅从验证行计算并用于候选选择的 `GeneralizationReport`；含 `partition="validation"` 与不可变来源索引。 |
| `generalization_report_` | 选择后仅从独立测试行计算的 `GeneralizationReport`；含 `partition="test"` 与不可变来源索引。 |

错误：

- 未提供图输入与几何时抛出 `ValidationError`。
- 未知网格键、非法参数值、非有限数值候选或搜索预算超限时抛出 `ValidationError`。候选工作量按探测器候选加图×模型组合计数，`max_grid_candidates` 在拟合与检查点加载时同样校验。
- 运行时搜索与 `.gignn` 检查点加载使用同一套参数级网格契约；`graph_grid={"threshold": [None]}` 等可选值原样往返。
- 预测方法在 `fit` 前调用时抛出 `NotFittedError`。

## `GIGNNClassifier`

签名：

```python
GIGNNClassifier(
    hidden_channels=(128,),
    num_neighbors_list=(6, 12),
    graph_builder="knn",
    k_neighbors=8,
    distance_threshold=None,
    edge_weight_sigma=2.0,
    edge_weight_mode="product",
    normalize_edge_weights=False,
    inference="full",
    max_epochs=100,
    learning_rate=0.01,
    weight_decay=5e-4,
    dropout=0.0,
    random_state=None,
    device="cpu",
    class_order=None,
)
```

- `edge_weight_mode="product"` 精确复现式 (2)：逐特征高斯核的乘积即 `exp(-||x_i - x_j||^2 / (2 sigma^2))`；`mean`、`min`、`max` 为非论文备选。
- `normalize_edge_weights=False` 按式 (2) 在邻居均值内缩放消息；`True` 再除以权重和（核加权平均）。
- `inference="full"`（默认）为确定性全邻居聚合 —— 采样估计量的期望；`"sampled"` 复现 notebook 的随机 `multihop_sampling` 推理。
- `fit` 接受 `train_mask=`：布尔数组把损失限制在掩码节点内，所有节点仍留在图中（论文 k 折协议的直推式训练）；掩码外标签允许缺失。
- 当边权中位数坍缩到 `1e-3` 以下 —— 典型于把原始量纲而非分层编码特征喂给 `sigma=2` 的核 —— `fit` 发出 `EdgeWeightWarning`；先用 `GeoDetectorSelector` 离散化或增大 `edge_weight_sigma`。

方法：

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `fit(X, y, *, geometry=None, edge_index=None, train_mask=None)` | `self` | 构建或校验图，计算高斯边属性/边权，转换为 PyG 数据，训练加权多跳 GIGNN；`train_mask` 把损失限制在掩码节点。 |
| `predict(X, *, geometry=None, edge_index=None)` | `np.ndarray` | 返回原始标签。 |
| `predict_proba(X, *, geometry=None, edge_index=None)` | `np.ndarray` | Softmax 概率。 |
| `score(X, y, *, geometry=None, edge_index=None)` | `float` | 准确率分。 |
| `save(path)` | `None` | 写检查点。要求已拟合的估计器。 |
| `load(path)` | `GIGNNClassifier` | 加载检查点并校验模式/资源。 |

学习属性：

| 属性 | 说明 |
| --- | --- |
| `training_result_` | 带逐 epoch 损失的 `TrainingResult`。 |
| `model_` | 拟合的 `GIGNN`。 |
| `label_encoder_` | 拟合的不可变风格标签编码器。 |
| `n_features_in_` | 训练特征列数。 |
| `feature_names_in_` | 按序的训练特征列。 |
| `classes_` | 原始标签类。 |

预测要求与训练相同的特征名且顺序一致。

`num_neighbors_list` 遵循 notebook 语义：每个元素是一跳的采样邻居数。默认 `(6, 12)` 对应 notebook 的 `NUM_NEIGHBORS_LIST = [6, 12]`（一个隐藏层加输出层）。长度必须等于 `len(hidden_channels) + 1`。

采样使用两种显式 RNG 模式。整数 `random_state` 在每次调用重建独立的 `np.random.default_rng(seed)`，因此相同种子产出相同邻域。`random_state=None` 消耗 NumPy 全局 RNG 流；连续调用推进流，而 `set_random_seed(seed)` 或 `np.random.seed(seed)` 可重放完整序列。

## 研究产物导出

模块：`gignn.artifacts`

主 API：

```python
from gignn import export_research_artifacts

bundle = export_research_artifacts(
    model,
    "artifacts/run_001",
    raw_frame=X,
    geometry=geometry,
    include_raw=False,
    provenance={"study_area": "example"},
)
```

等价的拟合模型便捷方法：

```python
bundle = model.export_research_artifacts("artifacts/run_001", raw_frame=X, geometry=geometry)
```

导出写出含 `manifest.json`、`README.md` 与四个证据层的目录：

| 层 | 关键文件 |
| --- | --- |
| `raw/` | `input_schema.json`、`input_summary.json`、`input_fingerprint.json`，`include_raw=True` 时的可选 `raw_frame.csv` |
| `preprocessing/` | `feature_summary.csv`、`missing_values.csv`、`fill_summary.json`、`target_distribution.csv`、训练/验证/测试 `split_indices.csv`、`selected_columns.json` |
| `process/` | `factor_scores.csv`、`interaction_scores.csv`、`discretization_bins.csv`、`graph_candidates.csv`、`selected_edges.csv`、`model_candidates.csv`、`training_loss.csv` |
| `results/` | 独立测试 `metrics.json`、`validation_metrics.json`、`per_class_report.csv`、`predictions.csv`、`probabilities.csv`、可选 `simulation_report.csv`、`figures_manifest.csv` |

默认 `include_raw=False`，不写 `raw/raw_frame.csv`；模式、摘要与指纹仍会记录以支持可复现。默认 `overwrite=False`，输出目录已含文件时抛出 `ValidationError`。仅在有意替换既有产物包时使用 `overwrite=True`。

## 地理探测器 API

模块：`gignn.geodetector`

| 名称 | 签名 / 种类 | 用途 |
| --- | --- | --- |
| `spatial_discretization` | `spatial_discretization(data_frame, continuous_columns, method="quantiles", num_intervals=4)` | 离散化连续列并返回元数据。 |
| `factor_detector` | `factor_detector(df, y, factors, alpha=0.05)` | 为因子计算 q 统计与 p 值。 |
| `geodetector_q` | `geodetector_q(y_values, factor_values)` | 计算单个 q 统计。 |
| `interaction_detector` | `interaction_detector(df, y, factors)` | 计算成对交互 q 值与交互类型。 |
| `GeoDetectorSelector` | `GeoDetectorSelector(discretizer="natural_breaks", max_bins=5, top_k=None, alpha=None, strategy="interaction", top_interactions=10)` | 拟合/变换特征选择器。 |
| `DiscretizationMetadata` | 冻结 dataclass | 存储方法与分箱边界。 |

`natural_breaks`、`fisher_jenks` 使用包内确定性 Fisher-Jenks 动态规划（全局最优、已向量化）；`maximum_breaks` 是独立的最宽间隙法。`interaction_detector` 还会报告 `p_value_interaction`（合并分层上的 ANOVA F 检验）。

`GeoDetectorSelector` 默认实现论文 §2.2.1/图 8 的流程：`strategy="interaction"` 按因子在前 `top_interactions` 条交互网络中的度中心性排序、以因子 q 打破并列（复现案例用 `top_k=3`）；`strategy="factor"` 仅按因子 q 排序。拟合属性 `interactions_` 与 `interaction_degree_` 分别保存排序后的交互表与逐因子度。`target_to_numeric` 要求数值或有序分类目标；两类以上的无序非数值目标抛出 `ValidationError`，因为方差型 q 对其无定义。

## 图 API

模块：`gignn.graph`

| 名称 | 签名 / 种类 | 用途 |
| --- | --- | --- |
| `build_spatial_edges` | `build_spatial_edges(geometry=None, *, edge_index=None, num_nodes=None, strategy="knn", k=8, threshold=None, max_nodes=1000, max_edges=1000000)` | 校验显式边或构建有限坐标几何边。KNN 平局按目标索引升序；距离阈值含边界且双向发出。 |
| `build_pyg_data` | `build_pyg_data(X, y, edge_index, *, edge_attr=None, edge_weight=None)` | 构建 `torch_geometric.data.Data`，保留可选边属性与边权。 |
| `build_adjacency_dict` | `build_adjacency_dict(edge_index, *, num_nodes)` | 构建去重的双向邻接字典。 |
| `calculate_gaussian_edge_attributes` | `calculate_gaussian_edge_attributes(node_features, edge_index, *, sigma=2.0)` | 计算高斯源-目标特征属性。 |
| `calculate_edge_weights` | `calculate_edge_weights(edge_attributes, *, mode="product")` | 把逐特征边属性聚合为模型可用的标量边权；`product` 是精确的式 (2) 向量核。 |
| `filter_isolated_nodes` | `filter_isolated_nodes(adjacency_dict)` | 移除孤立节点并返回保留的原始索引。 |

## 泛化 API

| 名称 | 返回 | 用途 |
| --- | --- | --- |
| `standardize_spatial_dataset` | `(StandardizedSpatialDataset, SpatialDatasetReport)` | 规范化空间单元帧以建模。 |
| `generalization_diagnostics` | `GeneralizationDiagnostics` | 比较观测/预测标签并产出偏差诊断；`edge_index` 追加 `bias_moran`。`Bias` 为预测码减观测码 —— 正值表示高估（论文图 10）。 |
| `morans_i` | `MoranResult` | `morans_i(values, edge_index, permutations=999, random_state=None)`；行标准化对称邻接权重，正态假设下 `p_norm`/`z_norm`，置换推断的折叠单尾 `p_sim`/`z_sim`。 |
| `cross_validate` | `CrossValidationReport` | `cross_validate(estimator_factory, X, y, n_splits=5, random_state=None, class_order=None, geometry=None, edge_index=None, moran_permutations=999, factory_kwargs=None)`；直推式分层折、汇合折外预测、可选偏差 Moran's I。 |
| `model_accuracy_table` | `pd.DataFrame` | 从模型准确率值构建绘图/报告表。 |

## 绘图 API

全部绘图函数返回 Matplotlib `Figure`/`Axes` 对象，仅在提供 `output_path` 时保存。发表辅助从 `gignn` 惰性加载，导入时不修改 Matplotlib rcParams。

| 辅助 | 用途 |
| --- | --- |
| `NATURE_PALETTE` | 方法、方向性效果与中性支撑的语义 Nature 风格颜色。 |
| `NATURE_COLOR_CYCLE` | 从 `NATURE_PALETTE` 派生的默认发表颜色顺序。 |
| `apply_publication_style(font_size=12, axes_linewidth=1.2, use_tex=False)` | 应用无衬线字体栈、可编辑 SVG 文本、无框图例与极简边框。 |
| `infer_publication_layout(row_count, category_count=1, panel_count=1, label_lengths=None, geometry_aspect=None)` | 推断自适应 `figsize`、字号、间距、图例位置、标记尺寸与条宽。 |
| `add_panel_label(ax, label, ...)` | 用轴相对坐标添加加粗面板标签。 |
| `save_publication_figure(figure, output_path, formats=("svg", "png"), dpi=300, pad=2, close=False)` | 把一张图保存为多种发表格式并可选关闭。 |

| 函数 | 必需列 / 输入 | 返回 |
| --- | --- | --- |
| `plot_classification_map` | `geo_frame`、`column` | `(Figure, Axes)` |
| `plot_prediction_comparison` | `truth_column`、`prediction_column` | `(Figure, np.ndarray)` |
| `plot_observed_prediction_bias` | `observed_column`、`prediction_column`、`bias_column` | `(Figure, np.ndarray)` |
| `plot_accuracy_comparison` | 准确率表列 | `(Figure, Axes)` |
| `plot_discretization_histogram` | 数据帧与数值列 | `(Figure, Axes)` |
| `plot_training_curves` | 损失、训练准确率、测试准确率列 | `(Figure, np.ndarray)` |
| `plot_metric_boxplot` | 指标/值/分组列 | `(Figure, Axes)` |
| `plot_interaction_network` | 源、目标、权重列 | `(Figure, Axes)` |

## 高级模块

### `gignn.validation`

| 名称 | 用途 |
| --- | --- |
| `validate_feature_table` | 校验有限数值特征帧。 |
| `validate_X_y` | 校验特征/标签对齐。 |
| `validate_edge_index` | 校验并规范化边索引。 |
| `LabelEncoder` | 不可变风格标签编码器。 |

### `gignn.training`

| 名称 | 用途 |
| --- | --- |
| `TrainingConfig` | 冻结训练配置。 |
| `TrainingResult` | 冻结训练损失报告。 |
| `train_node_classifier` | 固定 epoch 的节点分类训练循环。 |
| `calculate_class_weights` | 逆频率类权重。 |
| `set_random_seed` | 为 Python、NumPy 与 Torch 播种。 |

### `gignn.models`

| 名称 | 用途 |
| --- | --- |
| `GIGNN` | notebook 对齐的加权多跳 GIGNN 模块。 |
| `NeighborAggregator` / `GCN` | notebook 对齐 GIGNN 模块的构建块。 |
| `multihop_sampling` | 多跳采样辅助：整数种子用独立 `default_rng(seed)`；`None` 消耗 NumPy 全局 RNG 流。 |
| `GraphSAGEClassifier` | 兼容多层 GraphSAGE 节点分类器。 |

### `gignn.io`

| 名称 | 用途 |
| --- | --- |
| `save_checkpoint` | 归一化 NumPy 标量、写入前拒绝非 weights-only 安全载荷、fsync 临时文件、原子替换目标并在支持时 fsync 父目录。 |
| `load_checkpoint` | 在 CPU 上以 `weights_only=True` 加载检查点，带文件尺寸守卫与稳定最终路径符号链接拒绝。 |

## 报告 Dataclass

字段级细节见[报告](REPORTS.md)：

- `SpatialRelationshipReport`
- `GraphSelectionReport`
- `ModelSelectionReport`
- `GeneralizationReport`
- `SimulationReport`
