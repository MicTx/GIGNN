# 功能总览

本文档从当前代码库映射 GIGNN 能力。按用户可见层组织，而非源文件顺序。

关于与 notebook GIGNN 核心的实现完整度审计，见[实现状态](IMPLEMENTATION_STATUS.md)。

## 1. 数据准备

### 1.1 空间数据集标准化

API：`gignn.standardize_spatial_dataset`

用途：

- 接受含目标列、可选 ID、可选几何与特征列的 pandas `DataFrame`。
- 显式解析特征列，或通过排除目标/ID/几何列解析。
- 把特征转换为有限数值。
- 校验前可选填充缺失特征值。
- 返回描述标准化数据与预处理决策的不可变 dataclass。

输出：

- `StandardizedSpatialDataset.frame`：完整帧的重置索引副本。
- `StandardizedSpatialDataset.features`：数值特征表。
- `StandardizedSpatialDataset.target`：拒绝缺失标签的目标序列。
- `StandardizedSpatialDataset.ids`：可选 ID 序列。
- `StandardizedSpatialDataset.geometry`：可选几何序列。
- `SpatialDatasetReport`：选定列、行数与缺失值填充计数。

### 1.2 输入校验

API：`gignn.validation.validate_feature_table`、`validate_X_y`、`validate_edge_index`、`LabelEncoder`

用途：

- 强制非空有限数值特征表。
- 强制 `X` 与 `y` 长度对齐。
- 拒绝缺失标签。
- 从 `(2, num_edges)` 或 `(num_edges, 2)` 规范化 PyG 风格边索引。
- 把标签编码为连续类索引并把预测解码回原始标签。

## 2. 地理探测器特征发现

### 2.1 空间离散化

API：`gignn.geodetector.spatial_discretization`

支持方法：

- `quantiles`
- `equal_interval`
- `natural_breaks`
- `fisher_jenks`
- `maximum_breaks`
- `box_plot`
- `std_mean`

当前实现校验数值有限输入并返回带方法与分箱的 `DiscretizationMetadata`。`natural_breaks`/`fisher_jenks` 使用包内精确 Fisher-Jenks 动态规划；`maximum_breaks` 是独立的最宽间隙法。

### 2.2 因子探测

API：`gignn.geodetector.factor_detector`、`geodetector_q`

用途：

- 为候选因子计算地理探测器 q 统计。
- 把非数值目标转换为类别编码。
- 为 q 统计计算近似 F 检验 p 值。
- 可选按 `alpha` 过滤因子。

### 2.3 交互探测

API：`gignn.geodetector.interaction_detector`

用途：

- 评估成对因子交互。
- 报告单个 q 值与合并交互 q 值。
- 把交互分类为 `nonlinear_enhance`、`bi_enhance`、`independent` 或 `weaken`。

### 2.4 特征选择

API：`gignn.geodetector.GeoDetectorSelector`

用途：

- 在训练特征上拟合离散化元数据。
- 用地理探测器分数给因子排序 —— `strategy="interaction"`（默认）按论文 §2.2.1/图 8 在前 `top_interactions` 条交互网络中以度中心性排序；`strategy="factor"` 仅按因子 q 排序。
- 选出全部显著因子或 `top_k` 个因子。
- 用存储的离散化元数据与选定特征变换未来数据。

## 3. 图构建

### 3.1 显式边校验

API：`gignn.graph.build_spatial_edges(edge_index=..., num_nodes=...)`

用途：

- 校验用户提供的图边。
- 接受 `(2, num_edges)` 或 `(num_edges, 2)` 布局。
- 强制整数节点索引、非空边、范围内节点与最大边数限制。

### 3.2 基于几何的图

API：`gignn.graph.build_spatial_edges(geometry=..., strategy=...)`

策略：

- `knn`：按质心坐标把每个节点连接到 `k` 个最近邻。
- `distance`：把正距离阈值内的节点相互连接。
- `contiguity`：连接 `touches` 或 `intersects` 方法返回真的几何。

`knn` 与 `distance` 策略要求几何对象暴露数值 `x/y` 或 `centroid.x/centroid.y`。自动图构建受 `MAX_AUTO_GRAPH_NODES` 守卫。

### 3.3 图工具

API：

- `build_adjacency_dict`：把边索引转换为去重邻接字典。
- `calculate_gaussian_edge_attributes`：从源/目标节点特征差计算高斯边属性。
- `calculate_edge_weights`：把逐特征高斯属性聚合为标量边权。
- `filter_isolated_nodes`：移除孤立节点并返回保留节点索引。
- `build_pyg_data`：构建带 `x`、`y`、`edge_index`、可选 `edge_attr`、可选 `edge_weight` 与标签编码器的 `torch_geometric.data.Data`。

## 4. GNN 模型与训练

### 4.1 notebook 对齐 GIGNN

内部模型：`gignn.models.GIGNN`

用途：

- 用 `NeighborAggregator`、`GCN`、`GIGNN` 模块匹配核心 notebook 结构。
- 用 `num_neighbors_list` 表达逐跳邻居采样，如 notebook 的 `[6, 12]`。
- 邻居聚合时消费高斯边权。
- 返回节点分类的输出 logits。

`gignn.models.GraphSAGEClassifier` 仍作为更低层的兼容模型可用，但 `GIGNNClassifier` 现在默认使用加权多跳 `GIGNN`。

这是 notebook 核心实现边界。不声称完整复现每个论文数据集、基线实验、图或保存产物。

### 4.2 训练工具

API：`TrainingConfig`、`TrainingResult`、`train_node_classifier`、`calculate_class_weights`、`set_random_seed`

用途：

- 训练节点分类器固定 epoch 数。
- 使用类平衡交叉熵权重。
- 使用可配置学习率与权重衰减的 Adam 优化器。
- 提供 `random_state` 时为 Python、NumPy 与 Torch 播种。
- 返回逐 epoch 训练损失。

## 5. 估计器

### 5.1 `GIGNNClassifier`

用途：

- 面向已知道图与模型配置用户的直接 sklearn 风格估计器。
- 处理图解析、高斯边属性/边权、PyG 转换、加权多跳 GIGNN 训练、预测、概率预测、评分与检查点保存/加载。

主要方法：

- `fit(X, y, geometry=None, edge_index=None)`
- `predict(X, geometry=None, edge_index=None)`
- `predict_proba(X, geometry=None, edge_index=None)`
- `score(X, y, geometry=None, edge_index=None)`
- `save(path)`
- `load(path)`

学习属性：

- `model_`
- `label_encoder_`
- `n_features_in_`
- `feature_names_in_`
- `classes_`
- `training_result_`

### 5.2 `GIGNNAutoClassifier`

用途：

- 搜索特征发现参数、图策略与模型参数的高层工作流估计器。
- 拟合一个选定的低层 `GIGNNClassifier`。
- 为每个主要决策阶段发出报告。

主要阶段：

1. 校验 `geometry` 或 `edge_index` 可用。
2. 按随机留出、显式外部标签或不可分割空间分组解析不相交训练/验证/测试切分。
3. 仅在训练行上拟合有序目标编码与地理探测器元数据。
4. 为每个几何候选独立构建训练/验证/测试图；按切分过滤并局部重索引显式边。
5. 用 `validation_report_` 选择图/模型候选，用 `generalization_report_` 一次性评估选定候选，然后在全图上拟合部署估计器。
6. 存储选定特征、选定边索引、估计器、报告与有界保存/加载状态。
7. 支持预测、概率预测与同一性/拓扑锁定的情景模拟。

学习属性：

- `selector_`
- `estimator_`
- `selected_features_`
- `edge_index_`
- `discovery_report_`
- `graph_report_`
- `model_selection_report_`
- `validation_report_`
- `generalization_report_`

## 6. 报告与诊断

### 6.1 自动工作流报告

Dataclass：

- `SpatialRelationshipReport`
- `GraphSelectionReport`
- `ModelSelectionReport`
- `GeneralizationReport`
- `SimulationReport`

这些报告暴露特征选择、图选择、模型选择、验证指标与情景变化。

### 6.2 泛化诊断

API：`generalization_diagnostics`、`model_accuracy_table`

用途：

- 比较观测与预测标签。
- 产出准确率、分类报告与偏差表；偏差为 `predicted_code - observed_code`（正值为高估，论文图 10）。
- 基于有序类编码把偏差标为 `Correct`、`Overestimated (...)` 或 `Underestimated (...)`。
- 用 `morans_i` 检验残差空间自相关（`edge_index` 同时为诊断添加 `bias_moran`）。
- 用 `cross_validate` 做直推式 k 折评估，汇合折外预测及其 Moran's I。
- 为绘图或报告构建准确率对比表。

## 7. 绘图

顶层绘图导出从 `gignn.plotting` 惰性加载。

函数：

- `plot_classification_map`
- `plot_prediction_comparison`
- `plot_observed_prediction_bias`
- `plot_accuracy_comparison`
- `plot_discretization_histogram`
- `plot_training_curves`
- `plot_metric_boxplot`
- `plot_interaction_network`

全部绘图函数校验必需列、返回 Matplotlib 对象，且仅在提供 `output_path` 时写文件。

## 8. 持久化与安全上限

### 8.1 检查点

API：`GIGNNClassifier.save`、`GIGNNClassifier.load`、`GIGNNAutoClassifier.save`、`GIGNNAutoClassifier.load`

用途：

- 存储估计器参数、state dict、类别、特征数与特征名。
- 在 CPU 上以 `torch.load(..., weights_only=True)` 加载直接估计器检查点。
- 把自动工作流存储为有界 `.gignn` ZIP 归档，含 JSON 元数据与以 `allow_pickle=False` 加载的 NPY 数组。
- 重建前校验文件与解压尺寸、归档成员、数组哈希/dtype/形状、模式与包版本、类/特征数、张量数与张量尺寸。

### 8.2 资源上限

关键守卫：

- `MAX_AUTO_GRAPH_NODES = 1_000`
- `MAX_EDGES = 1_000_000`
- `MAX_FEATURES = 10_000`
- `MAX_CLASSES = 10_000`
- `MAX_GRAPH_NEIGHBORS = 1_000`
- `MAX_HIDDEN_CHANNEL = 4_096`
- `MAX_HIDDEN_LAYERS = 8`
- `MAX_TRAINING_EPOCHS = 10_000`
- `MAX_CHECKPOINT_BYTES = 100 MiB`
- `MAX_CHECKPOINT_TENSORS = 128`
- `MAX_CHECKPOINT_TENSOR_ELEMENTS = 10_000_000`
