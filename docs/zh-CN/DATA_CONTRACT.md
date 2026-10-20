# 数据契约

本文档定义当前代码库强制的输入与边界契约。

## 特征表

`X` 必须是非空 pandas `DataFrame`。

要求：

- 至少一行与一个特征列。
- 每个特征列必须能用 `pandas.to_numeric` 转换为数值。
- 所有特征值必须有限。
- `GIGNNClassifier` 的预测输入必须使用与训练输入相同的特征列且顺序一致。

GIGNN 在校验前复制输入帧，不会有意修改用户提供的 `DataFrame` 对象。

研究产物导出默认把原始表视为敏感数据。`export_research_artifacts(..., include_raw=False)` 记录输入模式、行/列摘要与 SHA-256 指纹，但不写 `raw/raw_frame.csv`。仅当把源表复制进产物包对研究工作流可接受时才设 `include_raw=True`。指纹与可选的原始 CSV 输出使用相同的净化规范化 UTF-8 字节与 `\n` 行尾。

## 标签

`y` 可以是 pandas `Series` 或其他可迭代对象。

要求：

- 长度必须与 `X` 一致。
- 必须非空。
- 不得含缺失值。
- 值可以是字符串或非连续整数。

标签内部用 `LabelEncoder` 编码。编码器是不可变风格：先 `encoder = LabelEncoder().fit(y)` 再 `encoder.transform(y)`。

## 几何

未显式提供图边时必须提供几何。

对 `knn` 与 `distance` 图策略，每个元素必须暴露以下之一：

- `item.x` 与 `item.y`
- `item.centroid.x` 与 `item.centroid.y`

解析出的坐标必须是有限数值。NaN 与无穷会以 `ValidationError` 拒绝。

对 `contiguity`，几何对象应暴露可调用的 `touches(other)` 或 `intersects(other)` 方法。

提供 `num_nodes` 时，几何长度必须与行数一致。

## 边索引

`edge_index` 是 PyG 风格有向边表。

接受形状：

- `(2, num_edges)`
- `(num_edges, 2)`

要求：

- 二维类数组输入。
- 整数节点索引。
- 至少一条边。
- 节点索引在 `[0, num_nodes - 1]`。
- 边数不超过 `MAX_EDGES`。

同时提供 `edge_index` 与 `geometry` 时，直接图解析中显式 `edge_index` 校验优先。自动工作流中，显式边成为一个图候选，几何派生图也可能被评估。

## 图策略

| 策略 | 必需参数 | 行为 |
| --- | --- | --- |
| `knn` | `k >= 1` | 每个节点按质心欧氏距离连接到 `min(k, n-1)` 个最近的其他节点。边是有向的；等距候选按目标索引升序解析。常规路径只查询 `k+1` 个邻居，仅扩展边界距离并列集。 |
| `distance` | `threshold > 0` | 每个节点与质心欧氏距离 `<= threshold` 的其他节点双向连接。排除自环。 |
| `contiguity` | 带 `touches` 或 `intersects` 的几何 | 为相接或相交的几何发出有向边。 |

图构建未产出边时抛出 `ValidationError`。

## 边属性与边权

`GIGNNClassifier` 从当前拟合或预测调用的特征表计算高斯边属性：

```text
edge_attr[e, f] = exp(-((x_source,f - x_target,f)^2) / (2 * sigma^2))
```

默认 `edge_weight_sigma` 为 `2.0`（论文式 (2)），期望输入 GeoDetector 分层编码特征；在原始量纲下核会坍缩，`fit` 发出 `EdgeWeightWarning`。逐特征边属性默认以 `edge_weight_mode="product"` 聚合为标量 `edge_weight`，此时标量权重精确等于 `exp(-||x_i - x_j||^2 / (2 sigma^2))`。支持的聚合模式有 `product`、`mean`、`min`、`max`。

`build_pyg_data(..., edge_attr=..., edge_weight=...)` 在返回的 PyG `Data` 对象上保留这些数组。`edge_attr` 形状必须为 `(num_edges, num_edge_features)` 或 `(num_edges,)`；`edge_weight` 形状必须为 `(num_edges,)`。

## 多跳邻居采样

`GIGNNClassifier.num_neighbors_list` 控制 notebook 风格的逐跳邻居采样。默认 `(6, 12)` 对应 notebook 的 `NUM_NEIGHBORS_LIST = [6, 12]`。

要求：

- 值必须是正整数。
- 长度必须等于 `len(hidden_channels) + 1`。
- 每个值受 `MAX_GRAPH_NEIGHBORS` 上限约束。
- 无邻居的节点采样自身，使预测仍能为每行输入产出一个输出。

## 自动工作流输入

`GIGNNAutoClassifier.fit` 至少要求以下之一：

- `geometry`
- `edge_index`

搜索网格上限与参数级模式在拟合前与加载自动检查点时各校验一次。探测器、图与模型网格拒绝未知键、空候选列表、布尔当整数、非有限数值候选以及超出资源上限的值。`threshold=None` 这类可选图值保持合法并在 `.gignn` 检查点中往返。`max_grid_candidates` 必须是不超过 `MAX_FEATURES` 的非布尔正整数；它在运行时与检查点校验中都按 `detector_candidates + graph_candidates * model_candidates` 约束实际工作量。

`validation_fraction` 与 `test_fraction` 必须为正且和小于一。默认随机模式创建不相交的训练/验证/测试切分；`split_labels` 可提供显式外部切分，`spatial_groups` 把每个分组完整保留在一个切分内。`split_labels` 与 `spatial_groups` 互斥。空间分组不可分割，因此请求的比例可能无法达到；当实际行比例偏差超过 10 个百分点时 GIGNN 发出 `UserWarning`，消息同时包含行比例与分组比例。

非数值有序目标要求 `class_order`，除非 `y` 是有序 pandas categorical 且其声明类别与观测标签完全一致。数值目标默认按数值升序。无序字符串以及不完整或重复的类顺序被拒绝。

几何派生候选中，`knn`、`distance`、`contiguity` 图在每个切分内分别构建。显式边按切分过滤并局部重索引。失败的无边候选被记录并跳过；仅当每个候选都不可用时拟合才失败。

## 预测输入

对 `GIGNNAutoClassifier.predict`、`predict_proba`、`simulate`：

- 提供 `edge_index` 时，预测使用它。
- 未提供 `edge_index` 且拟合选定的策略不是 `edge_index` 时，用 `geometry` 以选定图策略重建预测边。
- 拟合选定策略是 `edge_index` 时，单独的 `geometry` 不替换拟合图；除非显式提供新 `edge_index`，复用存储的 `edge_index_`。
- `simulate` 要求基线帧与情景帧行数、特征列顺序、pandas 行索引一致。拟合工作流有 `node_ids` 时，调用方还必须提供与拟合完全一致的 ID 顺序。
- 情景拓扑以规范化有向边多重集比较，因此仅边列顺序不同不改变拓扑同一性。

复用存储边索引仅在预测行与训练图兼容时合适。同时重置两帧行索引且不提供 `node_ids` 会丢失库无法重建的行同一性信息；要求同一性强制的调用方必须带显式 ID 拟合与模拟。

## 泛化诊断

`generalization_diagnostics` 要求观测标签与预测标签等长且无缺失值。

偏差计算为（论文图 10 约定）：

```text
Bias = predicted_code - observed_code
```

偏差标签生成规则：

- 偏差为零时 `Correct`。
- 偏差为正时 `Overestimated (...)`。
- 偏差为负时 `Underestimated (...)`。

标签有序、混合类型或不可安全排序时提供 `class_order`。传入 `edge_index` 时，`generalization_diagnostics` 还会用 `morans_i` 检验偏差的空间自相关（结果在 `GeneralizationDiagnostics.bias_moran`）；`morans_i(values, edge_index)` 要求至少三个有限观测，且边索引至少连接两个不同节点。

## 资源上限

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `MAX_AUTO_GRAPH_NODES` | `1_000` | 防止意外的平方级自动图构建。 |
| `MAX_EDGES` | `1_000_000` | 边索引规模上限。 |
| `MAX_FEATURES` | `10_000` | 检查点特征数上限。 |
| `MAX_CLASSES` | `10_000` | 检查点类数上限。 |
| `MAX_GRAPH_NEIGHBORS` | `1_000` | KNN 邻居数上限。 |
| `MAX_HIDDEN_CHANNEL` | `4_096` | 隐藏宽度上限。 |
| `MAX_HIDDEN_LAYERS` | `8` | 隐藏深度上限。 |
| `MAX_TRAINING_EPOCHS` | `10_000` | 训练循环长度上限。 |
| `MAX_CHECKPOINT_BYTES` | `100 MiB` | 直接与自动检查点文件大小上限；自动归档还限制解压成员总大小。 |
| `MAX_CHECKPOINT_TENSORS` | `128` | state-dict 张量数上限。 |
| `MAX_CHECKPOINT_TENSOR_ELEMENTS` | `10_000_000` | 检查点张量总元素上限。 |
