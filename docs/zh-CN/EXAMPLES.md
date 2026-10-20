# 示例

仓库当前包含一个内置的可执行示例。

该示例走当前与 notebook 对齐的 GIGNN 估计器路径，包括自动工作流使用 `GIGNNClassifier` 时对 `num_neighbors_list` 的自动选择。

## `examples/quickstart.py`

目的：

- 构建一个小的数值特征表。
- 创建简单的类点几何对象。
- 标准化空间数据集。
- 拟合 `GIGNNAutoClassifier`。
- 运行情景模拟。
- 在训练行上预测。
- 构建邻接结构与高斯边属性。
- 计算泛化诊断与准确率表。
- 生成直方图、训练曲线与准确率对比图。

运行：

```bash
.venv/bin/python examples/quickstart.py
```

预期输出包含类似以下的行：

```text
selected features: [...]
selected graph: knn
best model params: {...}
generalization accuracy: ...
standardized features: [...]
adjacency nodes: ... edge attributes: (...)
predictions: [...]
changed in simulation: [...]
bias types: [...]
plotting: generated histogram, training-curve, and accuracy-comparison figures
```

quickstart 不保存图片文件，它在内存中演练绘图的返回路径。

## 扩展示例

新增示例时，保持每个示例聚焦一个用户故事：

| 示例类型 | 建议聚焦 |
| --- | --- |
| 自动工作流 | `GIGNNAutoClassifier.fit`、报告、预测、模拟。 |
| 显式图 | `edge_index` 输入与 `GIGNNClassifier`。 |
| 仅地理探测器 | 离散化、因子 q 排序、交互、选择器变换。 |
| 泛化 | `standardize_spatial_dataset`、诊断、偏差图。 |
| 绘图 | 图片生成与可选 `output_path`。 |

示例应使用小的合成数据，除非有意引入并获得授权的真实数据。
