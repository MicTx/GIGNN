# 用户指南

本指南展示 GIGNN 的预期用户工作流：准备空间单元表、构建或提供图、训练分类器、检查报告、模拟特征变化并可视化输出。

当前包实现了 notebook GIGNN 核心。不声称完整复现每个论文实验或外部数据集；见[实现状态](IMPLEMENTATION_STATUS.md)。

## 1. 准备输入

每行必须代表一个空间单元、样点、格网或区域。特征列必须是数值且有限。

```python
import pandas as pd

X = pd.DataFrame(
    {
        "ndvi": [0.12, 0.44, 0.51, 0.83],
        "dem": [12.0, 19.0, 21.0, 35.0],
        "road_density": [1.0, 2.5, 2.1, 4.2],
    }
)
y = pd.Series(["low", "medium", "medium", "high"])
```

标签可以是字符串或非连续整数。GIGNN 内部编码标签并以原始标签空间返回预测。

## 2. 提供空间结构

GIGNN 需要图结构。可以显式提供或从几何派生。

### 显式 `edge_index`

```python
import numpy as np

edge_index = np.array(
    [
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ]
)
```

`edge_index` 形状可以是 `(2, num_edges)` 或 `(num_edges, 2)`。节点索引必须是整数、非负且在范围内。

### 几何

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self

geometry = [Point(index, 0) for index in range(len(X))]
```

对 `knn` 与 `distance` 图，每个几何元素必须暴露有限的数值 `x/y` 或 `centroid.x/centroid.y`。KNN 平局按行索引升序解析；距离图包含恰好等于阈值的点对并发出双向边。对 `contiguity`，几何对象必须提供 `touches` 或 `intersects` 方法。

## 3. 使用自动工作流

想让 GIGNN 选择特征、图策略与模型参数时，`GIGNNAutoClassifier` 是推荐入口。

```python
from gignn import GIGNNAutoClassifier

model = GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [3]},
    graph_grid={"strategy": ["knn"], "k": [4]},
    model_grid={"hidden_channels": [(64,)], "learning_rate": [0.01], "max_epochs": [100]},
    top_k_features=None,
    random_state=42,
)

model.fit(
    X,
    y,
    geometry=geometry,
    class_order=["low", "medium", "high"],
)
predictions = model.predict(X, geometry=geometry)
probabilities = model.predict_proba(X, geometry=geometry)
```

拟合后检查这些属性：

| 属性 | 含义 |
| --- | --- |
| `selected_features_` | 地理探测器排序选出的特征。 |
| `edge_index_` | 最终模型选定的边索引。 |
| `discovery_report_` | 特征 q 分、交互、离散化器与选定特征。 |
| `graph_report_` | 图候选、选定策略与选定边索引。 |
| `model_selection_report_` | 候选模型参数与验证准确率。 |
| `validation_report_` | 用于候选选择的仅验证指标。 |
| `generalization_report_` | 独立测试准确率、宏 F1、加权 F1 与逐类指标。 |

## 4. 使用直接估计器

已经知道图与模型配置时用 `GIGNNClassifier`。

```python
from gignn import GIGNNClassifier

classifier = GIGNNClassifier(
    hidden_channels=(128, 64),
    num_neighbors_list=(6, 12, 12),
    graph_builder="knn",
    k_neighbors=4,
    max_epochs=100,
    learning_rate=0.01,
    random_state=42,
)

classifier.fit(X, y, geometry=geometry)
predictions = classifier.predict(X, geometry=geometry)
probabilities = classifier.predict_proba(X, geometry=geometry)
accuracy = classifier.score(X, y, geometry=geometry)
```

`num_neighbors_list` 遵循 notebook 的逐跳邻居采样语义。默认单隐藏层模型为 `(6, 12)`；如上配置两个隐藏层时提供三个跳数。

要求每次调用重建相同采样邻域时使用整数 `random_state`。`random_state=None` 时采样消耗 NumPy 全局随机流；连续调用可能不同，而 `set_random_seed(seed)` 可重放序列。

直接估计器校验预测特征列与训练列完全一致，包括顺序。

## 5. 运行情景模拟

`simulate` 用同一拟合的选择器与估计器比较基线与情景特征。

```python
scenario = X.copy()
scenario["ndvi"] = scenario["ndvi"] + 0.1

simulation = model.simulate(scenario, baseline_X=X, geometry=geometry)

simulation.baseline_predictions
simulation.scenario_predictions
simulation.probability_delta
simulation.changed_mask
```

`probability_delta` 是 `scenario_probabilities - baseline_probabilities`。`changed_mask` 在预测类变化处为真。

## 6. 标准化空间数据集

```python
from gignn import standardize_spatial_dataset

dataset, report = standardize_spatial_dataset(
    geo_df,
    id_column="SA2_CODE21",
    target_column="Urbanization level",
    geometry_column="geometry",
    fill_missing=0.0,
)

X = dataset.features
y = dataset.target
geometry = dataset.geometry
```

帧含非特征的数值列时使用显式 `feature_columns`。

## 7. 诊断泛化

```python
from gignn import generalization_diagnostics, model_accuracy_table

diagnostics = generalization_diagnostics(
    observed=y,
    predicted=predictions,
    class_names={"low": "Low", "medium": "Medium", "high": "High"},
    class_order=["low", "medium", "high"],
)

accuracy_table = model_accuracy_table({"GIGNN": diagnostics.accuracy, "Baseline": 0.5}, year=2021)
```

标签有序或不可自然排序时使用 `class_order`。偏差值按 `predicted_code - observed_code` 计算（正值为高估，论文图 10）。传入 `edge_index` 可额外得到 `diagnostics.bias_moran`，即残差空间自相关的 Moran's I 检验；`morans_i(values, edge_index)` 可单独调用，`cross_validate` 执行直推式 k 折评估。

## 8. 绘制输出

```python
from gignn import plot_accuracy_comparison, plot_discretization_histogram, plot_training_curves

plot_discretization_histogram(X, "ndvi", intervals=[0.3, 0.6])
plot_accuracy_comparison(accuracy_table)
plot_training_curves(training_history)
```

绘图函数返回 Matplotlib 对象。仅在提供 `output_path` 时写文件。

## 9. 保存与加载模型

```python
classifier.save("gignn.pt")
loaded = GIGNNClassifier.load("gignn.pt")

auto_model.save("workflow.gignn")
loaded_auto = GIGNNAutoClassifier.load("workflow.gignn")
```

两个类都强制有界检查点加载。直接估计器使用 weights-only PyTorch 检查点、归一化 NumPy 标量标签/配置值，并在替换目标前拒绝自定义载荷对象。自动工作流使用独立的 ZIP/JSON/NPY 格式，不调用 `torch.load`；它恢复冻结的选择器、有序标签顺序、切分索引、选定图配置、验证/测试报告、训练损失与低层模型权重。运行时与检查点加载使用同一套参数级自动网格校验，包括对可选 `threshold=None` 的支持。
