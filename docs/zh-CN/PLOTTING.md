# 绘图

GIGNN 绘图函数经顶层惰性加载暴露。导入 `gignn` 不会立即导入绘图模块。

```python
from gignn import (
    NATURE_PALETTE,
    add_panel_label,
    apply_publication_style,
    infer_publication_layout,
    plot_classification_map,
    plot_accuracy_comparison,
    save_publication_figure,
)
```

全部绘图函数：

- 校验必需的输入列。
- 返回 Matplotlib `Figure` 与 `Axes` 对象。
- 仅在提供 `output_path` 时保存文件。
- 需要时为 `output_path` 创建父目录。
- SVG 输出时把文本保存为可编辑的 `<text>` 节点。

批处理工作流中自行用 `matplotlib.pyplot.close(figure)` 关闭图，或在保存最终发表输出时用 `save_publication_figure(..., close=True)`。

## 发表样式辅助

准备论文或幻灯片输出时，创建图前调用一次 `apply_publication_style`。

```python
from gignn import apply_publication_style

apply_publication_style(font_size=12, axes_linewidth=1.2)
```

它设置 Nature 风格无衬线字体栈、以 `svg.fonttype="none"` 保持 SVG 文本可编辑、关闭上/右边框并使用无框图例。导入 `gignn` 不会自动修改全局 Matplotlib 设置。

用 `NATURE_PALETTE` / `NATURE_COLOR_CYCLE` 获得克制的语义配色。调色板包含主方法的深蓝、方向性效果的绿/红，以及支撑元素的中性色。

多面板图用 `add_panel_label` 添加小写加粗面板标签：

```python
from gignn import add_panel_label

add_panel_label(ax, "a")
```

绘图脚本需要在绘制前适应数据规模、类别数、标签长度或面板数时用 `infer_publication_layout`：

```python
from gignn import infer_publication_layout

layout = infer_publication_layout(
    row_count=len(data),
    category_count=data["class"].nunique(),
    panel_count=3,
    label_lengths=[len(label) for label in labels],
)
```

返回的布局含 `figsize`、`font_size`、`title_size`、`tick_rotation`、`legend_location`、`marker_size`、`bar_width`、`wspace`、`hspace`。`plot_accuracy_comparison(..., figsize=None)` 用该辅助函数为长模型名自适应宽度、刻度旋转、条宽与标签字号。

## 发表导出

`save_publication_figure` 支持 SVG 优先的多格式导出并返回保存路径。

```python
from gignn import save_publication_figure

saved = save_publication_figure(
    figure,
    "figures/gignn_sa2_result",
    formats=("svg", "png"),
    dpi=300,
)
```

默认图保持打开以便继续编辑。传 `close=True` 在保存后关闭图。既有绘图函数仍接受 `output_path="figure.png"` 保存单个文件。

## 分类地图

```python
figure, ax = plot_classification_map(
    geo_frame,
    column="prediction",
    title="Predicted Class",
    output_path="outputs/prediction.png",
)
```

要求带 `.plot(column=..., ax=...)` 方法的 DataFrame 型对象。GeoDataFrame 可用于地图绘制。

## 预测对比

```python
figure, axes = plot_prediction_comparison(
    geo_frame,
    truth_column="observed",
    prediction_column="prediction",
    titles=("Observed", "Predicted"),
)
```

`titles` 必须恰好含两个值。

## 观测-预测偏差

```python
figure, axes = plot_observed_prediction_bias(
    geo_frame,
    observed_column="observed",
    prediction_column="prediction",
    bias_column="Bias Type",
    model_name="GIGNN",
    accuracy=0.82,
)
```

产出三个面板：观测、预测与偏差。

## 准确率对比

```python
accuracy_table = model_accuracy_table({"GIGNN": 0.82, "Baseline": 0.50}, year=2021)
figure, ax = plot_accuracy_comparison(accuracy_table)
```

默认表列为 `Model` 与 `Accuracy`。

## 离散化直方图

```python
figure, ax = plot_discretization_histogram(
    X,
    "ndvi",
    intervals=[0.3, 0.6],
)
```

选定列必须含有限数值。可选区间以竖直虚线显示。

## 训练曲线

```python
figure, axes = plot_training_curves(
    history,
    loss_column="loss",
    train_accuracy_column="train_accuracy",
    test_accuracy_column="test_accuracy",
)
```

`history` 必须含损失、训练准确率与测试准确率列。

## 指标箱线图

```python
figure, ax = plot_metric_boxplot(
    metrics,
    metric_column="Metric",
    value_column="Value",
    group_column="Group",
)
```

不分组箱线图设 `group_column=None`。

## 交互网络

```python
figure, ax = plot_interaction_network(model.discovery_report_.interaction_scores)
```

默认列为 `factor_1`、`factor_2`、`q_interaction`。交互表必须至少含一条边。权重列必须含有限非负数值；负值或非有限的地理探测器交互权重抛出 `ValidationError`。

## 输出文件

向单个绘图函数传 `output_path` 时，函数解析路径、创建父目录并以 `bbox_inches="tight"` 保存该单个文件。

```python
plot_accuracy_comparison(accuracy_table, output_path="reports/accuracy.png", dpi=300)
```

发表工作流优先 SVG 加 PNG 预览：

```python
figure, ax = plot_accuracy_comparison(accuracy_table)
save_publication_figure(figure, "reports/accuracy", formats=("svg", "png"), dpi=300)
```

无 `output_path` 或显式 `save_publication_figure` 调用时不写任何文件。
