# Plotting

GIGNN plotting functions are exposed at the top level through lazy loading. Importing `gignn` does not immediately import the plotting module.

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

All plotting functions:

- Validate required input columns.
- Return Matplotlib `Figure` and `Axes` objects.
- Save files only when `output_path` is provided.
- Create parent directories for `output_path` if needed.
- Save SVG text as editable `<text>` nodes when SVG output is used.

Close figures yourself in batch workflows with `matplotlib.pyplot.close(figure)`, or use `save_publication_figure(..., close=True)` when saving final publication outputs.

## Publication Style Helpers

Use `apply_publication_style` once before creating figures when preparing paper or slide outputs.

```python
from gignn import apply_publication_style

apply_publication_style(font_size=12, axes_linewidth=1.2)
```

It sets a Nature-style sans-serif font stack, keeps SVG text editable with `svg.fonttype="none"`, disables top/right spines, and uses frameless legends. Importing `gignn` does not automatically modify global Matplotlib settings.

Use `NATURE_PALETTE` / `NATURE_COLOR_CYCLE` for restrained semantic colors. The palette includes deep blue for the primary method, green/red for directional effects, and neutral tones for support elements.

Use `add_panel_label` for lowercase bold panel labels in multi-panel figures:

```python
from gignn import add_panel_label

add_panel_label(ax, "a")
```

Use `infer_publication_layout` when a plotting script needs to adapt to data size, category count, label length, or panel count before drawing:

```python
from gignn import infer_publication_layout

layout = infer_publication_layout(
    row_count=len(data),
    category_count=data["class"].nunique(),
    panel_count=3,
    label_lengths=[len(label) for label in labels],
)
```

The returned layout contains `figsize`, `font_size`, `title_size`, `tick_rotation`, `legend_location`, `marker_size`, `bar_width`, `wspace`, and `hspace`. `plot_accuracy_comparison(..., figsize=None)` uses this helper to adapt width, tick rotation, bar width, and label sizes for long model names.

## Publication Export

`save_publication_figure` supports SVG-first multi-format export and returns the saved paths.

```python
from gignn import save_publication_figure

saved = save_publication_figure(
    figure,
    "figures/gignn_sa2_result",
    formats=("svg", "png"),
    dpi=300,
)
```

By default, the figure remains open so callers can continue editing it. Pass `close=True` to close the figure after saving. Existing plotting functions still accept `output_path="figure.png"` and save that single file.

## Classification Map

```python
figure, ax = plot_classification_map(
    geo_frame,
    column="prediction",
    title="Predicted Class",
    output_path="outputs/prediction.png",
)
```

Requires a DataFrame-like object with a `.plot(column=..., ax=...)` method. GeoDataFrame works for map plots.

## Prediction Comparison

```python
figure, axes = plot_prediction_comparison(
    geo_frame,
    truth_column="observed",
    prediction_column="prediction",
    titles=("Observed", "Predicted"),
)
```

`titles` must contain exactly two values.

## Observed-Prediction Bias

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

Produces three panels: observed, predicted, and bias.

## Accuracy Comparison

```python
accuracy_table = model_accuracy_table({"GIGNN": 0.82, "Baseline": 0.50}, year=2021)
figure, ax = plot_accuracy_comparison(accuracy_table)
```

The default table columns are `Model` and `Accuracy`.

## Discretization Histogram

```python
figure, ax = plot_discretization_histogram(
    X,
    "ndvi",
    intervals=[0.3, 0.6],
)
```

The selected column must contain finite numeric values. Optional intervals are shown as vertical dashed lines.

## Training Curves

```python
figure, axes = plot_training_curves(
    history,
    loss_column="loss",
    train_accuracy_column="train_accuracy",
    test_accuracy_column="test_accuracy",
)
```

`history` must contain loss, train accuracy, and test accuracy columns.

## Metric Boxplot

```python
figure, ax = plot_metric_boxplot(
    metrics,
    metric_column="Metric",
    value_column="Value",
    group_column="Group",
)
```

Set `group_column=None` for ungrouped boxplots.

## Interaction Network

```python
figure, ax = plot_interaction_network(model.discovery_report_.interaction_scores)
```

Default columns are `factor_1`, `factor_2`, and `q_interaction`. The interactions table must contain at least one edge. The weight column must contain finite, non-negative numeric values; negative or non-finite GeoDetector interaction weights raise `ValidationError`.

## Output Files

When `output_path` is passed to an individual plotting function, the function resolves the path, creates parent directories, and saves that single file with `bbox_inches="tight"`.

```python
plot_accuracy_comparison(accuracy_table, output_path="reports/accuracy.png", dpi=300)
```

For publication workflows, prefer SVG plus a PNG preview:

```python
figure, ax = plot_accuracy_comparison(accuracy_table)
save_publication_figure(figure, "reports/accuracy", formats=("svg", "png"), dpi=300)
```

Without `output_path` or an explicit `save_publication_figure` call, no file is written.
