from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any

from gignn.exceptions import ValidationError

NATURE_PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "gold": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
}

NATURE_COLOR_CYCLE = (
    NATURE_PALETTE["blue_main"],
    NATURE_PALETTE["green_3"],
    NATURE_PALETTE["red_strong"],
    NATURE_PALETTE["teal"],
    NATURE_PALETTE["violet"],
    NATURE_PALETTE["neutral_light"],
)


@dataclass(frozen=True)
class PublicationLayout:
    figsize: tuple[float, float]
    font_size: float
    title_size: float
    tick_rotation: float
    legend_location: str
    marker_size: float
    bar_width: float
    wspace: float
    hspace: float


def infer_publication_layout(
    *,
    row_count: int,
    category_count: int = 1,
    panel_count: int = 1,
    label_lengths: Sequence[int] | None = None,
    geometry_aspect: float | None = None,
) -> PublicationLayout:
    labels = list(label_lengths or [])
    max_label = max(labels, default=0)
    safe_rows = max(int(row_count), 1)
    safe_categories = max(int(category_count), 1)
    safe_panels = max(int(panel_count), 1)
    aspect = 1.0 if geometry_aspect is None else max(float(geometry_aspect), 0.35)

    width = 3.2 + 0.95 * safe_panels + 0.18 * safe_categories + 0.08 * max_label
    height = 2.8 + 0.16 * safe_categories + min(np.sqrt(safe_rows) * 0.12, 1.4)
    if geometry_aspect is not None:
        width = max(width, height * min(aspect, 2.8))
    width = float(np.clip(width, 4.2, 13.5))
    height = float(np.clip(height, 3.2, 7.2))

    complexity = safe_categories + safe_panels + max_label / 12 + safe_rows / 220
    font_size = float(np.clip(13.1 - 0.55 * complexity, 7.5, 12.0))
    tick_rotation = 30.0 if max_label >= 16 or safe_categories >= 5 else 0.0
    legend_location = (
        "outside right" if safe_categories >= 4 or safe_panels >= 3 else "best"
    )
    marker_size = float(
        np.clip(78 - np.sqrt(safe_rows) * 2.7 - safe_categories * 2.0, 20, 72)
    )
    bar_width = float(np.clip(0.72 - safe_categories * 0.025, 0.36, 0.68))
    wspace = float(np.clip(0.16 + safe_panels * 0.04 + max_label / 180, 0.18, 0.48))
    hspace = float(np.clip(0.18 + safe_categories * 0.015, 0.2, 0.42))

    return PublicationLayout(
        figsize=(round(width, 2), round(height, 2)),
        font_size=round(font_size, 2),
        title_size=round(font_size + 1.5, 2),
        tick_rotation=tick_rotation,
        legend_location=legend_location,
        marker_size=round(marker_size, 2),
        bar_width=round(bar_width, 2),
        wspace=round(wspace, 2),
        hspace=round(hspace, 2),
    )


def apply_publication_style(
    *, font_size: float = 12, axes_linewidth: float = 1.2, use_tex: bool = False
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = max(axes_linewidth, 0.8)
    plt.rcParams["ytick.major.width"] = max(axes_linewidth, 0.8)
    plt.rcParams["text.usetex"] = bool(use_tex)


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.06,
    y: float = 1.02,
    fontsize: float = 14,
    color: str = "black",
    fontweight: str = "bold",
) -> Any:
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left",
        va="bottom",
    )


def save_publication_figure(
    figure: Figure,
    output_path: str | Path,
    *,
    formats: str | Sequence[str] | None = ("svg", "png"),
    dpi: int = 300,
    pad: float = 2,
    close: bool = False,
) -> list[Path]:
    import matplotlib.pyplot as plt

    base = Path(output_path).expanduser().resolve()
    if formats is None:
        saved = [base if base.suffix else base.with_suffix(".svg")]
        save_specs = [(saved[0], saved[0].suffix.lstrip(".").lower() or "svg")]
    else:
        normalized_formats = _normalize_publication_formats(formats)
        if not normalized_formats:
            raise ValidationError("formats must contain at least one format.")
        if base.suffix:
            base = base.with_suffix("")
        saved = [base.with_suffix(f".{fmt}") for fmt in normalized_formats]
        save_specs = list(zip(saved, normalized_formats))
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=pad)
    with plt.rc_context({"svg.fonttype": "none"}):
        for path, fmt in save_specs:
            figure.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return saved


def plot_classification_map(
    geo_frame: pd.DataFrame,
    *,
    column: str,
    title: str | None = None,
    cmap: str = "Blues",
    edgecolor: str = "#d9d9e3",
    figsize: tuple[float, float] = (12, 8),
    legend: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    _require_columns(geo_frame, [column])
    figure, ax = _subplots(figsize=figsize)
    geo_frame.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=legend,
        edgecolor=edgecolor,
    )
    ax.set_title(title or column)
    ax.set_axis_off()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, ax


def plot_prediction_comparison(
    geo_frame: pd.DataFrame,
    *,
    truth_column: str,
    prediction_column: str,
    titles: tuple[str, str] = ("Observed", "Predicted"),
    cmap: str = "Blues",
    edgecolor: str = "#d9d9e3",
    figsize: tuple[float, float] = (16, 7),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, np.ndarray]:
    _require_columns(geo_frame, [truth_column, prediction_column])
    if len(titles) != 2:
        raise ValidationError("titles must contain exactly two values.")
    figure, axes = _subplots(ncols=2, figsize=figsize)
    for ax, column, title in zip(axes, [truth_column, prediction_column], titles):
        geo_frame.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            legend=False,
            edgecolor=edgecolor,
        )
        ax.set_title(title)
        ax.set_axis_off()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, axes


def plot_observed_prediction_bias(
    geo_frame: pd.DataFrame,
    *,
    observed_column: str,
    prediction_column: str,
    bias_column: str,
    model_name: str,
    accuracy: float,
    class_labels: Sequence[str] | None = None,
    bias_categories: Sequence[str] = (
        "Correct",
        "Underestimated (-1)",
        "Overestimated (1)",
    ),
    cmap: str = "Blues",
    bias_cmap: str = "coolwarm",
    edgecolor: str = "#d9d9e3",
    figsize: tuple[float, float] = (24, 8),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, np.ndarray]:
    _require_columns(geo_frame, [observed_column, prediction_column, bias_column])
    plot_frame = geo_frame.copy(deep=True)
    plot_frame[bias_column] = _ordered_categorical(
        plot_frame[bias_column], categories=bias_categories
    )
    figure, axes = _subplots(ncols=3, figsize=figsize)
    plot_specs = [
        (observed_column, "Observed", cmap),
        (
            prediction_column,
            f"Predicted by {model_name} with accuracy {accuracy:.3f}",
            cmap,
        ),
        (bias_column, "Bias", bias_cmap),
    ]
    for ax, (column, title, color_map) in zip(axes, plot_specs):
        plot_frame.plot(
            column=column,
            ax=ax,
            cmap=color_map,
            legend=column == bias_column,
            categorical=column == bias_column,
            edgecolor=edgecolor,
        )
        ax.set_title(title)
        ax.set_axis_off()
    if class_labels is not None:
        figure.suptitle(" | ".join(class_labels))
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, axes


def plot_accuracy_comparison(
    accuracy_table: pd.DataFrame,
    *,
    model_column: str = "Model",
    accuracy_column: str = "Accuracy",
    title: str | None = None,
    figsize: tuple[float, float] | None = (12, 6),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    _require_columns(accuracy_table, [model_column, accuracy_column])
    labels = [str(value) for value in accuracy_table[model_column]]
    values = _finite_numeric_values(
        accuracy_table[accuracy_column], column_name=accuracy_column
    )
    plot_table = accuracy_table.assign(**{accuracy_column: values})
    layout = infer_publication_layout(
        row_count=len(accuracy_table),
        category_count=len(labels),
        panel_count=1,
        label_lengths=[len(label) for label in labels],
    )
    figure, ax = _subplots(figsize=layout.figsize if figsize is None else figsize)
    plot_table.plot(
        kind="bar",
        x=model_column,
        y=accuracy_column,
        color=NATURE_PALETTE["blue_main"],
        width=layout.bar_width,
        alpha=0.88,
        edgecolor="black",
        linewidth=1.0,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel(model_column, fontsize=layout.font_size)
    ax.set_ylabel("Accuracy", fontsize=layout.font_size)
    _set_tight_ylim(
        ax,
        values,
        lower=0.0,
        upper_floor=1.0,
    )
    ax.set_title(title or "Model Accuracy Comparison", fontsize=layout.title_size)
    ax.tick_params(axis="x", rotation=layout.tick_rotation, labelsize=layout.font_size)
    ax.tick_params(axis="y", labelsize=layout.font_size)
    _style_axes(ax)
    for index, patch in enumerate(ax.patches):
        patch.set_facecolor(NATURE_COLOR_CYCLE[index % len(NATURE_COLOR_CYCLE)])
        ax.annotate(
            f"{patch.get_height():.3f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, ax


def plot_discretization_histogram(
    data_frame: pd.DataFrame,
    column: str,
    *,
    intervals: Iterable[float] | None = None,
    bins: str | int | Sequence[float] = "auto",
    figsize: tuple[float, float] = (10, 6),
    hist_color: str = "#63abdd",
    line_color: str = "#d62728",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    _require_columns(data_frame, [column])
    values = pd.to_numeric(data_frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError(f"Column {column!r} must contain only finite values.")
    figure, ax = _subplots(figsize=figsize)
    ax.hist(
        values,
        bins=bins,
        color=hist_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
    )
    for interval in [] if intervals is None else intervals:
        ax.axvline(float(interval), color=line_color, linestyle="--", linewidth=1.5)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    _style_axes(ax)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, ax


def plot_training_curves(
    history: pd.DataFrame,
    *,
    loss_column: str = "loss",
    train_accuracy_column: str = "train_accuracy",
    test_accuracy_column: str = "test_accuracy",
    title: str = "Training Curves",
    figsize: tuple[float, float] = (14, 5),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, np.ndarray]:
    _require_columns(
        history, [loss_column, train_accuracy_column, test_accuracy_column]
    )
    figure, axes = _subplots(ncols=2, figsize=figsize)
    axes[0].plot(
        history[loss_column].to_numpy(dtype=float),
        color=NATURE_PALETTE["blue_main"],
        linewidth=2.2,
    )
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(
        history[train_accuracy_column].to_numpy(dtype=float),
        label="Train",
        color=NATURE_PALETTE["blue_main"],
        linewidth=2.2,
    )
    axes[1].plot(
        history[test_accuracy_column].to_numpy(dtype=float),
        label="Test",
        color=NATURE_PALETTE["red_strong"],
        linewidth=2.2,
    )
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    for ax in axes:
        _style_axes(ax)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, axes


def plot_metric_boxplot(
    metrics: pd.DataFrame,
    *,
    metric_column: str = "Metric",
    value_column: str = "Value",
    group_column: str | None = "Group",
    figsize: tuple[float, float] = (12, 6),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    required = [metric_column, value_column]
    if group_column is not None:
        required.append(group_column)
    _require_columns(metrics, required)
    figure, ax = _subplots(figsize=figsize)
    categories = list(pd.unique(metrics[metric_column]))
    if group_column is None:
        data = [
            metrics.loc[metrics[metric_column] == category, value_column]
            for category in categories
        ]
        try:
            ax.boxplot(data, tick_labels=categories)
        except TypeError:
            ax.boxplot(data, labels=categories)
    else:
        _grouped_boxplot(
            metrics, metric_column, value_column, group_column, categories, ax
        )
    ax.set_xlabel(metric_column)
    ax.set_ylabel(value_column)
    _style_axes(ax)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, ax


def plot_interaction_network(
    interactions: pd.DataFrame,
    *,
    source_column: str = "factor_1",
    target_column: str = "factor_2",
    weight_column: str = "q_interaction",
    title: str = "GeoDetector Interaction Network",
    figsize: tuple[float, float] = (10, 10),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    _require_columns(interactions, [source_column, target_column, weight_column])
    import networkx as nx

    weight_values = _finite_numeric_values(
        interactions[weight_column], column_name=weight_column
    )
    if weight_values.size and float(np.nanmin(weight_values)) < 0:
        raise ValidationError(f"{weight_column} must contain only non-negative values.")

    figure, ax = _subplots(figsize=figsize)
    graph = nx.Graph()
    for row in interactions[[source_column, target_column, weight_column]].itertuples(
        index=False
    ):
        graph.add_edge(str(row[0]), str(row[1]), weight=float(row[2]))
    if graph.number_of_edges() == 0:
        raise ValidationError("interactions must contain at least one edge.")
    positions = nx.spring_layout(graph, seed=42)
    weights = np.array(
        [data["weight"] for _, _, data in graph.edges(data=True)], dtype=float
    )
    widths = 1.0 + 4.0 * (weights / weights.max() if weights.max() > 0 else weights)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=NATURE_PALETTE["blue_secondary"],
        edgecolors="white",
        linewidths=0.8,
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=widths,
        edge_color=NATURE_PALETTE["neutral_mid"],
        alpha=0.75,
    )
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=10)
    ax.set_title(title)
    ax.set_axis_off()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=dpi)
    return figure, ax


def _ordered_categorical(
    values: pd.Series, *, categories: Sequence[str]
) -> pd.Categorical:
    observed = [value for value in pd.unique(values) if value not in categories]
    ordered_categories = list(categories) + [str(value) for value in observed]
    return pd.Categorical(
        values.astype(str), categories=ordered_categories, ordered=True
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}.")


def _subplots(*, ncols: int = 1, figsize: tuple[float, float]) -> tuple[Figure, Any]:
    import matplotlib.pyplot as plt

    return plt.subplots(ncols=ncols, figsize=figsize)


def _style_axes(ax: Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend_ and ax.legend_.set_frame_on(False)


def _normalize_publication_formats(formats: str | Sequence[str]) -> tuple[str, ...]:
    raw_formats = (formats,) if isinstance(formats, str) else tuple(formats)
    normalized = tuple(format_name.lstrip(".").lower() for format_name in raw_formats)
    if any(not format_name for format_name in normalized):
        raise ValidationError("formats must contain non-empty format names.")
    if any("/" in format_name or "\\" in format_name for format_name in normalized):
        raise ValidationError("formats must be format names, not paths.")
    return normalized


def _finite_numeric_values(values: pd.Series, *, column_name: str) -> np.ndarray:
    try:
        numeric_values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Column {column_name!r} must contain finite numeric values."
        ) from exc
    if not np.isfinite(numeric_values).all():
        raise ValidationError(
            f"Column {column_name!r} must contain finite numeric values."
        )
    return numeric_values


def _set_tight_ylim(
    ax: Axes,
    values: np.ndarray,
    *,
    lower: float | None = None,
    upper_floor: float | None = None,
) -> None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    minimum = (
        float(finite_values.min())
        if lower is None
        else min(float(lower), float(finite_values.min()))
    )
    maximum = float(finite_values.max())
    if minimum == maximum:
        margin = 0.05 if maximum == 0 else abs(maximum) * 0.1
    else:
        margin = (maximum - minimum) * 0.12
    upper = maximum + margin
    if upper_floor is not None:
        upper = max(upper, float(upper_floor))
    ax.set_ylim(minimum - margin if lower is None else lower, upper)


def _save_figure(
    figure: Figure,
    output_path: str | Path | None,
    *,
    dpi: int,
    close: bool = False,
) -> None:
    if output_path is None:
        return
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _boxplot_colors(count: int) -> list[str]:
    return [
        NATURE_COLOR_CYCLE[index % len(NATURE_COLOR_CYCLE)] for index in range(count)
    ]


def _grouped_boxplot(
    metrics: pd.DataFrame,
    metric_column: str,
    value_column: str,
    group_column: str,
    categories: list[object],
    ax: Axes,
) -> None:
    groups = list(pd.unique(metrics[group_column]))
    width = 0.8 / max(len(groups), 1)
    positions = []
    group_labels = []
    data = []
    for category_index, category in enumerate(categories):
        center = category_index + 1
        for group_index, group in enumerate(groups):
            offset = (group_index - (len(groups) - 1) / 2) * width
            values = metrics.loc[
                (metrics[metric_column] == category) & (metrics[group_column] == group),
                value_column,
            ]
            positions.append(center + offset)
            group_labels.append(str(group))
            data.append(values.to_numpy(dtype=float))
    colors = _boxplot_colors(len(groups))
    boxplot = ax.boxplot(
        data, positions=positions, widths=width * 0.8, patch_artist=True
    )
    for index, patch in enumerate(boxplot["boxes"]):
        patch.set_facecolor(colors[index % len(groups)])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels([str(category) for category in categories])
    handles = [boxplot["boxes"][index] for index in range(len(groups))]
    ax.legend(
        handles=handles, labels=[str(group) for group in groups], title=group_column
    )
