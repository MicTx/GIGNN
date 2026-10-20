import numpy as np
import pandas as pd
import pytest

import gignn
from gignn.exceptions import ValidationError
from gignn.plotting import (
    NATURE_PALETTE,
    add_panel_label,
    apply_publication_style,
    infer_publication_layout,
    plot_accuracy_comparison,
    save_publication_figure,
    plot_classification_map,
    plot_discretization_histogram,
    plot_interaction_network,
    plot_metric_boxplot,
    plot_prediction_comparison,
    plot_training_curves,
)


class PlotGeometryFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return PlotGeometryFrame

    def plot(self, *args, **kwargs):
        ax = kwargs.get("ax")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        column = kwargs.get("column")
        values = np.arange(len(self)) if column is None else self[column].to_numpy()
        ax.scatter(np.arange(len(self)), values)
        return ax


def test_plot_classification_map_returns_figure_axes(tmp_path):
    frame = PlotGeometryFrame({"prediction": [0, 1, 2], "truth": [0, 1, 1]})
    path = tmp_path / "prediction.png"

    figure, ax = plot_classification_map(frame, column="prediction", output_path=path)

    assert path.is_file()
    assert figure is ax.figure


def test_legacy_plot_output_path_without_suffix_uses_matplotlib_default(tmp_path):
    frame = PlotGeometryFrame({"prediction": [0, 1, 2], "truth": [0, 1, 1]})
    path = tmp_path / "prediction"

    plot_classification_map(frame, column="prediction", output_path=path)

    assert path.with_suffix(".png").is_file()
    assert not path.with_suffix(".svg").exists()


def test_plot_prediction_comparison_returns_two_axes():
    frame = PlotGeometryFrame({"truth": [0, 1, 2], "prediction": [0, 2, 2]})

    figure, axes = plot_prediction_comparison(
        frame, truth_column="truth", prediction_column="prediction"
    )

    assert len(axes) == 2
    assert axes[0].figure is figure
    assert axes[1].figure is figure


def test_plot_prediction_comparison_rejects_wrong_title_count():
    frame = PlotGeometryFrame({"truth": [0, 1, 2], "prediction": [0, 2, 2]})

    with pytest.raises(ValidationError, match="titles must contain exactly two"):
        plot_prediction_comparison(
            frame,
            truth_column="truth",
            prediction_column="prediction",
            titles=("Observed",),
        )


def test_top_level_plotting_export_is_lazy():
    assert gignn.plot_training_curves is plot_training_curves
    assert gignn.apply_publication_style is apply_publication_style
    assert gignn.save_publication_figure is save_publication_figure
    assert gignn.infer_publication_layout is infer_publication_layout
    assert gignn.NATURE_PALETTE is NATURE_PALETTE


def test_plot_discretization_histogram_draws_interval_lines():
    frame = pd.DataFrame({"ndvi": [0.1, 0.2, 0.4, 0.8]})

    figure, ax = plot_discretization_histogram(frame, "ndvi", intervals=[0.3, 0.6])

    assert figure is ax.figure
    assert len(ax.lines) == 2


def test_plot_training_curves_uses_loss_and_accuracy():
    history = pd.DataFrame(
        {"loss": [1.0, 0.5], "train_accuracy": [0.4, 0.8], "test_accuracy": [0.3, 0.7]}
    )

    figure, axes = plot_training_curves(history)

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "Loss"
    assert axes[1].get_ylabel() == "Accuracy"


def test_plot_metric_boxplot_handles_report_table():
    metrics = pd.DataFrame(
        {
            "Metric": ["precision", "precision", "recall", "recall"],
            "Value": [0.7, 0.8, 0.6, 0.9],
            "Group": ["low", "high", "low", "high"],
        }
    )

    figure, ax = plot_metric_boxplot(
        metrics, metric_column="Metric", value_column="Value"
    )

    assert figure is ax.figure
    assert ax.get_xlabel() == "Metric"
    legend = ax.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"low", "high"}


def test_plot_interaction_network_uses_factor_pairs():
    interactions = pd.DataFrame(
        {
            "factor_1": ["ndvi", "ndvi"],
            "factor_2": ["dem", "road_density"],
            "q_interaction": [0.6, 0.8],
        }
    )

    figure, ax = plot_interaction_network(interactions)

    assert figure is ax.figure
    assert ax.get_title() == "GeoDetector Interaction Network"


@pytest.mark.parametrize("bad_weight", [-0.1, np.nan, np.inf])
def test_plot_interaction_network_rejects_invalid_weights(bad_weight):
    interactions = pd.DataFrame(
        {
            "factor_1": ["ndvi"],
            "factor_2": ["dem"],
            "q_interaction": [bad_weight],
        }
    )

    with pytest.raises(ValidationError, match="finite|non-negative"):
        plot_interaction_network(interactions)


def test_apply_publication_style_sets_editable_svg_defaults():
    import matplotlib.pyplot as plt

    original = dict(plt.rcParams)
    try:
        apply_publication_style(font_size=11, axes_linewidth=1.25)

        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert list(plt.rcParams["font.sans-serif"][:3]) == [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ]
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["font.size"] == 11
        assert plt.rcParams["axes.linewidth"] == 1.25
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["legend.frameon"] is False
    finally:
        plt.rcParams.update(original)


def test_apply_publication_style_default_disables_existing_tex():
    import matplotlib.pyplot as plt

    original = dict(plt.rcParams)
    try:
        plt.rcParams["text.usetex"] = True

        apply_publication_style()

        assert plt.rcParams["text.usetex"] is False
    finally:
        plt.rcParams.update(original)


def test_nature_palette_exposes_semantic_colors():
    assert NATURE_PALETTE["blue_main"] == "#0F4D92"
    assert NATURE_PALETTE["neutral_dark"] == "#4D4D4D"
    assert NATURE_PALETTE["delta_up"] == "#2E9E44"
    assert NATURE_PALETTE["delta_down"] == "#E53935"


def test_add_panel_label_adds_bold_axes_relative_text():
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()

    text = add_panel_label(ax, "a")

    assert text.get_text() == "a"
    assert text.get_fontweight() == "bold"
    assert text.get_transform() is ax.transAxes
    assert text in ax.texts
    assert figure is ax.figure


def test_save_publication_figure_writes_multiple_formats_without_closing(tmp_path):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.set_title("Editable title")

    saved = save_publication_figure(
        figure,
        tmp_path / "nested" / "figure",
        formats=("svg", "png"),
        dpi=120,
    )

    assert saved == [
        (tmp_path / "nested" / "figure.svg").resolve(),
        (tmp_path / "nested" / "figure.png").resolve(),
    ]
    assert all(path.is_file() for path in saved)
    assert figure.number in plt.get_fignums()
    assert "<text" in saved[0].read_text()


def test_save_publication_figure_accepts_single_format_string(tmp_path):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.set_title("Single format")

    saved = save_publication_figure(figure, tmp_path / "figure", formats="png")

    assert saved == [(tmp_path / "figure.png").resolve()]
    assert saved[0].is_file()
    assert not (tmp_path / "figure.p").exists()


def test_save_publication_figure_preserves_inferred_suffix_case(tmp_path):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.set_title("Suffix case")

    saved = save_publication_figure(figure, tmp_path / "Figure.SVG", formats=None)

    assert saved == [(tmp_path / "Figure.SVG").resolve()]
    assert saved[0].is_file()


def test_save_publication_figure_rejects_empty_format(tmp_path):
    import matplotlib.pyplot as plt

    figure, _ = plt.subplots()

    with pytest.raises(ValidationError, match="formats must contain non-empty"):
        save_publication_figure(figure, tmp_path / "figure", formats=("",))


def test_save_publication_figure_can_close_explicitly(tmp_path):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.set_title("Close me")

    saved = save_publication_figure(
        figure,
        tmp_path / "figure.svg",
        formats=None,
        close=True,
    )

    assert saved == [(tmp_path / "figure.svg").resolve()]
    assert figure.number not in plt.get_fignums()


def test_infer_publication_layout_adapts_to_long_labels_and_panels():
    layout = infer_publication_layout(
        row_count=189,
        category_count=5,
        panel_count=3,
        label_lengths=[4, 12, 38],
        geometry_aspect=2.4,
    )

    assert layout.figsize[0] > 10
    assert layout.figsize[1] >= 4.5
    assert layout.font_size <= 10
    assert layout.tick_rotation == 30
    assert layout.legend_location == "outside right"
    assert layout.wspace >= 0.25
    assert 20 <= layout.marker_size <= 80


def test_infer_publication_layout_keeps_compact_small_figures():
    layout = infer_publication_layout(
        row_count=12,
        category_count=2,
        panel_count=1,
        label_lengths=[3, 4],
    )

    assert layout.figsize[0] <= 5.5
    assert layout.figsize[1] <= 4.0
    assert layout.font_size >= 11
    assert layout.tick_rotation == 0
    assert layout.legend_location == "best"
    assert layout.marker_size >= 60


def test_plot_accuracy_comparison_infers_layout_for_long_model_names():
    table = pd.DataFrame(
        {
            "Model": [
                "GIGNN direct classifier",
                "GIGNN automatic GeoDetector workflow",
                "Long baseline model name for comparison",
            ],
            "Accuracy": [0.95, 0.70, 0.62],
        }
    )

    figure, ax = plot_accuracy_comparison(table, figsize=None)

    assert figure.get_size_inches()[0] > 7
    assert ax.get_xticklabels()[0].get_rotation() == 30
    assert ax.xaxis.label.get_size() <= 10


def test_plot_accuracy_comparison_preserves_accuracy_domain():
    table = pd.DataFrame(
        {
            "Model": ["GIGNN", "Baseline"],
            "Accuracy": [0.72, 0.81],
        }
    )

    _, ax = plot_accuracy_comparison(table, figsize=None)

    assert ax.get_ylim()[0] == 0
    assert ax.get_ylim()[1] >= 1.0


@pytest.mark.parametrize("bad_value", ["bad", np.nan, np.inf])
def test_plot_accuracy_comparison_rejects_invalid_accuracy_values(bad_value):
    table = pd.DataFrame(
        {
            "Model": ["GIGNN", "Baseline"],
            "Accuracy": [0.72, bad_value],
        }
    )

    with pytest.raises(ValidationError, match="Accuracy.*finite numeric"):
        plot_accuracy_comparison(table)
