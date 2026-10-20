import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def symlink_supported(tmp_path):
    target = tmp_path / ".symlink-probe-target"
    link = tmp_path / ".symlink-probe"
    target.write_text("probe", encoding="utf-8")
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    else:
        link.unlink()
        target.unlink()


@pytest.fixture
def indicator_frame():
    return pd.DataFrame(
        {
            "ndvi": [0.12, 0.44, 0.51, 0.83, 0.25, 0.68],
            "dem": [12.0, 19.0, 21.0, 35.0, 14.0, 28.0],
            "road_density": [1.0, 2.5, 2.1, 4.2, 1.3, 3.7],
        }
    )


@pytest.fixture
def labels():
    return pd.Series(
        pd.Categorical(
            ["low", "medium", "medium", "high", "low", "high"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )


@pytest.fixture
def non_contiguous_labels():
    return pd.Series([10, 20, 20, 30, 10, 30])


@pytest.fixture
def edge_index():
    return np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
        ],
        dtype=np.int64,
    )
