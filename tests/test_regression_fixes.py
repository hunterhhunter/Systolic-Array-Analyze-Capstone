from __future__ import annotations

import pandas as pd
import pytest

from tools.aggregator import TileRecord
from tools.emit_scalesim_topology import emit_conv2d_topology
from tools.io_utils import write_dataframe_outputs
from tools.sweep_runner import _weighted_reuse_aware_sum


def test_grouped_reuse_sum_charges_only_first_logical_tile_full_prefetch():
    rec = TileRecord(0, 102, 29, 0, 100.0, 100.0, 100.0)
    assert _weighted_reuse_aware_sum([rec], [128], dataflow="ws", fold_fraction=0.5) == pytest.approx(
        102 + 127 * 65.5
    )


def test_conv2d_topology_repeats_rows_for_batch_dimension():
    manifest = {
        "kind": "conv2d",
        "tile": {"oh": 2, "ow": 2, "oc": 4},
        "ops": [
            {
                "name": "conv",
                "kind": "conv2d",
                "shape": {
                    "batch": 2,
                    "ofmap_h": 2,
                    "ofmap_w": 2,
                    "num_filters": 4,
                    "filter_h": 3,
                    "filter_w": 3,
                    "channels": 3,
                    "stride_h": 1,
                    "stride_w": 1,
                    "dilation_h": 1,
                    "dilation_w": 1,
                },
            }
        ],
    }
    text = emit_conv2d_topology(manifest)
    rows = [line for line in text.splitlines() if line and not line.startswith("Layer name")]
    assert len(rows) == 2
    assert "conv_b000" in rows[0]
    assert "conv_b001" in rows[1]


def test_write_dataframe_outputs_writes_csv_even_when_parquet_engine_missing(monkeypatch, tmp_path):
    df = pd.DataFrame([{"x": 1}])

    def boom(*args, **kwargs):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    written = write_dataframe_outputs(df, tmp_path / "out.parquet", tmp_path / "out.csv")
    assert tmp_path.joinpath("out.csv").exists()
    assert tmp_path.joinpath("out.csv").read_text().startswith("x")
    assert tmp_path.joinpath("out.parquet") not in written
