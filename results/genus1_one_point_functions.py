#!/usr/bin/env python3
"""Reproduce the genus one free boson one point functions.

For the five edge length ratios used in the paper, this script compares the
sewn disc calculation with the exact period matrix result for
``:<dX dX>:`` and ``:<dX dbarX>:`` and writes
``genus1OnePointDiagnostics.pdf``.  Use ``--quick`` for a quicker version.
"""

from __future__ import annotations

import argparse
import csv
import math
from functools import reduce
from pathlib import Path
import sys

import numpy as np

try:
    from string_amplitudes import (
        compute_period_map,
        free_boson_one_point_functions,
        generate_ribbon_graphs,
        genus_one_disk_frame_one_point_functions,
        genus_one_period_matrix_one_point_functions,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        compute_period_map,
        free_boson_one_point_functions,
        generate_ribbon_graphs,
        genus_one_disk_frame_one_point_functions,
        genus_one_period_matrix_one_point_functions,
    )


RATIOS = ((3, 3, 4), (1, 1, 3), (1, 2, 2), (1, 3, 6), (1, 4, 5))
COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _common_length_unit() -> int:
    return reduce(math.lcm, (2 * sum(ratio) for ratio in RATIOS), 1)


def _scaled_lengths(total_length: int, ratio: tuple[int, int, int]) -> tuple[int, ...]:
    scale = total_length // (2 * sum(ratio))
    return tuple(scale * entry for entry in ratio)


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="in", which="both", top=False, right=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "paper"
        / "figures"
        / "genus1OnePointDiagnostics.pdf",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = generate_ribbon_graphs(1, n_faces=1)[0]
    unit = _common_length_unit()
    multipliers = (8, 20, 40) if args.quick else tuple(range(8, 154, 4))
    records: list[dict[str, object]] = []
    for multiplier in multipliers:
        total_length = unit * multiplier
        for ratio in RATIOS:
            lengths = _scaled_lengths(total_length, ratio)
            period_data = compute_period_map(
                graph, lengths, period_quadrature_order=128
            )
            numerical_disk = free_boson_one_point_functions(graph, lengths)
            exact_disk = genus_one_disk_frame_one_point_functions(period_data)
            exact_flat = genus_one_period_matrix_one_point_functions(
                period_data.period_matrix
            )
            derivative = period_data.normalized_one_forms[0](0.0j)
            schwarzian_term = (
                derivative**2 * exact_flat.holomorphic
                - exact_disk.holomorphic
            )
            numerical_holomorphic = (
                numerical_disk.holomorphic + schwarzian_term
            ) / derivative**2
            numerical_mixed = numerical_disk.mixed / abs(derivative) ** 2
            records.append(
                {
                    "L": total_length,
                    "ratio": ":".join(map(str, ratio)),
                    "edge_lengths": ";".join(map(str, lengths)),
                    "holomorphic_numerical_real": numerical_holomorphic.real,
                    "holomorphic_numerical_imag": numerical_holomorphic.imag,
                    "holomorphic_exact_real": exact_flat.holomorphic.real,
                    "holomorphic_exact_imag": exact_flat.holomorphic.imag,
                    "holomorphic_relative_difference": abs(
                        numerical_holomorphic / exact_flat.holomorphic - 1.0
                    ),
                    "mixed_numerical": numerical_mixed.real,
                    "mixed_exact": exact_flat.mixed.real,
                    "mixed_relative_difference": abs(
                        numerical_mixed / exact_flat.mixed - 1.0
                    ),
                }
            )
        print(f"completed L={total_length}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (holomorphic_axis, mixed_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), gridspec_kw={"wspace": 0.25}
    )
    for ratio, color in zip(RATIOS, COLORS):
        label = ":".join(map(str, ratio))
        group = [row for row in records if row["ratio"] == label]
        x = [row["L"] for row in group]
        plot_options = {
            "color": color,
            "linewidth": 1.4,
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "white",
        }
        holomorphic_axis.semilogy(
            x,
            [row["holomorphic_relative_difference"] for row in group],
            **plot_options,
        )
        mixed_axis.semilogy(
            x,
            [row["mixed_relative_difference"] for row in group],
            label=rf"${ratio[0]}:{ratio[1]}:{ratio[2]}$",
            **plot_options,
        )

    holomorphic_axis.set_ylabel("Relative difference")
    for axis in (holomorphic_axis, mixed_axis):
        axis.set_xlabel(r"Total circumference $L$")
        _style_axis(axis)
    mixed_axis.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.98)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    last_length = max(int(row["L"]) for row in records)
    endpoint = [row for row in records if row["L"] == last_length]
    print(
        f"L={last_length}: maximum holomorphic relative difference = "
        f"{max(row['holomorphic_relative_difference'] for row in endpoint):.6e}"
    )
    print(
        f"L={last_length}: maximum mixed relative difference = "
        f"{max(row['mixed_relative_difference'] for row in endpoint):.6e}"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
