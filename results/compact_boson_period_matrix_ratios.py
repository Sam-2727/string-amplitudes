#!/usr/bin/env python3
"""Reproduce the genus 2--6 compact boson ratio comparison.

For one fixed one face ribbon graph at each genus, this script evaluates the compact boson partition function
 in the disc construction and in the conventional
period matrix construction, then writes the two panel figure
``compact_boson_genus2_genus3_genus4_genus5_genus6_period_matrix_ratio_two_panel.pdf``.
(this figure appears in the final paper)
The genus five and genus six paper cutoffs are recommended to be run on a supercomputer.
``--quick`` lowers every cutoff and edge length for a local test.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

try:
    from string_amplitudes import (
        compute_period_map,
        generate_ribbon_graphs,
        sample_ribbon_graph,
    )
    from string_amplitudes.partition_function import (
        _compact_boson_reduced_form,
        _truncated_winding_sum,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        compute_period_map,
        generate_ribbon_graphs,
        sample_ribbon_graph,
    )
    from string_amplitudes.partition_function import (  # type: ignore[no-redef]
        _compact_boson_reduced_form,
        _truncated_winding_sum,
    )


EDGE_LENGTHS = {2: 200, 3: 120, 4: 86, 5: 67, 6: 55}
LATTICE_CUTOFFS = {2: 8, 3: 5, 4: 3, 5: 3, 6: 3}
SAMPLE_SEEDS = {5: 505, 6: 606}
COLORS = {
    2: "#0072B2",
    3: "#D55E00",
    4: "#009E73",
    5: "#E69F00",
    6: "#CC79A7",
}
MARKERS = {2: "o", 3: "s", 4: "^", 5: "D", 6: "v"}


def _graph_for_genus(genus: int):
    if genus <= 4:
        return generate_ribbon_graphs(genus, n_faces=1)[0]
    graph, metadata = sample_ribbon_graph(
        genus, n_faces=1, seed=SAMPLE_SEEDS[genus]
    )
    print(f"g={genus} sampled graph metadata: {metadata}")
    return graph


def _period_matrix_form(period_matrix: np.ndarray) -> np.ndarray:
    real = period_matrix.real
    imaginary_inverse = np.linalg.inv(period_matrix.imag)
    return np.block(
        [
            [imaginary_inverse, imaginary_inverse @ real],
            [real.T @ imaginary_inverse, period_matrix.imag + real.T @ imaginary_inverse @ real],
        ]
    )


def _theta_sum(form: np.ndarray, radius: float, cutoff: int, chunk_size: int) -> float:
    # _truncated_winding_sum has exp(-4*pi*R^2*n^T*T*n), hence Q/4.
    return _truncated_winding_sum(form / 4.0, radius, cutoff, chunk_size)


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
        / "compact_boson_genus2_genus3_genus4_genus5_genus6_period_matrix_ratio_two_panel.pdf",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    radii = np.arange(1.1, 2.01, 0.1)
    reference_radius = 1.0
    records: list[dict[str, object]] = []
    for genus in range(2, 7):
        graph = _graph_for_genus(genus)
        edge_length = 12 if args.quick else EDGE_LENGTHS[genus]
        cutoff = 1 if args.quick else LATTICE_CUTOFFS[genus]
        lengths = (edge_length,) * len(graph[0])
        period_data = compute_period_map(
            graph, lengths, period_quadrature_order=64 if args.quick else 128
        )
        _, reduced_form = _compact_boson_reduced_form(graph, lengths)
        period_form = _period_matrix_form(period_data.period_matrix)

        graph_reference = _theta_sum(
            4.0 * reduced_form,
            reference_radius,
            cutoff,
            args.chunk_size,
        )
        period_reference = _theta_sum(
            period_form,
            reference_radius,
            cutoff,
            args.chunk_size,
        )
        for radius in radii:
            graph_sum = _theta_sum(
                4.0 * reduced_form, radius, cutoff, args.chunk_size
            )
            period_sum = _theta_sum(
                period_form, radius, cutoff, args.chunk_size
            )
            graph_ratio = radius * graph_sum / graph_reference
            period_ratio = radius * period_sum / period_reference
            records.append(
                {
                    "genus": genus,
                    "edge_length": edge_length,
                    "radius": radius,
                    "reference_radius": reference_radius,
                    "lattice_cutoff": cutoff,
                    "graph_ratio": graph_ratio,
                    "period_matrix_ratio": period_ratio,
                    "relative_difference": abs(graph_ratio / period_ratio - 1.0),
                }
            )
        print(f"completed genus {genus}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (value_axis, difference_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), gridspec_kw={"wspace": 0.24}
    )
    genus_handles = []
    for genus in range(2, 7):
        group = [row for row in records if row["genus"] == genus]
        x = [row["radius"] for row in group]
        value_axis.plot(
            x,
            [row["period_matrix_ratio"] for row in group],
            color=COLORS[genus],
            linewidth=1.5,
        )
        value_axis.scatter(
            x,
            [row["graph_ratio"] for row in group],
            color=COLORS[genus],
            marker=MARKERS[genus],
            s=20,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        difference_axis.semilogy(
            x,
            [row["relative_difference"] for row in group],
            color=COLORS[genus],
            marker=MARKERS[genus],
            markerfacecolor="white",
            markersize=4.1,
            linewidth=1.45,
        )
        genus_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[genus],
                marker=MARKERS[genus],
                markerfacecolor="white",
                markersize=4.1,
                label=rf"$g={genus}$",
            )
        )

    value_axis.legend(
        (
            Line2D([0], [0], color="black", linewidth=1.5),
            Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=4.5),
        ),
        ("Right hand side", "Left hand side"),
        frameon=False,
        loc="upper left",
    )
    fig.legend(
        handles=genus_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
    )
    value_axis.set_ylabel("Value")
    difference_axis.set_ylabel("Relative difference")
    for axis in (value_axis, difference_axis):
        axis.set_xlabel(r"Radius $R_1$")
        axis.set_xlim(1.05, 2.04)
        axis.set_xticks(np.arange(1.1, 2.01, 0.1))
        _style_axis(axis)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.97)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    for genus in range(2, 7):
        maximum = max(
            row["relative_difference"]
            for row in records
            if row["genus"] == genus
        )
        print(f"g={genus}: maximum relative difference = {maximum:.6e}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
