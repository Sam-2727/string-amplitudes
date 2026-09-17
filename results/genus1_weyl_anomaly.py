#!/usr/bin/env python3
"""Reproduce the genus one Weyl anomaly comparison in the paper.

The script computes the ratio of partition functions for two
different genus one ribbon graph geometries
with the same circumference. It fits the empirical large-L
residual to ``a / L**b``, and writes ``genus1WeylAnomalyTwoPanel.pdf``.
Use ``--quick`` for a short smoke test rather than the paper scan.
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
        generate_ribbon_graphs,
        get_boundary_data,
        matter_log_determinant,
    )
    from string_amplitudes.ribbon_graph_to_period_matrix import (
        _raw_holomorphic_forms,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        compute_period_map,
        generate_ribbon_graphs,
        get_boundary_data,
        matter_log_determinant,
    )
    from string_amplitudes.ribbon_graph_to_period_matrix import (  # type: ignore[no-redef]
        _raw_holomorphic_forms,
    )


REFERENCE_RATIO = (3, 3, 4)
TARGET_RATIOS = ((1, 1, 3), (1, 2, 2), (1, 3, 6), (1, 4, 5))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _common_length_unit() -> int:
    denominators = (2 * sum(ratio) for ratio in (REFERENCE_RATIO, *TARGET_RATIOS))
    return reduce(math.lcm, denominators, 1)


def _scaled_lengths(total_length: int, ratio: tuple[int, int, int]) -> tuple[int, ...]:
    denominator = 2 * sum(ratio)
    if total_length % denominator:
        raise ValueError(f"L={total_length} is not divisible by {denominator}")
    scale = total_length // denominator
    return tuple(scale * entry for entry in ratio)


def _reduce_modulus(modulus: complex) -> complex:
    modulus = complex(modulus)
    for _ in range(100):
        shifted = modulus - round(modulus.real)
        if abs(shifted.real - modulus.real) > 1e-15:
            modulus = shifted
        elif abs(modulus) < 1.0 - 1e-14:
            modulus = -1.0 / modulus
        elif modulus.real > 0.5 + 1e-14:
            modulus -= 1.0
        elif modulus.real < -0.5 - 1e-14:
            modulus += 1.0
        else:
            return modulus
    raise RuntimeError("modular reduction did not converge")


def _dedekind_eta(modulus: complex, tolerance: float = 1e-15) -> complex:
    q = np.exp(2j * math.pi * modulus)
    product_value = 1.0 + 0.0j
    q_power = q
    while abs(q_power) > tolerance:
        product_value *= 1.0 - q_power
        q_power *= q
    return complex(np.exp(1j * math.pi * modulus / 12.0) * product_value)


def _raw_vertex_coefficients(graph, lengths: tuple[int, ...]) -> np.ndarray:
    boundary = get_boundary_data(graph, lengths)
    starts = boundary["boundary_segment_starts"][0]
    occurrences = {
        edge: tuple(position for _, position in positions)
        for edge, positions in boundary["sewing"].items()
    }
    form = _raw_holomorphic_forms(
        starts,
        occurrences,
        lengths,
        boundary["boundary_lengths"][0],
        genus=1,
        zero_column_tolerance=None,
    )[0]
    values = []
    for index, prevertex in enumerate(form.prevertices):
        other = np.delete(form.prevertices, index)
        regular_factor = np.prod(
            (1.0 - prevertex / other) ** (-form.singularity_power)
        )
        polynomial = np.polynomial.polynomial.polyval(
            prevertex, form.coefficients
        )
        values.append(polynomial * regular_factor)
    return np.asarray(values, dtype=np.complex128)


def _surface_terms(graph, lengths: tuple[int, ...]) -> dict[str, float | complex]:
    total_length = 2 * sum(lengths)
    period_data = compute_period_map(graph, lengths, period_quadrature_order=128)
    modulus = _reduce_modulus(period_data.period_matrix[0, 0])
    log_z_disc = -0.5 * (
        matter_log_determinant(graph, lengths) - math.log(total_length / 2.0)
    )
    log_z_flat = -0.5 * math.log(modulus.imag) - 2.0 * math.log(
        abs(_dedekind_eta(modulus))
    )
    local_coefficients = _raw_vertex_coefficients(graph, lengths)
    s_weyl = math.log(float(np.mean(np.abs(local_coefficients)))) / 12.0
    return {
        "tau": modulus,
        "log_z_disc": log_z_disc,
        "log_z_flat": log_z_flat,
        "s_weyl": s_weyl,
    }


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
        / "genus1WeylAnomalyTwoPanel.pdf",
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
        reference = _surface_terms(
            graph, _scaled_lengths(total_length, REFERENCE_RATIO)
        )
        for ratio in TARGET_RATIOS:
            lengths = _scaled_lengths(total_length, ratio)
            target = _surface_terms(graph, lengths)
            lhs = float(target["s_weyl"]) - float(reference["s_weyl"])
            rhs = (
                float(target["log_z_disc"])
                - float(reference["log_z_disc"])
                - float(target["log_z_flat"])
                + float(reference["log_z_flat"])
            )
            records.append(
                {
                    "L": total_length,
                    "ratio": ":".join(map(str, ratio)),
                    "edge_lengths": ";".join(map(str, lengths)),
                    "lhs": lhs,
                    "rhs": rhs,
                    "relative_difference": abs(lhs - rhs) / abs(rhs),
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
    fig, (full_axis, zoom_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), gridspec_kw={"wspace": 0.25}
    )
    for ratio, color in zip(TARGET_RATIOS, COLORS):
        label = ":".join(map(str, ratio))
        group = [row for row in records if row["ratio"] == label]
        lengths = np.asarray([row["L"] for row in group], dtype=float)
        residuals = np.asarray(
            [row["relative_difference"] for row in group], dtype=float
        )
        full_axis.semilogy(lengths, residuals, color=color, linewidth=1.5)
        zoom_mask = (lengths >= 2000.0) & (lengths <= 4000.0)
        zoom_axis.semilogy(
            lengths[zoom_mask],
            residuals[zoom_mask],
            color=color,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            label=rf"${ratio[0]}:{ratio[1]}:{ratio[2]}$",
        )
        fit_mask = (lengths >= 2000.0) & (lengths <= 3000.0)
        if np.count_nonzero(fit_mask) >= 2:
            slope, intercept = np.polyfit(
                np.log(lengths[fit_mask]), np.log(residuals[fit_mask]), 1
            )
            fit_lengths = np.linspace(2000.0, 6000.0, 200)
            fit = np.exp(intercept) * fit_lengths**slope
            full_axis.semilogy(
                fit_lengths, fit, color=color, linestyle="--", linewidth=1.0
            )
            shown = fit_lengths <= 4000.0
            zoom_axis.semilogy(
                fit_lengths[shown],
                fit[shown],
                color=color,
                linestyle="--",
                linewidth=1.0,
            )

    full_axis.set_xlim(min(row["L"] for row in records) * 0.9, 6000)
    zoom_axis.set_xlim(2000, 4000)
    full_axis.set_ylabel("Relative difference")
    for axis in (full_axis, zoom_axis):
        axis.set_xlabel(r"Total circumference $L$")
        _style_axis(axis)
    zoom_axis.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.98)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"maximum relative difference: {max(row['relative_difference'] for row in records):.6e}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
