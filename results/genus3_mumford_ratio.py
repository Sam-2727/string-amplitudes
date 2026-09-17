#!/usr/bin/env python3
"""Reproduce the genus three bc-correlator/Mumford-form ratio test.

The script evaluates the six b correlator and the period matrix expression
along ``ell_1 = ... = ell_14 = ell``, ``ell_15 = 4500 - 14*ell``, then writes
the results to
``genus3_mumford_fixed_L_family_two_panel.pdf``. Note that we do not integrate the b-ghosts,
so only compare to the form in terms of the Igusa cusp form. Therefore, the exact normalization is
not meaningful. ``--quick`` rescales the family and evaluates three points with a smaller theta cutoff.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

try:
    from string_amplitudes import (
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        compute_period_map,
        generate_ribbon_graphs,
        holomorphic_one_form_antiderivatives,
        igusa_cusp_form_genus_three,
        riemann_constant_vector,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        compute_period_map,
        generate_ribbon_graphs,
        holomorphic_one_form_antiderivatives,
        igusa_cusp_form_genus_three,
        riemann_constant_vector,
    )


B_POINTS = (
    0.08 + 0.12j,
    -0.14 + 0.16j,
    0.19 - 0.09j,
    0.21 + 0.09j,
    -0.16 + 0.12j,
    0.05 - 0.18j,
)
ANCHOR_B_POINTS = (0.12 + 0.08j, -0.11 + 0.17j, 0.18 - 0.07j)
NORMALIZATION_POINT = -0.19 + 0.13j
DIVISOR_POINTS = (0.17 + 0.05j, -0.12 + 0.11j, 0.11 - 0.09j)
FILTER_DIVISORS = ((0.09 + 0.13j, -0.08 + 0.16j),)


def _canonical_zeros(one_form) -> tuple[complex, ...]:
    coefficients = np.asarray(one_form.coefficients, dtype=np.complex128)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
    roots = np.roots(coefficients[: nonzero[-1] + 1][::-1])
    roots = roots[np.abs(roots) < 0.99]
    if roots.size != 4:
        raise ValueError(f"expected four canonical zeros; found {roots.size}")
    return tuple(complex(root) for root in roots)


def _observables(graph, lengths: tuple[int, ...], cutoff: int) -> dict[str, complex]:
    period_data = compute_period_map(graph, lengths, period_quadrature_order=128)
    surface = RiemannSurfaceData(
        genus=3,
        Omega=period_data.period_matrix,
        normalized_one_forms=period_data.normalized_one_forms,
        antiderivatives_normalized_forms=holomorphic_one_form_antiderivatives(
            period_data.normalized_one_forms, quadrature_order=96
        ),
    )
    riemann_constant = riemann_constant_vector(
        surface,
        _canonical_zeros(surface.normalized_one_forms[0]),
        filter_divisors=FILTER_DIVISORS,
        lattice_cutoff=cutoff,
    )
    trial_data = BCGhostCorrelatorData(
        surface=surface,
        riemann_constant=riemann_constant,
        divisor_points=DIVISOR_POINTS,
        normalization_point=NORMALIZATION_POINT,
        sigma_normalization=1.0,
        chiral_z1=1.0,
    )
    trial = bc_correlator(
        ANCHOR_B_POINTS,
        (NORMALIZATION_POINT,),
        trial_data,
        lambda_weight=1.0,
        lattice_cutoff=cutoff,
    )
    form_matrix = np.asarray(
        [
            [form(point) for point in ANCHOR_B_POINTS]
            for form in surface.normalized_one_forms
        ],
        dtype=np.complex128,
    )
    sigma_normalization = np.exp(
        np.log(np.linalg.det(form_matrix) / trial) / (surface.genus - 1)
    )
    ghost_data = BCGhostCorrelatorData(
        surface=surface,
        riemann_constant=riemann_constant,
        divisor_points=DIVISOR_POINTS,
        normalization_point=NORMALIZATION_POINT,
        sigma_normalization=sigma_normalization,
        chiral_z1=1.0,
    )
    correlator = bc_correlator(
        B_POINTS,
        (),
        ghost_data,
        lambda_weight=2.0,
        lattice_cutoff=cutoff,
    )
    pairs = tuple((i, j) for i in range(3) for j in range(i, 3))
    rows = []
    for point in B_POINTS:
        values = tuple(form(point) for form in surface.normalized_one_forms)
        rows.append(tuple(values[i] * values[j] for i, j in pairs))
    determinant = np.linalg.det(np.asarray(rows, dtype=np.complex128))
    cusp_form = igusa_cusp_form_genus_three(
        surface.Omega, lattice_cutoff=cutoff
    )
    return {
        "correlator": complex(correlator),
        "determinant": complex(determinant),
        "cusp_form": complex(cusp_form),
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
        / "genus3_mumford_fixed_L_family_two_panel.pdf",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    graph = generate_ribbon_graphs(3, n_faces=1)[0]
    cutoff = 2 if args.quick else 4
    scale = 10 if args.quick else 1
    ell_values = (14, 22, 30) if args.quick else tuple(range(140, 301, 20))
    reference_lengths = (300 // scale,) * 15
    reference = _observables(graph, reference_lengths, cutoff)
    records: list[dict[str, object]] = []
    for ell in ell_values:
        half_length = 4500 // scale
        lengths = (ell,) * 14 + (half_length - 14 * ell,)
        value = _observables(graph, lengths, cutoff)
        lhs = abs(value["correlator"] / reference["correlator"]) ** 2
        rhs = (
            abs(value["determinant"] / reference["determinant"]) ** 2
            * abs(reference["cusp_form"] / value["cusp_form"])
        )
        records.append(
            {
                "ell": ell * scale,
                "edge_lengths": ";".join(map(str, lengths)),
                "lhs": lhs,
                "rhs": rhs,
                "relative_difference": abs(lhs / rhs - 1.0),
                "theta_cutoff": cutoff,
            }
        )
        print(f"completed ell={ell * scale}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    x = [row["ell"] for row in records]
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (value_axis, difference_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), gridspec_kw={"wspace": 0.24}
    )
    value_axis.semilogy(
        x,
        [row["rhs"] for row in records],
        color="#242424",
        linewidth=1.5,
        label="Right hand side",
    )
    value_axis.scatter(
        x,
        [row["lhs"] for row in records],
        color="#008A9A",
        s=20,
        label="Left hand side",
        zorder=3,
    )
    difference_axis.plot(
        x,
        [row["relative_difference"] for row in records],
        color="#AF3B73",
        marker="o",
        markerfacecolor="white",
        markersize=4.0,
        linewidth=1.45,
    )
    value_axis.set_ylabel("Value")
    difference_axis.set_ylabel("Relative difference")
    difference_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else rf"${value:.1e}$")
    )
    value_axis.legend(frameon=False, loc="upper right")
    for axis in (value_axis, difference_axis):
        axis.set_xlabel(r"Edge length $\ell$")
        _style_axis(axis)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.98)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(
        "maximum relative difference: "
        f"{max(row['relative_difference'] for row in records):.6e}"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
