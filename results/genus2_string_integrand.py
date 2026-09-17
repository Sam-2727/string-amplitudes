#!/usr/bin/env python3
"""Reproduce the genus-two full string-integrand comparison.

The script evaluates the disc frame matter/ghost integrand and compares to the known
period matrix formula for a prescribed set of ``ell_i``, then
writes the results to ``genus2_measure_fixed_L_family_two_panel.pdf``. The b-ghost integration
is performed, so the precise normalization is meaningful. The number of integration points is
contrled by ``--num-integration-points``. ``--quick`` uses a rescaled three-point family.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

try:
    from string_amplitudes import (
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        bghost_measure,
        compute_period_map,
        generate_ribbon_graphs,
        genus_two_period_matrix_integrand,
        holomorphic_one_form_antiderivatives,
        matter_log_determinant,
        riemann_constant_vector,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        bghost_measure,
        compute_period_map,
        generate_ribbon_graphs,
        genus_two_period_matrix_integrand,
        holomorphic_one_form_antiderivatives,
        matter_log_determinant,
        riemann_constant_vector,
    )


ANCHOR_B_POINTS = (0.12 + 0.08j, -0.11 + 0.17j)
NORMALIZATION_POINT = -0.19 + 0.13j
DIVISOR_POINTS = (0.21 + 0.09j, -0.16 + 0.12j)
FILTER_DIVISORS = ((0.11 + 0.09j,),)


def _canonical_zeros(one_form) -> tuple[complex, ...]:
    coefficients = np.asarray(one_form.coefficients, dtype=np.complex128)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
    roots = np.roots(coefficients[: nonzero[-1] + 1][::-1])
    roots = roots[np.abs(roots) < 0.99]
    if roots.size != 2:
        raise ValueError(f"expected two canonical zeros; found {roots.size}")
    return tuple(complex(root) for root in roots)


def _disc_integrand(
    graph,
    lengths: tuple[int, ...],
    *,
    cutoff: int,
    num_integration_points: int,
) -> float:
    period_data = compute_period_map(graph, lengths, period_quadrature_order=128)
    surface = RiemannSurfaceData(
        genus=2,
        Omega=period_data.period_matrix,
        normalized_one_forms=period_data.normalized_one_forms,
        antiderivatives_normalized_forms=holomorphic_one_form_antiderivatives(
            period_data.normalized_one_forms, quadrature_order=96
        ),
    )
    matter_logdet = matter_log_determinant(graph, lengths)
    chiral_z1 = math.exp(
        0.5
        * (
            matter_logdet
            - math.log(float(np.linalg.det(surface.Omega.imag)))
        )
    )
    matter_partition = (2.0 * math.pi) ** -2 * math.exp(-0.5 * matter_logdet)
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
        chiral_z1=chiral_z1,
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
    ghost_data = BCGhostCorrelatorData(
        surface=surface,
        riemann_constant=riemann_constant,
        divisor_points=DIVISOR_POINTS,
        normalization_point=NORMALIZATION_POINT,
        sigma_normalization=np.linalg.det(form_matrix) / trial,
        chiral_z1=chiral_z1,
    )
    ghost_coefficient = bghost_measure(
        graph,
        lengths,
        ghost_data,
        c_point=0.0j,
        dependent_edge=8,
        num_integration_points=num_integration_points,
        endpoint_exponent=-2.0 / 3.0,
        theta_lattice_cutoff=cutoff,
        max_correlator_evaluations=None,
    )
    if abs(ghost_coefficient) == 0.0:
        raise FloatingPointError("the integrated ghost coefficient vanished")
    # Form ratios in logarithmic space: the absolute density can lie outside
    # float64 even when the fixed-L normalized comparison is well conditioned.
    return float(
        math.log(abs(ghost_coefficient)) + 26.0 * math.log(matter_partition)
    )


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
        / "genus2_measure_fixed_L_family_two_panel.pdf",
    )
    parser.add_argument("--num-integration-points", type=int, default=4)
    parser.add_argument("--theta-cutoff", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    graph = generate_ribbon_graphs(2, n_faces=1)[0]
    scale = 10 if args.quick else 1
    ell_values = (10, 20, 30) if args.quick else tuple(range(100, 301, 20))
    cutoff = min(args.theta_cutoff, 4) if args.quick else args.theta_cutoff
    quadrature = min(args.num_integration_points, 2) if args.quick else args.num_integration_points
    reference_lengths = (300 // scale,) * 9
    reference_log_disc = _disc_integrand(
        graph,
        reference_lengths,
        cutoff=cutoff,
        num_integration_points=quadrature,
    )
    reference_period = genus_two_period_matrix_integrand(
        graph,
        reference_lengths,
        theta_lattice_cutoff=cutoff,
        dependent_edge=8,
    )

    records: list[dict[str, object]] = []
    for ell in ell_values:
        half_length = 2700 // scale
        lengths = (ell,) * 8 + (half_length - 8 * ell,)
        log_disc = _disc_integrand(
            graph,
            lengths,
            cutoff=cutoff,
            num_integration_points=quadrature,
        )
        period = genus_two_period_matrix_integrand(
            graph,
            lengths,
            theta_lattice_cutoff=cutoff,
            dependent_edge=8,
        )
        lhs = math.exp(log_disc - reference_log_disc)
        rhs = period / reference_period
        records.append(
            {
                "ell": ell * scale,
                "edge_lengths": ";".join(map(str, lengths)),
                "lhs": lhs,
                "rhs": rhs,
                "relative_difference": abs(lhs / rhs - 1.0),
                "num_integration_points": quadrature,
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
