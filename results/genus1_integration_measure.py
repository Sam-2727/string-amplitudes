#!/usr/bin/env python3
"""Reproduce the genus one string integration measure.

The paper scans all integer triples ``ell_1 + ell_2 + ell_3 = 1500``,
compares the disc frame matter +ghost measure with the flat frame torus measure,
and plots the relative difference in both ``tau`` and ``(ell_1, ell_2)`` coordinates.  
This script can be broken up into smaller parts through ``--task-index`` and ``--task-count``;
``--plot-only`` merges the completed parts and creates ``median_binned_sigma_0p65.pdf``.  Use
``--quick`` for a small/quick test.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from pathlib import Path
import sys

import numpy as np

try:
    from string_amplitudes import (
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        compute_period_map,
        critical_bosonic_string_integrand,
        generate_ribbon_graphs,
        holomorphic_one_form_antiderivatives,
        riemann_constant_vector,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from string_amplitudes import (  # type: ignore[no-redef]
        BCGhostCorrelatorData,
        RiemannSurfaceData,
        bc_correlator,
        compute_period_map,
        critical_bosonic_string_integrand,
        generate_ribbon_graphs,
        holomorphic_one_form_antiderivatives,
        riemann_constant_vector,
    )


C_POINT = -0.17 + 0.14j
ANCHOR_B_POINT = 0.21 + 0.17j
DIVISOR_POINT = 0.23 + 0.11j


def _dedekind_eta(modulus: complex, tolerance: float = 1e-15) -> complex:
    q = np.exp(2j * math.pi * modulus)
    product_value = 1.0 + 0.0j
    q_power = q
    while abs(q_power) / max(1.0 - abs(q), 1e-15) > tolerance:
        product_value *= 1.0 - q_power
        q_power *= q
    return complex(np.exp(1j * math.pi * modulus / 12.0) * product_value)


def _surface(graph, lengths: tuple[int, int, int]) -> RiemannSurfaceData:
    data = compute_period_map(graph, lengths, period_quadrature_order=128)
    return RiemannSurfaceData(
        genus=1,
        Omega=data.period_matrix,
        normalized_one_forms=data.normalized_one_forms,
        antiderivatives_normalized_forms=holomorphic_one_form_antiderivatives(
            data.normalized_one_forms, quadrature_order=96
        ),
    )


def _tau(graph, lengths: tuple[int, int, int]) -> complex:
    return complex(compute_period_map(graph, lengths).period_matrix[0, 0])


def _tau_jacobian(graph, lengths: tuple[int, int, int]) -> complex:
    def derivatives(step: int) -> tuple[complex, complex]:
        values = []
        for edge in (0, 1):
            plus = list(lengths)
            minus = list(lengths)
            plus[edge] += step
            plus[2] -= step
            minus[edge] -= step
            minus[2] += step
            values.append(( _tau(graph, tuple(plus)) - _tau(graph, tuple(minus)) ) / (2.0 * step))
        return values[0], values[1]

    first = derivatives(2)
    second = derivatives(4)
    d1 = (4.0 * first[0] - second[0]) / 3.0
    d2 = (4.0 * first[1] - second[1]) / 3.0
    return d1 * np.conjugate(d2) - d2 * np.conjugate(d1)


def _point(graph, lengths: tuple[int, int, int], *, quadrature: int, cutoff: int):
    surface = _surface(graph, lengths)
    tau = complex(surface.Omega[0, 0])
    riemann_constant = riemann_constant_vector(surface)
    trial_data = BCGhostCorrelatorData(
        surface=surface,
        riemann_constant=riemann_constant,
        divisor_points=(DIVISOR_POINT,),
        normalization_point=C_POINT,
        sigma_normalization=1.0,
        chiral_z1=1.0,
    )
    trial = bc_correlator(
        (ANCHOR_B_POINT,),
        (C_POINT,),
        trial_data,
        lambda_weight=1.0,
        lattice_cutoff=cutoff,
    )
    chiral_z1 = abs(
        trial / surface.normalized_one_forms[0](ANCHOR_B_POINT)
    ) ** (2.0 / 3.0)
    ghost_data = BCGhostCorrelatorData(
        surface=surface,
        riemann_constant=riemann_constant,
        divisor_points=(DIVISOR_POINT,),
        normalization_point=C_POINT,
        sigma_normalization=1.0,
        chiral_z1=chiral_z1,
    )
    matter_partition = (
        (2.0 * math.pi) ** -1 * tau.imag**-0.5 / abs(chiral_z1)
    )
    raw_disc = critical_bosonic_string_integrand(
        graph,
        lengths,
        matter_partition,
        ghost_data,
        c_point=C_POINT,
        dependent_edge=2,
        num_integration_points=quadrature,
        endpoint_exponent=-2.0 / 3.0,
        theta_lattice_cutoff=cutoff,
        max_correlator_evaluations=None,
    )
    numerical = (
        (2.0 * math.pi) ** 20
        * abs(surface.normalized_one_forms[0](C_POINT)) ** 2
        * raw_disc
    )
    jacobian = _tau_jacobian(graph, lengths)
    analytic = (
        (2.0 * math.pi) ** -24
        * tau.imag**-13
        * abs(_dedekind_eta(tau)) ** -48
        * abs(jacobian)
    )
    return tau, numerical, analytic, abs(numerical / analytic - 1.0)


def _points(total_boundary_length: int, step: int):
    half = total_boundary_length // 2
    lower = max(5, step)
    for first in range(lower, half, step):
        for second in range(lower, half - first, step):
            third = half - first - second
            if third >= lower:
                yield first, second, third


def _reduce_tau(modulus: complex) -> complex:
    modulus = complex(modulus)
    for _ in range(100):
        modulus -= round(modulus.real)
        if abs(modulus) < 1.0:
            modulus = -1.0 / modulus
            continue
        return modulus
    raise RuntimeError("modular reduction did not converge")


def _plot(records: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    tau = np.asarray(
        [_reduce_tau(complex(float(row["tau_re"]), float(row["tau_im"]))) for row in records]
    )
    first = np.asarray([float(row["ell1"]) for row in records])
    second = np.asarray([float(row["ell2"]) for row in records])
    residual = np.asarray([float(row["relative_difference"]) for row in records])
    positive = residual[residual > 0.0]
    norm = LogNorm(
        vmin=max(float(np.percentile(positive, 2)), 1e-12),
        vmax=float(np.percentile(positive, 98)),
    )
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (tau_axis, length_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), gridspec_kw={"wspace": 0.25}
    )
    first_map = tau_axis.hexbin(
        tau.real,
        tau.imag,
        C=residual,
        gridsize=180,
        reduce_C_function=np.median,
        mincnt=1,
        norm=norm,
        cmap="viridis",
    )
    length_axis.hexbin(
        first,
        second,
        C=residual,
        gridsize=180,
        reduce_C_function=np.median,
        mincnt=1,
        norm=norm,
        cmap="viridis",
    )
    tau_axis.set_xlabel(r"$\operatorname{Re}\tau$")
    tau_axis.set_ylabel(r"$\operatorname{Im}\tau$")
    length_axis.set_xlabel(r"$\ell_1$")
    length_axis.set_ylabel(r"$\ell_2$")
    for axis in (tau_axis, length_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="in", top=False, right=False)
    colorbar = fig.colorbar(first_map, ax=(tau_axis, length_axis), pad=0.02)
    colorbar.set_label("Relative difference")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.18, top=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=3000)
    parser.add_argument("--grid-step", type=int, default=1)
    parser.add_argument("--num-integration-points", type=int, default=32)
    parser.add_argument("--theta-cutoff", type=int, default=8)
    parser.add_argument(
        "--task-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "paper"
        / "figures"
        / "genus1_measure_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "paper"
        / "figures"
        / "median_binned_sigma_0p65.pdf",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    args.data_directory.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        if args.quick:
            args.L = 300
            args.grid_step = 25
            args.num_integration_points = min(args.num_integration_points, 2)
            args.theta_cutoff = min(args.theta_cutoff, 4)
        if args.task_count <= 0 or not 0 <= args.task_index < args.task_count:
            raise ValueError("require 0 <= task-index < task-count")
        graph = generate_ribbon_graphs(1, n_faces=1)[0]
        path = args.data_directory / (
            f"part-{args.task_index:05d}-of-{args.task_count:05d}.csv"
        )
        fields = (
            "ell1",
            "ell2",
            "ell3",
            "tau_re",
            "tau_im",
            "lhs",
            "rhs",
            "relative_difference",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            processed = 0
            for index, lengths in enumerate(_points(args.L, args.grid_step)):
                if index % args.task_count != args.task_index:
                    continue
                tau, lhs, rhs, difference = _point(
                    graph,
                    lengths,
                    quadrature=args.num_integration_points,
                    cutoff=args.theta_cutoff,
                )
                writer.writerow(
                    {
                        "ell1": lengths[0],
                        "ell2": lengths[1],
                        "ell3": lengths[2],
                        "tau_re": tau.real,
                        "tau_im": tau.imag,
                        "lhs": lhs,
                        "rhs": rhs,
                        "relative_difference": difference,
                    }
                )
                processed += 1
                if processed % 10 == 0:
                    print(f"completed {processed} points", flush=True)
        print(f"wrote {path}")

    part_paths = sorted(glob.glob(str(args.data_directory / "part-*.csv")))
    if not part_paths:
        raise FileNotFoundError(f"no result parts found in {args.data_directory}")
    records: list[dict[str, str]] = []
    for part_path in part_paths:
        with open(part_path, newline="", encoding="utf-8") as handle:
            records.extend(csv.DictReader(handle))
    _plot(records, args.output)
    print(f"plotted {len(records):,} points")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
