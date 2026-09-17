r"""Construct the bosonic string integrand in ribbon graph coordinates
using the formula of Verlinde, Verlinde in terms of the free boson partition
function on the ribbon graph surface.

* :func:`bghost_direction_coefficients` construct the matrix converting elementary
seam integrals to the :math:`\mathcal{B}` insertions associated with each edge length.
* :func:`integrated_bghost_edge_components` integrates the holomorphic
  :math:`bc` correlator over elementary seams.
* :func:`bghost_measure_from_edge_components` assembles the full bc ghost measure from
integrated :`\mathcal{B}` components.
* :func:`bghost_measure` assembles the bc ghost correlation function using the
Verlinde-Verlinde formula.
* :func:`critical_bosonic_string_integrand` multiplies the ghost measure by
  26 identical noncompact boson partition functions to compute the full critical bosonic
  string integrand.

The direct quadrature is the transparent reference implementation.  Its work
grows as ``binomial(6*g - 3, 3*g - 2) * quadrature_order**(3*g - 2)``;
production calculations at high genus require a separately validated
accelerated contraction.
"""

from __future__ import annotations

from itertools import combinations, product
import math
from numbers import Integral, Real
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .riemann_surface_holomorphic_data import (
    BCGhostCorrelatorData,
    prepare_bc_correlator,
)
from .ribbon_graph_generator import get_boundary_data


ComplexArray = NDArray[np.complex128]
IntegerArray = NDArray[np.int64]
BCorrelator = Callable[[Sequence[complex]], complex]
EdgeComponents = Mapping[tuple[int, ...], complex]


def bghost_direction_coefficients(
    ribbon_graph,
    *,
    dependent_edge: int | None = None,
) -> IntegerArray:
    r"""Returns the integer change of basis matrix that expresses each
    :math:`\mathcal{B}_{\ell_a}` ghost as a sum of elementary integrals.

    Each edge of the ribbon graph is represented as two pairs of edges on the boundary
    of the disc. Denote :math:`i_1(b)` as one of the representatives of the two edges.
    Define the associated integral

    .. math::

       C_b
       \equiv
       \frac{1}{L}
       \int_{S_{i_1(b)}} dz\,z\,b(z).    


    Then, the matrix :math:`D_{ab}` is constructed such that


    .. math::

       \mathcal{B}_{\ell_a}=\sum_{b=1}^E D_{ab}C_b.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected trivalent ribbon graph with one face.
    dependent_edge : int or None, optional
        Edge that is not used as moduli parameter.

    Returns
    -------
    numpy.ndarray
        Integer matrix of shape ``(n_edges - 1, n_edges)``.
    """

    edges = ribbon_graph[0]
    n_edges = len(edges)
    if dependent_edge is None:
        dependent_edge = n_edges - 1
    if isinstance(dependent_edge, bool) or not isinstance(dependent_edge, Integral):
        raise TypeError("dependent_edge must be an integer")
    dependent_edge = int(dependent_edge)
    if not 0 <= dependent_edge < n_edges:
        raise ValueError(
            f"dependent_edge={dependent_edge} is outside 0..{n_edges - 1}"
        )

    boundary_data = get_boundary_data(ribbon_graph)
    if boundary_data["n_faces"] != 1:
        raise ValueError("b-ghost directions require a one-face ribbon graph")
    genus = int(boundary_data["genus"])
    if n_edges != 6 * genus - 3:
        raise ValueError("b-ghost directions require a trivalent one-face graph")
    boundary_edges = boundary_data["edge_sequences"][0]
    occurrence_pairs = tuple(
        tuple(position for _, position in boundary_data["sewing"][edge])
        for edge in range(n_edges)
    )
    boundary_array = np.asarray(boundary_edges, dtype=np.int64)
    independent_edges = tuple(
        edge for edge in range(n_edges) if edge != dependent_edge
    )
    coefficients = np.zeros((n_edges - 1, n_edges), dtype=np.int64)
    for seam_edge, (first, second) in enumerate(occurrence_pairs):
        included_edges = np.concatenate(
            (boundary_array[: first + 1], boundary_array[:second])
        )
        counts = np.bincount(included_edges, minlength=n_edges)
        for row, independent_edge in enumerate(independent_edges):
            coefficients[row, seam_edge] = (
                int(counts[independent_edge]) - int(counts[dependent_edge])
            )
    return coefficients


def _unit_interval_gauss_jacobi(
    order: int,
    *,
    start_exponent: float,
    end_exponent: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Construct the Gauss-Jacobi quadrature nodes and weights given known
    divergences at endpoints.

    The function specifically returns parameters :math:`t_r` (the quadrature nodes) and :math:`w_r`
    (the weights) such that

    .. math::

       \int_0^1
       t^a(1-t)^b f(t)\,dt
       \simeq
       \sum_{r=1}^{N} w_r f(t_r).

    Parameters
    ----------
    order : int
        Positive number :math:`N` of integration points
    start_exponent : float
        Exponent :math:`a` of :math:`t^a`.
    end_exponent : float
        Exponent :math:`b` of :math:`(1-t)^b`.

    Returns
    -------
    tuple of numpy.ndarray
        Two ``float64`` arrays of shape ``(order,)``.  The first contains the
        quadrature nodes in the open interval :math:`(0,1)`; the second
        contains weights.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``order`` is not positive, either exponent is nonfinite, or either
        exponent is at most :math:`-1`.
    """

    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError("quadrature_order must be an integer")
    order = int(order)
    if order <= 0:
        raise ValueError("quadrature_order must be positive")
    start_exponent = float(start_exponent)
    end_exponent = float(end_exponent)
    if not math.isfinite(start_exponent) or not math.isfinite(end_exponent):
        raise ValueError("endpoint exponents must be finite")
    if start_exponent <= -1.0 or end_exponent <= -1.0:
        raise ValueError("Gauss--Jacobi endpoint exponents must exceed -1")

    # On x in [-1,1], Jacobi's conventional weight is
    # (1-x)^alpha (1+x)^beta.  Under x=2t-1 this corresponds to
    # t^beta (1-t)^alpha, hence alpha=end and beta=start.
    alpha = end_exponent
    beta = start_exponent
    diagonal = np.empty(order, dtype=np.float64)
    total = alpha + beta
    for n in range(order):
        denominator = (2.0 * n + total) * (2.0 * n + total + 2.0)
        if n == 0 and abs(denominator) <= np.finfo(float).eps:
            diagonal[n] = (beta - alpha) / (total + 2.0)
        else:
            diagonal[n] = (beta * beta - alpha * alpha) / denominator

    off_diagonal = np.empty(max(order - 1, 0), dtype=np.float64)
    for index, n in enumerate(range(1, order)):
        if n == 1:
            # Cancel the removable factor 1 + alpha + beta analytically.
            ratio = (
                4.0
                * (1.0 + alpha)
                * (1.0 + beta)
                / ((2.0 + total) ** 2 * (3.0 + total))
            )
        else:
            numerator = 4.0 * n * (n + alpha) * (n + beta) * (n + total)
            denominator = (
                (2.0 * n + total) ** 2
                * (2.0 * n + total + 1.0)
                * (2.0 * n + total - 1.0)
            )
            ratio = numerator / denominator
        off_diagonal[index] = math.sqrt(ratio)

    jacobi_matrix = np.diag(diagonal)
    if order > 1:
        jacobi_matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes_x, eigenvectors = np.linalg.eigh(jacobi_matrix)
    nodes = 0.5 * (nodes_x + 1.0)
    zeroth_moment = (
        math.gamma(start_exponent + 1.0)
        * math.gamma(end_exponent + 1.0)
        / math.gamma(start_exponent + end_exponent + 2.0)
    )
    weights = zeroth_moment * np.square(eigenvectors[0, :])
    return (
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _edge_quadrature_data(
    boundary_data: Mapping[str, object],
    *,
    quadrature_order: int,
    endpoint_exponent: float | None,
) -> tuple[tuple[ComplexArray, ...], tuple[ComplexArray, ...]]:
    r"""Precompute disk points and integration weights for every elementary seam.

    For the first boundary occurrence :math:`S_{i_1(e)}` of edge :math:`e`,
    parameterize the segment by

    .. math::

       u_e(t)=s_e+\ell_e t,
       \qquad
       z_e(t)=\exp\!\left(\frac{2\pi i\,u_e(t)}{L}\right),
       \qquad 0\leq t\leq 1,

    where :math:`s_e` is the segment's starting boundary coordinate,
    :math:`\ell_e` is its discretized length, and :math:`L` is the full disk
    boundary length.  The contour integration of the b-ghost along the segment
    :math:`e` becomes

    .. math::

       C_e
       =\frac{1}{L}\int_{S_{i_1(e)}}z b(z)\,dz
       =\frac{2\pi i\ell_e}{L^2}
        \int_0^1 z_e(t)^2 b\!\left(z_e(t)\right)\,dt.

    The returned data is the array :math:`W_{e,r}` and the points :math:`z_{e,r}` in

    .. math::

       C_e\simeq
       \sum_{r=1}^{N}W_{e,r}b(z_{e,r}),
       \qquad
       W_{e,r}=\frac{2\pi i\ell_e}{L^2}w_r z_{e,r}^2.

    Parameters
    ----------
    boundary_data : mapping
        Ribbon graph boundary data returned by
        :func:`~string_amplitudes.ribbon_graph_generator.get_boundary_data`.
    quadrature_order : int
        Number :math:`N` of quadrature nodes placed on each elementary seam integral.
    endpoint_exponent : float or None
        Common exponent :math:`a` used to account for endpoint behavior of the
        form :math:`t^a(1-t)^a`.  ``None`` selects Gauss--Legendre
        quadrature; otherwise Gauss--Jacobi quadrature is used.

    Returns
    -------
    tuple of tuple of numpy.ndarray
        A pair ``(points_by_edge, weights_by_edge)``.  Each outer tuple has
        one entry per ribbon-graph edge, and every entry is a ``complex128``
        array of shape ``(quadrature_order,)``.  ``points_by_edge[e][r]`` is
        :math:`z_{e,r}` and ``weights_by_edge[e][r]`` is the complete complex
        coefficient :math:`W_{e,r}` multiplying the correlator at that point.

    Notes
    -----
    When ``endpoint_exponent`` is supplied, the Gauss--Jacobi rule natively
    integrates :math:`t^a(1-t)^a f(t)`.  Because the caller supplies the full
    correlator rather than the regular factor :math:`f`, the Jacobi weights
    are divided by :math:`t_r^a(1-t_r)^a` before the contour Jacobian is
    applied.
    """

    occurrence_pairs = tuple(
        tuple(position for _, position in boundary_data["sewing"][edge])
        for edge in range(len(boundary_data["edge_lengths"]))
    )
    lengths = boundary_data["edge_lengths"]
    starts = boundary_data["boundary_segment_starts"][0]
    boundary_length = boundary_data["boundary_lengths"][0]

    if endpoint_exponent is None:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        unit_nodes = 0.5 * (np.asarray(nodes, dtype=np.float64) + 1.0)
        unit_weights = 0.5 * np.asarray(weights, dtype=np.float64)
    else:
        exponent = float(endpoint_exponent)
        unit_nodes, weighted_weights = _unit_interval_gauss_jacobi(
            quadrature_order,
            start_exponent=exponent,
            end_exponent=exponent,
        )
        endpoint_weight = unit_nodes**exponent * (1.0 - unit_nodes) ** exponent
        unit_weights = weighted_weights / endpoint_weight

    coefficient = np.complex128(
        2.0j * math.pi / float(boundary_length * boundary_length)
    )
    points_by_edge = []
    weights_by_edge = []
    for edge, (first, _) in enumerate(occurrence_pairs):
        start = starts[first]
        length = lengths[edge]
        boundary_coordinates = start + length * unit_nodes
        points = np.exp(
            2.0j * math.pi * boundary_coordinates / float(boundary_length)
        ).astype(np.complex128)
        seam_weights = (
            length * unit_weights * coefficient * np.square(points)
        ).astype(np.complex128)
        points_by_edge.append(points)
        weights_by_edge.append(seam_weights)
    return tuple(points_by_edge), tuple(weights_by_edge)


def integrated_bghost_edge_components(
    ribbon_graph,
    edge_lengths: Sequence[int],
    correlator: BCorrelator,
    *,
    b_count: int | None = None,
    quadrature_order: int = 2,
    endpoint_exponent: float | None = -2.0 / 3.0,
    max_correlator_evaluations: int | None = 1_000_000,
) -> dict[tuple[int, ...], np.complex128]:
    r"""Given a b-ghost correlator, integrate the holomorphic correlator over every elementary seam tuple.

    For an ordered set of distinct edges :math:`e_1<\cdots<e_q`, the returned
    component is

    .. math::

       C_{e_1\cdots e_q}
       =\left\langle C_{e_1}\cdots C_{e_q}c(0)\right\rangle,
       \qquad
       C_e=\frac{1}{L}\int_{S_{i_1(e)}}z b(z)\,dz.

    The provided callable ``correlator`` receives ordered :math:`q` disk points and
    must return the corresponding holomorphic correlator.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected trivalent one face ribbon graph.
    edge_lengths : sequence of int
        Positive discretization length of every graph edge.
    correlator : callable
        Function from an ordered sequence of :math:`b`-insertion points to a
        complex holomorphic correlator.
    b_count : int or None, optional
        Number :math:`q` of holomorphic :math:`b` insertions.  For
        :math:`\mathcal M_{g,1}`, the default is :math:`3g-2`.
    quadrature_order : int, optional
        Number of quadrature nodes on each elementary seam.
    endpoint_exponent : float or None, optional
        Exponent used to take care of the singularity at both seam endpoints with Gauss-Jacobi
        quadrature.  ``None`` selects ordinary Gauss-Legendre quadrature.
    max_correlator_evaluations : int or None, optional
        Upper bound on the number of correlator calls permitted.  ``None`` disables it.

    Returns
    -------
    dict
        Map from each increasing :math:`q`-tuple of edge indices to its integrated correlator.
    """

    if not callable(correlator):
        raise TypeError("correlator must be callable")
    n_edges = len(ribbon_graph[0])
    boundary_data = get_boundary_data(ribbon_graph, edge_lengths)
    if boundary_data["n_faces"] != 1:
        raise ValueError("b-ghost integration requires a one-face ribbon graph")
    genus = int(boundary_data["genus"])
    if n_edges != 6 * genus - 3:
        raise ValueError("b-ghost integration requires a trivalent one-face graph")
    if b_count is None:
        b_count = 3 * genus - 2
    if isinstance(b_count, bool) or not isinstance(b_count, Integral):
        raise TypeError("b_count must be an integer")
    b_count = int(b_count)
    if not 1 <= b_count <= n_edges:
        raise ValueError(f"b_count must lie in 1..{n_edges}")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral):
        raise TypeError("quadrature_order must be an integer")
    quadrature_order = int(quadrature_order)
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")

    evaluation_count = math.comb(n_edges, b_count) * quadrature_order**b_count
    if max_correlator_evaluations is not None:
        if (
            isinstance(max_correlator_evaluations, bool)
            or not isinstance(max_correlator_evaluations, Integral)
        ):
            raise TypeError("max_correlator_evaluations must be an integer or None")
        if evaluation_count > int(max_correlator_evaluations):
            raise ValueError(
                f"quadrature requires {evaluation_count:,} correlator evaluations, "
                f"exceeding max_correlator_evaluations={int(max_correlator_evaluations):,}"
            )

    points_by_edge, weights_by_edge = _edge_quadrature_data(
        boundary_data,
        quadrature_order=quadrature_order,
        endpoint_exponent=endpoint_exponent,
    )
    components: dict[tuple[int, ...], np.complex128] = {}
    for edge_tuple in combinations(range(n_edges), b_count):
        total = np.complex128(0.0)
        compensation = np.complex128(0.0)
        for node_tuple in product(range(quadrature_order), repeat=b_count):
            b_points = tuple(
                points_by_edge[edge][node]
                for edge, node in zip(edge_tuple, node_tuple)
            )
            value = np.complex128(correlator(b_points))
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"correlator returned a non-finite value for edges {edge_tuple}"
                )
            for edge, node in zip(edge_tuple, node_tuple):
                value *= weights_by_edge[edge][node]
            corrected = value - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        components[tuple(edge_tuple)] = total
    return components


def _integer_determinant(matrix: IntegerArray) -> int:
    """Compute the exact integer determinant of a matrix with integer entries using Bareiss elimination.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square two-dimensional matrix with integer entries.

    Returns
    -------
    int
        Exact determinant of ``matrix``.
    """

    values = np.asarray(matrix, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("determinant input must be square")
    size = values.shape[0]
    if size == 0:
        return 1
    work = [[int(entry) for entry in row] for row in values]
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign = -sign
        pivot = work[column][column]
        for row in range(column + 1, size):
            for target_column in range(column + 1, size):
                numerator = (
                    work[row][target_column] * pivot
                    - work[row][column] * work[column][target_column]
                )
                if numerator % previous_pivot:
                    raise ArithmeticError("Bareiss elimination lost exact divisibility")
                work[row][target_column] = numerator // previous_pivot
        previous_pivot = pivot
    return sign * work[-1][-1]


def bghost_measure_from_edge_components(
    edge_components: EdgeComponents,
    direction_coefficients: np.ndarray,
) -> np.complex128:
    r"""Given the elementary contour integrals of the bc ghost correlation function,
    computed in :func:`integrated_bghost_edge_components`, and the matrix :math:`D` mapping from the
    elementary contour integrals to the :math:`\mathcal{B}` ghost (determined in :func:`bghost_direction_coefficients`), compute the
    :math:`\mathcal{B}` top-form.

    Specifically, define the holomorphic top-form

    .. math::

       H_S
       =
       \left\langle
       \mathcal{B}^{(1,0)}_{\ell_{i_1}}
       \wedge\cdots\wedge
       \mathcal{B}^{(1,0)}_{\ell_{i_q}}
       c(0)
       \right\rangle.

    This function returns the coefficient of

    .. math::

       \left\langle
       \mathcal{B}_{\ell_1}
       \wedge\cdots\wedge
       \mathcal{B}_{\ell_d}
       c\widetilde{c}(0)
       \right\rangle
       =
       \sum_{\lvert S\rvert=q}
       \operatorname{sgn}(S,S^c)\,
       H_S\overline{H_{S^c}}

    with the ordering convention

    .. math::

       d\ell_1\wedge\cdots\wedge d\ell_d.


    Parameters
    ----------
    edge_components : mapping
        Map from tuples of edges to integrated holomorphic bc ghost correlators.
        Returned by :func:`integrated_bghost_edge_components`.
    direction_coefficients : numpy.ndarray
        Integer transition-function matrix returned by
        :func:`bghost_direction_coefficients`.

    Returns
    -------
    numpy.complex128
        Oriented coefficient of the full b ghost top form.
    """

    directions = np.asarray(direction_coefficients)
    if directions.ndim != 2:
        raise ValueError("direction_coefficients must be a two-dimensional array")
    if not np.issubdtype(directions.dtype, np.integer):
        if not np.all(np.isfinite(directions)) or not np.all(directions == np.rint(directions)):
            raise ValueError("direction_coefficients must contain exact integers")
    directions = np.asarray(np.rint(directions), dtype=np.int64)
    dimension, n_edges = directions.shape
    if dimension == 0 or dimension % 2:
        raise ValueError("the number of real moduli must be positive and even")
    b_count = dimension // 2
    if b_count > n_edges:
        raise ValueError("there are fewer seam edges than holomorphic insertions")

    expected_edge_tuples = set(combinations(range(n_edges), b_count))
    supplied: dict[tuple[int, ...], np.complex128] = {}
    for raw_key, raw_value in edge_components.items():
        key = tuple(int(edge) for edge in raw_key)
        if key not in expected_edge_tuples:
            raise ValueError(f"invalid or non-increasing edge tuple {key}")
        value = np.complex128(raw_value)
        if not np.isfinite(value):
            raise FloatingPointError(f"edge component {key} is not finite")
        supplied[key] = value
    missing = expected_edge_tuples.difference(supplied)
    if missing:
        raise ValueError(
            f"edge_components is incomplete: missing {len(missing)} of "
            f"{len(expected_edge_tuples)} components"
        )

    coordinate_subsets = tuple(combinations(range(dimension), b_count))
    pulled_back: dict[tuple[int, ...], np.complex128] = {}
    for subset in coordinate_subsets:
        value = np.complex128(0.0)
        rows = list(subset)
        for edge_tuple, component in supplied.items():
            minor = directions[np.ix_(rows, list(edge_tuple))]
            determinant = _integer_determinant(minor)
            if determinant:
                value += np.complex128(determinant) * component
        pulled_back[tuple(subset)] = value

    all_coordinates = tuple(range(dimension))
    total = np.complex128(0.0)
    for subset in coordinate_subsets:
        complement = tuple(
            coordinate for coordinate in all_coordinates if coordinate not in subset
        )
        inversions = sum(
            coordinate - position
            for position, coordinate in enumerate(subset)
        )
        wedge_sign = -1 if inversions % 2 else 1
        total += (
            np.complex128(wedge_sign)
            * pulled_back[tuple(subset)]
            * np.conjugate(pulled_back[complement])
        )
    return total


def bghost_measure(
    ribbon_graph,
    edge_lengths: Sequence[int],
    ghost_data: BCGhostCorrelatorData,
    *,
    c_point: complex = 0.0j,
    dependent_edge: int | None = None,
    num_integration_points: int = 2,
    endpoint_exponent: float | None = -2.0 / 3.0,
    theta_lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
    max_correlator_evaluations: int | None = 1_000_000,
) -> np.complex128:
    r"""Evaluate the :math:`bc` ghost measure in the disc frame using the ribbon graph parameterization of
    :math:`\mathcal M_{g,1}`.

    This function uses a b ghost of holomorphic weight :math:`\lambda=2`, one fixed :math:`c` insertion (default location is the center of the disc), and
    :math:`3g-2` integrated holomorphic :math:`b` insertions along the seams of the disc.
    The result is the oriented coefficient of the top form in the ordered independent edge-length coordinates.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected trivalent one-face ribbon graph.
    edge_lengths : sequence of int
        Positive discretization length of every graph edge.
    ghost_data : BCGhostCorrelatorData
        Fixed surface, Riemann constant, auxiliary divisor, sigma
        normalization, and chiral determinant-line data.  Its surface must be
        reconstructed from the supplied ribbon graph and edge lengths in the
        same disk coordinate.
    c_point : complex, optional
        Disk coordinate of the fixed :math:`c` insertion.
    dependent_edge : int or None, optional
        For fixed total length of the circumference, one of the edges is not a moduli parameter.
                `dependent_edge` is the index of the edge that is treated as a redundant parameter.
    num_integration_points : int, optional
        Number of integration points in the contour integral associated with each edge segment.
    endpoint_exponent : float or None, optional
        Exponent of the endpoint in the Gauss-Jacobi procedure used to compute b-ghost contour integration,
                or ``None`` for Gauss--Legendre. This is necessary due to the divergence of the bc ghost correlation function
                near the vertices.
    theta_lattice_cutoff : int or None, optional
        Integer cutoff for Riemann theta sums.
    tolerance : float, optional
        When theta lattice cutoff is not provided, an estimation is used to keep the Riemann theta sum error under `tolerance`. Default is :math:`10^{-12}`.
    max_correlator_evaluations : int or None, optional
        Upper bound on the number of correlator calls permitted.  ``None`` disables it.

    Returns
    -------
    numpy.complex128
        bc ghost top form coefficient.
    """

    if not isinstance(ghost_data, BCGhostCorrelatorData):
        raise TypeError("ghost_data must be BCGhostCorrelatorData")
    boundary_data = get_boundary_data(ribbon_graph, edge_lengths)
    if boundary_data["n_faces"] != 1:
        raise ValueError("the ghost measure requires a one-face ribbon graph")
    genus = int(boundary_data["genus"])
    if ghost_data.surface.genus != genus:
        raise ValueError(
            f"surface genus {ghost_data.surface.genus} does not match "
            f"graph genus {genus}"
        )
    c_point = complex(np.complex128(c_point))
    if not np.isfinite(c_point):
        raise ValueError("c_point must be finite")

    prepared_correlator = prepare_bc_correlator(
        ghost_data,
        lambda_weight=2.0,
        lattice_cutoff=theta_lattice_cutoff,
        tolerance=tolerance,
    )

    def correlator(b_points: Sequence[complex]) -> np.complex128:
        return prepared_correlator(
            b_points,
            (c_point,),
        )

    edge_components = integrated_bghost_edge_components(
        ribbon_graph,
        edge_lengths,
        correlator,
        b_count=3 * genus - 2,
        quadrature_order=num_integration_points,
        endpoint_exponent=endpoint_exponent,
        max_correlator_evaluations=max_correlator_evaluations,
    )
    directions = bghost_direction_coefficients(
        ribbon_graph,
        dependent_edge=dependent_edge,
    )
    return bghost_measure_from_edge_components(edge_components, directions)


def critical_bosonic_string_integrand(
    ribbon_graph,
    edge_lengths: Sequence[int],
    matter_partition_per_scalar: float,
    ghost_data: BCGhostCorrelatorData,
    *,
    c_point: complex = 0.0j,
    dependent_edge: int | None = None,
    num_integration_points: int = 2,
    endpoint_exponent: float | None = -2.0 / 3.0,
    theta_lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
    max_correlator_evaluations: int | None = 1_000_000,
) -> float:
    r"""Return the critical bosonic string integrand in ribbon graph coordinates.

    The quantity returned by this function is

    .. math::

       \left|\left\langle
       \mathcal B_{\ell_1}\wedge\cdots\wedge
       \mathcal B_{\ell_{6g-4}}c\widetilde c(0)
       \right\rangle\right|\left(Z_X^{(g)}\right)^{26}.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected trivalent ribbon graph with one face (return from :func:`generate_ribbon_graphs`
        or :func:`sample_ribbon_graph`).
    edge_lengths : sequence of int
        Positive discretization length of every graph edge.
    matter_partition_per_scalar : float
        Positive, physically normalized partition function of one noncompact
        scalar in the same conformal frame
    ghost_data : BCGhostCorrelatorData
        Fixed surface and normalization data used by the ghost correlator.
        Its surface must correspond to the supplied ribbon graph and edge
        lengths in the same disk coordinate.
    c_point : complex, optional
        Disk coordinate of the fixed :math:`c\widetilde c` insertion.
    dependent_edge : int or None, optional
        For fixed total length of the circumference, one of the edges is not a moduli parameter.
        `dependent_edge` is the index of the edge that is treated as a redundant parameter.
    num_integration_points : int, optional
        Number of integration points in the contour integral associated with each edge segment.
    endpoint_exponent : float or None, optional
        Exponent of the endpoint in the Gauss-Jacobi procedure used to compute b-ghost contour integration,
        or ``None`` for Gauss--Legendre. This is necessary due to the divergence of the bc ghost correlation function
        near the vertices.
    theta_lattice_cutoff : int or None, optional
        Integer cutoff for Riemann theta sums.
    tolerance : float, optional
        When theta lattice cutoff is not provided, an estimation is used to keep the Riemann theta sum error under `tolerance`. Default is :math:`10^{-12}`.
    max_correlator_evaluations : int or None, optional
        Upper bound on the number of correlator calls permitted.  ``None`` disables it.

    Returns
    -------
    float
        Critical bosonic string integrand.
    """

    if isinstance(matter_partition_per_scalar, bool) or not isinstance(
        matter_partition_per_scalar,
        Real,
    ):
        raise TypeError("matter_partition_per_scalar must be a real number")
    matter_partition_per_scalar = float(matter_partition_per_scalar)
    if not math.isfinite(matter_partition_per_scalar) or matter_partition_per_scalar <= 0.0:
        raise ValueError("matter_partition_per_scalar must be positive and finite")

    ghost_coefficient = bghost_measure(
        ribbon_graph,
        edge_lengths,
        ghost_data,
        c_point=c_point,
        dependent_edge=dependent_edge,
        num_integration_points=num_integration_points,
        endpoint_exponent=endpoint_exponent,
        theta_lattice_cutoff=theta_lattice_cutoff,
        tolerance=tolerance,
        max_correlator_evaluations=max_correlator_evaluations,
    )
    ghost_density = float(abs(ghost_coefficient))
    if ghost_density == 0.0:
        return 0.0
    log_density = math.log(ghost_density) + 26.0 * math.log(
        matter_partition_per_scalar
    )
    maximum_log = math.log(float(np.finfo(np.float64).max))
    minimum_log = math.log(float(np.nextafter(0.0, 1.0)))
    if not minimum_log <= log_density <= maximum_log:
        raise FloatingPointError(
            "the string integrand is outside the float64 range"
        )
    return float(math.exp(log_density))
__all__ = (
    "bghost_direction_coefficients",
    "bghost_measure",
    "bghost_measure_from_edge_components",
    "critical_bosonic_string_integrand",
    "integrated_bghost_edge_components",
)
