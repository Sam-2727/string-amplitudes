r"""Reconstruct the :math:`\alpha`-cycle normalized holomorphic one forms
   and period matrix on a given genus :math:`g` Riemann surface with one puncture,
   represented as a discretized ribbon graph.

The public objects are:

* :class:`DiscHolomorphicOneForm`: stores the data of a given holomorphic one form in
  the unit disc coordinates.
* :class:`PeriodMapResult`: stores the period matrix and its
  :math:`\alpha`-cycle normalized basis of holomorphic one forms.
* :func:`compute_period_map`: constructs the :math:`\alpha`-cycle normalized basis
  of holomorphic one forms and the corresponding period matrix, returning the results in the
  :class:`PeriodMapResult` format.

"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .ribbon_graph_generator import get_boundary_data
from .riemann_surface_holomorphic_data import holomorphic_one_form_antiderivatives


ComplexArray = NDArray[np.complex128]
IntegerArray = NDArray[np.int64]
Cycle = tuple[tuple[int, int], ...]
CyclePair = Mapping[str, Cycle]


@dataclass(frozen=True, slots=True)
class DiscHolomorphicOneForm:
    r"""Represent a holomorphic one form in the unit disc representation of
    a ribbon graph with one face. Specifically, in the expansion,

    .. math::

       \omega(z)=
       \prod_a\left(1-\frac{z}{p_a}\right)^{-1/3}
       \sum_{n=0}^{m-1}c_nz^n,

    it stores the coordinates :math:`c_n`, the singularity power :math:`\rho`, and
    the positions of the ribbon graph vertices :math:`p_a`.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Complex polynomial coefficients in increasing-power order.
    prevertices : numpy.ndarray
        Complex prevertices entering the factored singular term.
    singularity_power : float, optional
        Positive exponent of each inverse prevertex factor.
    """

    coefficients: ComplexArray
    prevertices: ComplexArray
    singularity_power: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        coefficients = np.array(self.coefficients, dtype=np.complex128, copy=True)
        prevertices = np.array(self.prevertices, dtype=np.complex128, copy=True)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("coefficients must be a nonempty one-dimensional array")
        if prevertices.ndim != 1 or prevertices.size == 0:
            raise ValueError("prevertices must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(prevertices)):
            raise ValueError("one-form data must be finite")
        singularity_power = float(self.singularity_power)
        if not math.isfinite(singularity_power) or singularity_power <= 0.0:
            raise ValueError("singularity_power must be finite and positive")
        coefficients.setflags(write=False)
        prevertices.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "prevertices", prevertices)
        object.__setattr__(self, "singularity_power", singularity_power)

    def __call__(
        self,
        point: complex | np.ndarray,
    ) -> np.complex128 | ComplexArray:
        """Evaluate the coefficient of the one-form at one or more points."""

        points = np.asarray(point, dtype=np.complex128)
        singular = np.prod(
            (1.0 - points[..., None] / self.prevertices)
            ** (-self.singularity_power),
            axis=-1,
        )
        polynomial = np.polynomial.polynomial.polyval(points, self.coefficients)
        result = np.asarray(singular * polynomial, dtype=np.complex128)
        if result.ndim == 0:
            return np.complex128(result)
        return result


@dataclass(frozen=True, slots=True)
class PeriodMapResult:
    r"""The period matrix and the A-cycle normalized holomorphic one forms.

    Parameters
    ----------
    period_matrix : numpy.ndarray
        Symmetric genus-g period matrix in the Siegel upper half-space.
    normalized_one_forms : tuple of DiscHolomorphicOneForm
        A-normalized basis satisfying
        :math:`\oint_{A_J}\omega_I=\delta_{IJ}`.
    """

    period_matrix: ComplexArray
    normalized_one_forms: tuple[DiscHolomorphicOneForm, ...]

    def __post_init__(self) -> None:
        period_matrix = np.array(self.period_matrix, dtype=np.complex128, copy=True)
        forms = tuple(self.normalized_one_forms)
        genus = len(forms)
        if genus < 1:
            raise ValueError("normalized_one_forms must not be empty")
        if period_matrix.shape != (genus, genus):
            raise ValueError(
                f"period_matrix must have shape ({genus}, {genus}), "
                f"got {period_matrix.shape}"
            )
        if not np.all(np.isfinite(period_matrix)):
            raise ValueError("period_matrix contains a non-finite entry")
        if not np.allclose(period_matrix, period_matrix.T, rtol=1e-10, atol=1e-11):
            raise ValueError("period_matrix must be symmetric")
        if float(np.min(np.linalg.eigvalsh(np.imag(period_matrix)))) <= 0.0:
            raise ValueError(
                "the imaginary part of period_matrix must be positive definite"
            )
        if not all(isinstance(form, DiscHolomorphicOneForm) for form in forms):
            raise TypeError(
                "normalized_one_forms must contain DiscHolomorphicOneForm objects"
            )
        period_matrix.setflags(write=False)
        object.__setattr__(self, "period_matrix", period_matrix)
        object.__setattr__(self, "normalized_one_forms", forms)

    @property
    def genus(self) -> int:
        """Return the genus, corresponding to the length of the basis of normalized
           one forms."""

        return len(self.normalized_one_forms)


def _signed_chord_intersection(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    r"""Take two pairs of sewn together edges :math:`(a_1,a_2),(b_1,b_2)`
       with :math:`a_1<a_2,b_1<b_2`. Define chords traversing from
       :math:`a_1` to :math:`a_2` as :math:`a` and :math:`b_1` to :math:`b_2`
       as :math:`b`. Define the tangent vectors of the chords as
       :math:`\dot{a},\dot{b}`. If the ordered pair :math:`(\dot{a},\dot{b})`
       agrees with the orientation of the disc, the intersection is determined to be
       positive (returns +1), and if not, the intersection is negative (returns -1). 
       If there is no intersection, returns 0."""

    first_start, first_end = sorted(first)
    second_start, second_end = sorted(second)
    if first_start < second_start < first_end < second_end:
        return 1
    if second_start < first_start < second_end < first_end:
        return -1
    return 0


def _chord_intersection_matrix(
    edge_occurrences: Mapping[int, tuple[int, int]],
    *,
    number_of_edges: int,
    genus: int,
) -> IntegerArray:
    """Repeatedly calls :func:`_signed_chord_intersection` to build a matrix
       :math:`I_{ij}`
       of the intersection number (either -1, 0, or 1) between the chords of edges
       :math:`e_i` and :math:`e_j`."""

    matrix = np.zeros((number_of_edges, number_of_edges), dtype=np.int64)
    for first in range(number_of_edges):
        if first not in edge_occurrences:
            raise ValueError(f"edge {first} is absent from the boundary word")
        for second in range(first + 1, number_of_edges):
            if second not in edge_occurrences:
                raise ValueError(f"edge {second} is absent from the boundary word")
            intersection = _signed_chord_intersection(
                edge_occurrences[first],
                edge_occurrences[second],
            )
            matrix[first, second] = intersection
            matrix[second, first] = -intersection
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 2 * genus:
        raise ValueError(
            "the boundary-chord intersection matrix has rank "
            f"{rank}; expected {2 * genus}"
        )
    return matrix


def _cycle_vector_to_terms(vector: IntegerArray) -> Cycle:
    r"""Converts one representation of a cycle to a different representation.
       Denote the elementary chords on the disc as
       :math:`c_i:p_i^{(1)}\to p_i^{(2)}`,
       i.e. the chords connecting the midpoints :math:`p_i^{(j)}` of the sewn segments
       associated to edge i. A given cycle can then be represented as

       .. math::

          \gamma_v = \sum_{i=0}^{E-1} v_i c_i.

       The input to this function is an ordered collection of :math:`\{v_i\}`,
       with each :math:`v_i` an integer. The output is an unordered collection 
       :math:`\{i,v_i\}`, with i the edge label."""

    return tuple(
        (edge, int(coefficient))
        for edge, coefficient in enumerate(vector)
        if coefficient
    )


def _validate_cycle_basis(
    intersection_matrix: IntegerArray,
    basis_pairs: Sequence[Mapping[str, Sequence[tuple[int, int]]]],
    *,
    genus: int,
) -> tuple[dict[str, Cycle], ...]:
    r"""Confirms that a supplied collection of candidate A and B cycles forms a
    basis of homology cycles. Denote the elementary chords on the disc as
    :math:`c_i:p_i^{(1)}\to p_i^{(2)}`, i.e. the chords connecting the
    midpoints :math:`p_i^{(j)}` of the sewn segments associated to edge i. A
    given cycle can then be represented as

    .. math::

       \gamma_v = \sum_{i=0}^{E-1} v_i c_i.

    The input to this function is an unordered collection
    :math:`\{i,v_i\}`, with i the edge label for each candidate A, B cycle.
    """

    number_of_edges = intersection_matrix.shape[0]
    if len(basis_pairs) != genus:
        raise ValueError(f"expected {genus} cycle pairs, got {len(basis_pairs)}")
    alpha_vectors = []
    beta_vectors = []
    canonical = []
    for pair_index, pair in enumerate(basis_pairs):
        vectors = {}
        terms_by_name = {}
        for name in ("alpha", "beta"):
            if name not in pair:
                raise ValueError(f"cycle pair {pair_index} is missing {name!r}")
            vector = np.zeros(number_of_edges, dtype=np.int64)
            for edge, coefficient in pair[name]:
                if (
                    isinstance(edge, bool)
                    or not isinstance(edge, Integral)
                    or not 0 <= int(edge) < number_of_edges
                ):
                    raise ValueError(f"invalid edge index {edge!r} in {name} cycle")
                if isinstance(coefficient, bool) or not isinstance(
                    coefficient,
                    Integral,
                ):
                    raise TypeError("cycle coefficients must be integers")
                vector[int(edge)] += int(coefficient)
            if not np.any(vector):
                raise ValueError(f"{name} cycle {pair_index} is zero")
            vectors[name] = vector
            terms_by_name[name] = _cycle_vector_to_terms(vector)
        alpha_vectors.append(vectors["alpha"])
        beta_vectors.append(vectors["beta"])
        canonical.append(terms_by_name)
    alpha = np.stack(alpha_vectors)
    beta = np.stack(beta_vectors)
    zero = np.zeros((genus, genus), dtype=np.int64)
    if not np.array_equal(alpha @ intersection_matrix @ alpha.T, zero):
        raise ValueError("alpha cycles are not mutually isotropic")
    if not np.array_equal(beta @ intersection_matrix @ beta.T, zero):
        raise ValueError("beta cycles are not mutually isotropic")
    if not np.array_equal(
        alpha @ intersection_matrix @ beta.T,
        np.eye(genus, dtype=np.int64),
    ):
        raise ValueError("the supplied cycles are not a symplectic basis")
    return tuple(canonical)


def _short_cycle_candidates(
    number_of_edges: int,
    *,
    maximum_support: int = 3,
) -> tuple[IntegerArray, ...]:
    r"""Enumerate list of candidate A, B cycles, which are combinations of the
       elementary chords on the ribbon graph. The parameter ``maximum_support``
       is the number of elementary chords that can have nonzero inclusion in the
       candidate A, B cycles. The number of candidates generated is
       
       .. math::

          \sum_{k=1}^{\min(\text{maximum support},E)} \binom{E}{k} 2^{k-1}.
       
       """

    candidates = []
    for support_size in range(
        1,
        min(int(maximum_support), number_of_edges) + 1,
    ):
        for support in combinations(range(number_of_edges), support_size):
            for signs in product((-1, 1), repeat=support_size):
                vector = np.zeros(number_of_edges, dtype=np.int64)
                vector[list(support)] = signs
                if vector[support[0]] < 0:
                    vector = -vector
                candidates.append(vector)
    candidates.sort(
        key=lambda vector: (
            int(np.count_nonzero(vector)),
            tuple(int(index) for index in np.flatnonzero(vector)),
            tuple(int(value) for value in vector[np.flatnonzero(vector)]),
        )
    )
    return tuple(candidates)


def _find_symplectic_basis(
    intersection_matrix: IntegerArray,
    *,
    genus: int,
) -> tuple[dict[str, Cycle], ...]:
    r"""Find one valid basis of A, B cycles. It first selects 
       a candidate for :math:`\alpha_1`, expressed as a vector in terms
       of the elementary chords on the disc. Then it finds a candidate
       :math:`\beta_1` satisfying :math:`\alpha_1^T I\beta_1=\pm 1` (reversing
       the orientation of :math:`\beta_1` if necessary such that
       :math:`\alpha_1^T I\beta_1=+1`), where :math:`I` is the intersection
       matrix, with :math:`I_{ab}` representing the intersection number between
       chords a,b. For each i, it then finds an :math:`\alpha_i`
       such that :math:`\alpha_{j}^T I \alpha_i=0`,
       :math:`\beta_j^T I \alpha_i=0`, :math:`j<i` and proceeds."""

    candidates = _short_cycle_candidates(intersection_matrix.shape[0])
    alphas: list[IntegerArray] = []
    betas: list[IntegerArray] = []

    def pairing(first: IntegerArray, second: IntegerArray) -> int:
        return int(first @ intersection_matrix @ second)

    def equivalent(first: IntegerArray, second: IntegerArray) -> bool:
        return np.array_equal(first, second) or np.array_equal(first, -second)

    def selected(candidate: IntegerArray) -> bool:
        return any(equivalent(candidate, value) for value in (*alphas, *betas))

    def search() -> bool:
        if len(alphas) == genus:
            return True
        for alpha in candidates:
            if selected(alpha):
                continue
            if any(pairing(alpha, value) for value in (*alphas, *betas)):
                continue
            alphas.append(alpha)
            for beta in candidates:
                if selected(beta) or equivalent(alpha, beta):
                    continue
                alpha_beta = pairing(alpha, beta)
                if abs(alpha_beta) != 1:
                    continue
                oriented_beta = beta * alpha_beta
                if any(pairing(value, oriented_beta) for value in alphas[:-1]):
                    continue
                if any(pairing(value, oriented_beta) for value in betas):
                    continue
                betas.append(oriented_beta)
                if search():
                    return True
                betas.pop()
            alphas.pop()
        return False

    if not search():
        raise ValueError(
            "no symplectic basis was found among edge-chord combinations "
            "with support at most three"
        )
    pairs = tuple(
        {
            "alpha": _cycle_vector_to_terms(alpha),
            "beta": _cycle_vector_to_terms(beta),
        }
        for alpha, beta in zip(alphas, betas)
    )
    return _validate_cycle_basis(intersection_matrix, pairs, genus=genus)


def _choose_pivots(null_subspace: ComplexArray) -> IntegerArray:
    r"""Suppose that ``null_subspace`` has n columns. This function selects n rows
       such that the resulting :math:`n\times n` matrix is invertible.
       
       Amongst the choices of invertible submatrices, a heuristic algorithm optimizes
       to select a choice with a large minimum singular value such that the matrix is
       well conditioned."""

    number_of_forms = null_subspace.shape[1]
    chosen: list[int] = []
    remaining = set(range(null_subspace.shape[0]))
    for _ in range(number_of_forms):
        best_index = None
        best_score = -1.0
        for index in remaining:
            block = null_subspace[chosen + [index], :]
            score = float(np.linalg.svd(block, compute_uv=False)[-1])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            raise RuntimeError("failed to choose one-form normalization pivots")
        chosen.append(best_index)
        remaining.remove(best_index)
    return np.asarray(chosen, dtype=np.int64)


def _raw_holomorphic_forms(
    boundary_segment_starts: Sequence[int],
    edge_occurrences: Mapping[int, tuple[int, int]],
    edge_lengths: Sequence[int],
    boundary_length: int,
    *,
    genus: int,
    zero_column_tolerance: float | None,
) -> tuple[DiscHolomorphicOneForm, ...]:
    r"""From discretized ribbon graph data, construct an unnormalized basis
       of holomorphic one forms.
       
       Given an edge site :math:`s`, define the identified
       points on the edge of the disc associated with this site as
       :math:`z_s^R,z_s^L`. The holomorphic one form is formulated in the
       global coordinate :math:`z` of the disc as
           
       .. math::

       \omega(z) = f(z)\,dz,

       where

       .. math::

          f(z)
          =
          \prod_a \left(1-\frac{z}{p_a}\right)^{-1/3}
          \sum_{k=0}^{H-1} c_k z^k.
       
       The condition for the one-form to be a holomorphic form on the genus g topology
       associated with the sewing is that
        
       .. math::

          z_s^R f\left(z_s^R\right)
          +
          z_s^L f\left(z_s^L\right)
          =
          0.

       For all identified pairs :math:`z_s^R,z_s^L`.
       """

    half_boundary_length = boundary_length // 2
    angular_scale = 2.0 * math.pi / boundary_length
    half_step = math.pi / boundary_length
    singular_phases = angular_scale * np.asarray(
        boundary_segment_starts,
        dtype=np.float64,
    )
    prevertices = np.exp(1j * singular_phases)
    right_angles = []
    left_angles = []
    for edge in range(len(edge_lengths)):
        length = edge_lengths[edge]
        first, second = edge_occurrences[edge]
        sites = np.arange(1, length + 1, dtype=np.float64)
        right_angles.append(
            angular_scale * (boundary_segment_starts[first] + sites) - half_step
        )
        left_angles.append(
            angular_scale
            * (boundary_segment_starts[second] + length + 1 - sites)
            - half_step
        )
    right_angles_array = np.concatenate(right_angles)
    left_angles_array = np.concatenate(left_angles)

    def singular_factor(angles: NDArray[np.float64]) -> ComplexArray:
        phase_difference = np.exp(
            1j * (angles[:, None] - singular_phases[None, :])
        )
        return np.asarray(
            np.prod((1.0 - phase_difference) ** (-1.0 / 3.0), axis=1),
            dtype=np.complex128,
        )

    powers = np.arange(1, half_boundary_length + 1, dtype=np.float64)
    seam_matrix = (
        singular_factor(right_angles_array)[:, None]
        * np.exp(1j * np.outer(right_angles_array, powers))
        + singular_factor(left_angles_array)[:, None]
        * np.exp(1j * np.outer(left_angles_array, powers))
    )
    column_norms = np.linalg.norm(seam_matrix, axis=0)
    if zero_column_tolerance is None:
        scale = float(np.max(column_norms)) if column_norms.size else 1.0
        zero_column_tolerance = (
            max(seam_matrix.shape)
            * np.finfo(float).eps
            * max(scale, 1.0)
            * 100.0
        )
    active = column_norms > float(zero_column_tolerance)
    if genus == 1 and active.size:
        active[0] = True
    active_matrix = seam_matrix[:, active]
    if active_matrix.shape[1] < genus:
        raise ValueError(
            f"only {active_matrix.shape[1]} active columns remain for genus {genus}"
        )
    gram_matrix = active_matrix.conj().T @ active_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    if not np.all(np.isfinite(eigenvalues[:genus])):
        raise np.linalg.LinAlgError("the seam eigensolve returned non-finite values")
    null_subspace = np.asarray(eigenvectors[:, :genus], dtype=np.complex128)
    pivots = _choose_pivots(null_subspace)
    selector = np.zeros((genus, active_matrix.shape[1]), dtype=np.complex128)
    selector[np.arange(genus), pivots] = 1.0
    coefficients_active = np.linalg.solve(
        gram_matrix + selector.conj().T @ selector,
        selector.conj().T,
    )
    pivot_values = selector @ coefficients_active
    if float(np.linalg.svd(pivot_values, compute_uv=False)[-1]) <= 1e-12:
        raise np.linalg.LinAlgError("the one-form normalization pivots are singular")
    coefficients_active = coefficients_active @ np.linalg.inv(pivot_values)
    forms = []
    for index in range(genus):
        coefficients = np.zeros(half_boundary_length, dtype=np.complex128)
        coefficients[active] = coefficients_active[:, index]
        coefficients.real[np.abs(coefficients.real) < 1e-15] = 0.0
        coefficients.imag[np.abs(coefficients.imag) < 1e-15] = 0.0
        forms.append(
            DiscHolomorphicOneForm(
                coefficients=coefficients,
                prevertices=prevertices,
            )
        )
    return tuple(forms)


def _edge_midpoint_pairs(
    boundary_segment_starts: Sequence[int],
    edge_occurrences: Mapping[int, tuple[int, int]],
    edge_lengths: Sequence[int],
) -> dict[int, tuple[np.complex128, np.complex128]]:
    """Find the discretized disc coordinate of the midpoint of each edge
       of the ribbon graph. At the level of the discretized disc, each ribbon
       graph edge is represented as two sewed together edges of the disc."""

    angular_scale = 2.0 * math.pi / (2 * sum(edge_lengths))
    pairs = {}
    for edge, (first, second) in edge_occurrences.items():
        offset = 0.5 * edge_lengths[edge]
        pairs[edge] = (
            np.complex128(
                np.exp(
                    1j
                    * angular_scale
                    * (boundary_segment_starts[first] + offset)
                )
            ),
            np.complex128(
                np.exp(
                    1j
                    * angular_scale
                    * (boundary_segment_starts[second] + offset)
                )
            ),
        )
    return pairs


def _raw_a_b_period_matrices(
    forms: Sequence[DiscHolomorphicOneForm],
    basis_pairs: Sequence[CyclePair],
    midpoint_pairs: Mapping[int, tuple[complex, complex]],
    *,
    quadrature_order: int,
) -> tuple[ComplexArray, ComplexArray]:
    r"""Integrate unnormalized holomorphic one forms over all A, B cycles. 
       These are assembled into the matrices
       
       .. math::

          A_{IJ} = \oint_{\alpha_J} \omega_I.
       
       and 
       
       .. math::

          B_{IJ} = \oint_{\beta_J} \omega_I.
    """

    primitives = holomorphic_one_form_antiderivatives(
        forms,
        quadrature_order=quadrature_order,
    )
    genus = len(forms)
    a_periods = np.zeros((genus, genus), dtype=np.complex128)
    b_periods = np.zeros((genus, genus), dtype=np.complex128)

    def period(primitive, cycle: Cycle) -> np.complex128:
        value = np.complex128(0.0)
        for edge, coefficient in cycle:
            start, end = midpoint_pairs[edge]
            value += coefficient * (primitive(end) - primitive(start))
        return value

    for form_index, primitive in enumerate(primitives):
        for cycle_index, pair in enumerate(basis_pairs):
            a_periods[form_index, cycle_index] = period(
                primitive, pair["alpha"]
            )
            b_periods[form_index, cycle_index] = period(
                primitive, pair["beta"]
            )
    return a_periods, b_periods


def compute_period_map(
    ribbon_graph: tuple,
    edge_lengths: Sequence[int],
    *,
    period_quadrature_order: int = 256,
    cycle_basis: Sequence[Mapping[str, Sequence[tuple[int, int]]]] | None = None,
    zero_column_tolerance: float | None = None,
) -> PeriodMapResult:
    r"""Given discretized ribbon graph data representing a genus g Riemann surface with
    one puncture, compute the associated period matrix.

    Parameters
    ----------
    ribbon_graph : tuple
        One face (i.e. one vertex operator) trivalent ribbon graph represented by
        ``(edges, vertices, cyclic ordering of edges at each vertex)``.
    edge_lengths : sequence of int
        Discretization length of every edge, ordered by edge index.
    period_quadrature_order : int, optional
        The number of Gauss-Legendre nodes used in the numerical evaluation of integrals.
    cycle_basis : sequence of mapping or None, optional
        Optional basis of A and B cycles in terms of the edge data and elementary cycles between the sewn together
        edges of the disc. For example,
            
        .. code-block:: python

           cycle_basis = (
               {
                "alpha": ((0, 1), (2, -1)),
                "beta": ((1, 1),),
               },
           )

        means the the A cycle is :math:`c_0-c_2`, where :math:`c_i` is the chord connecting
        the sewn together midpoints of edge i. Meanwhile, the B cycle would be simply
        :math:`c_1`. A valid cycle basis must satisfy

        .. math::

           \alpha_I \cdot \alpha_J = 0,
           \qquad
           \beta_I \cdot \beta_J = 0,
           \qquad
           \alpha_I \cdot \beta_J = \delta_{IJ}.

        If no cycle basis is provided, the function will construct one.

    zero_column_tolerance : float or None, optional
        Tolerance level under which columns of the matrix :math:`M` (constructed in
        :func:`_raw_holomorphic_forms`) encoding the sewing conditions on the coefficients of the holomorphic
        one form are treated as zero and excluded from the algorithm solving for the holomorphic one form.
        That is, it excludes columns for which
        :math:`\lVert M_{:,k}\rVert\leq\mathrm{zero\_column\_tolerance}`.

    Returns
    -------
    PeriodMapResult
        Period matrix and A-normalized holomorphic one-forms.

    """

    if (
        isinstance(period_quadrature_order, bool)
        or not isinstance(period_quadrature_order, Integral)
    ):
        raise TypeError("period_quadrature_order must be an integer")
    period_quadrature_order = int(period_quadrature_order)
    if period_quadrature_order < 2:
        raise ValueError("period_quadrature_order must be at least two")
    edges = ribbon_graph[0]
    boundary_data = get_boundary_data(ribbon_graph, edge_lengths)
    number_of_faces = int(boundary_data["n_faces"])
    if number_of_faces != 1:
        raise ValueError(
            "compute_period_map currently requires a one-face ribbon graph; "
            f"received {number_of_faces} faces"
        )
    genus = int(boundary_data["genus"])
    if genus <= 0:
        raise ValueError("the ribbon graph must have positive genus")
    lengths = boundary_data["edge_lengths"]
    boundary_segment_starts = boundary_data["boundary_segment_starts"][0]
    edge_occurrences = {
        edge_index: tuple(position for _, position in positions)
        for edge_index, positions in boundary_data["sewing"].items()
    }
    intersection_matrix = _chord_intersection_matrix(
        edge_occurrences,
        number_of_edges=len(edges),
        genus=genus,
    )
    if cycle_basis is None:
        basis_pairs = _find_symplectic_basis(
            intersection_matrix,
            genus=genus,
        )
    else:
        basis_pairs = _validate_cycle_basis(
            intersection_matrix,
            cycle_basis,
            genus=genus,
        )
    raw_forms = _raw_holomorphic_forms(
        boundary_segment_starts,
        edge_occurrences,
        lengths,
        boundary_data["boundary_lengths"][0],
        genus=genus,
        zero_column_tolerance=zero_column_tolerance,
    )
    a_periods, b_periods = _raw_a_b_period_matrices(
        raw_forms,
        basis_pairs,
        _edge_midpoint_pairs(
            boundary_segment_starts,
            edge_occurrences,
            lengths,
        ),
        quadrature_order=period_quadrature_order,
    )
    normalization = np.linalg.inv(a_periods)
    raw_period_matrix = normalization @ b_periods
    symmetry_scale = max(
        float(np.linalg.norm(raw_period_matrix)),
        np.finfo(float).tiny,
    )
    relative_symmetry_defect = float(
        np.linalg.norm(raw_period_matrix - raw_period_matrix.T) / symmetry_scale
    )
    if relative_symmetry_defect > 1.0e-2:
        raise ValueError(
            "the relative period-matrix symmetry defect is "
            f"{relative_symmetry_defect:.3e}; increase the discretization or "
            "period_quadrature_order"
        )
    period_matrix = 0.5 * (raw_period_matrix + raw_period_matrix.T)
    if float(np.min(np.linalg.eigvalsh(np.imag(period_matrix)))) <= 0.0:
        raise ValueError(
            "the reconstructed period matrix is outside the Siegel upper "
            "half-space; increase the discretization or quadrature order"
        )
    normalized_forms = []
    for form_index in range(genus):
        coefficients = np.zeros_like(raw_forms[0].coefficients)
        for weight, raw_form in zip(normalization[form_index], raw_forms):
            coefficients += weight * raw_form.coefficients
        normalized_forms.append(
            DiscHolomorphicOneForm(
                coefficients=coefficients,
                prevertices=raw_forms[0].prevertices,
                singularity_power=raw_forms[0].singularity_power,
            )
        )
    return PeriodMapResult(
        period_matrix=period_matrix,
        normalized_one_forms=tuple(normalized_forms),
    )


__all__ = (
    "DiscHolomorphicOneForm",
    "PeriodMapResult",
    "compute_period_map",
)
