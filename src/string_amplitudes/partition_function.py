r"""This module numerically computes free boson partition functions on a given
ribbon graph, represented as a sewed together discs. It can compute both compact and
non-compact partition functions. Currently, it supports genus :math:`g` with 1 puncture.

The public functions are:

* :func:`identity_wavefunctional_kernel`: constructs the discretized identity
  wavefunctional kernel.
* :func:`identity_wavefunctional_kernel_reduced`: constructs the sewn identity
  wavefunctional kernel with the zero mode subtracted and sewn edges identified.
* :func:`matter_log_determinant`: computes the logarithm of the determinant of
  the free boson quadratic form.
* :func:`compact_boson_partition_function`: computes the compact boson
  partition function by truncating over the winding sum.
"""

from collections import deque
from dataclasses import dataclass
from itertools import islice, product
import math
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .ribbon_graph_generator import get_boundary_data


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DEFAULT_MAX_LATTICE_POINTS = 10_000_000


@dataclass(frozen=True, slots=True)
class _OneFaceGluingData:
    """Validated boundary-site data shared by both partition functions."""

    genus: int
    edge_lengths: IntArray
    boundary_length: int
    independent_sites: IntArray
    partner_sites: IntArray
    prime: IntArray
    edge_points: tuple[IntArray, ...]
    edge_independent_columns: tuple[IntArray, ...]
    oriented_edges: tuple[tuple[int, int], ...]


def _cholesky_factor_and_log_det(
    matrix: np.ndarray,
    *,
    symmetrize: bool = False,
) -> tuple[np.ndarray, float]:
    r"""Return the Cholesky factor (i.e. the lower triangular matrix
       :math:`L` satisfying :math:`A=L L^\dagger`) and log determinant
       of a positive definite matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Real or complex square matrix to factor.
    symmetrize : bool, optional
        Whether to replace ``matrix`` by its Hermitian part before factoring.

    Returns
    -------
    tuple of (numpy.ndarray, float)
        Lower-triangular Cholesky factor, with the same numerical dtype as the
        processed matrix, and the real-valued natural logarithm of its
        determinant.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains a non-finite entry.")
    if symmetrize:
        matrix = 0.5 * (matrix + matrix.conj().T)
    try:
        lower = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "Cholesky factorization failed: matrix is not positive definite."
        ) from error
    diagonal = np.real(np.diag(lower))
    if np.any(diagonal <= 0.0):
        raise np.linalg.LinAlgError(
            "Cholesky factor has a non-positive diagonal."
        )
    logdet = float(2.0 * np.sum(np.log(diagonal)))
    return lower, logdet


def _cholesky_solve(
    lower: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    r"""Solve the matrix equation :math:`AX=B` using a provided cholesky
       decomposition.

    Parameters
    ----------
    lower : numpy.ndarray
        Real or complex square lower-triangular Cholesky factor of ``A``.
    right_hand_side : numpy.ndarray
        Real or complex vector of shape ``(n,)`` or matrix of shape ``(n, m)``.

    Returns
    -------
    numpy.ndarray
        Solution with the same shape as ``right_hand_side`` and a dtype
        obtained from ``lower`` and ``right_hand_side``.
    """
    lower = np.asarray(lower)
    right_hand_side = np.asarray(
        right_hand_side,
        dtype=np.result_type(lower.dtype, right_hand_side),
    )
    was_vector = right_hand_side.ndim == 1
    if was_vector:
        right_hand_side = right_hand_side[:, None]
    if right_hand_side.ndim != 2:
        raise ValueError("right_hand_side must be a vector or a matrix.")
    if right_hand_side.shape[0] != lower.shape[0]:
        raise ValueError(
            "right_hand_side has an incompatible leading dimension: "
            f"{right_hand_side.shape[0]} != {lower.shape[0]}."
        )

    # Explicit triangular substitution avoids a second dense factorization
    # and does not require SciPy solely for solve_triangular.
    forward = np.empty_like(right_hand_side)
    for row in range(lower.shape[0]):
        previous = lower[row, :row] @ forward[:row]
        forward[row] = (right_hand_side[row] - previous) / lower[row, row]

    solution = np.empty_like(forward)
    for row in range(lower.shape[0] - 1, -1, -1):
        following = lower[row + 1 :, row].conj() @ solution[row + 1 :]
        solution[row] = (forward[row] - following) / lower[row, row].conj()
    return solution[:, 0] if was_vector else solution


def _validate_winding_sum_parameters(
    radius,
    lattice_cutoff,
    chunk_size,
    max_lattice_points,
    *,
    dimension: int,
) -> tuple[float, int, int, int | None, int]:
    """Validate parameters in the sum over windings for the compact boson.
       In particular, radius must be a real number, lattice_cutoff is a 
       a nonnegative integer, dimension (should be twice the genus) is even,
       chunk_size is is positive, and max_lattice_points is a positive integer.

    Parameters
    ----------
    radius : float
        Compactification radius.
    lattice_cutoff : int
        Nonnegative component-wise cutoff for winding vectors.
    chunk_size : int
        Positive number of winding vectors evaluated in one batch.
    max_lattice_points : int or None
        Positive upper bound on the total number of winding vectors, or
        ``None`` to disable the work guard.
    dimension : int
        Nonnegative even winding-lattice dimension, equal to twice the genus.

    Returns
    -------
    tuple of (float, int, int, int or None, int)
        Validated radius, lattice cutoff, chunk size, lattice-point limit, and
        total lattice-point count.
    """
    if isinstance(radius, (bool, np.bool_)):
        raise TypeError("radius must be a real number, not a Boolean.")
    radius = float(radius)
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError(f"radius must be positive and finite, got {radius}.")

    for name, value, allow_zero in (
        ("lattice_cutoff", lattice_cutoff, True),
        ("chunk_size", chunk_size, False),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, not a Boolean.")
        if int(value) < (0 if allow_zero else 1):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}, got {value}.")
    lattice_cutoff = int(lattice_cutoff)
    chunk_size = int(chunk_size)

    if isinstance(dimension, bool) or not isinstance(dimension, Integral):
        raise TypeError("dimension must be a nonnegative even integer.")
    dimension = int(dimension)
    if dimension < 0 or dimension % 2:
        raise ValueError("dimension must be a nonnegative even integer 2g.")

    if max_lattice_points is not None:
        if isinstance(max_lattice_points, (bool, np.bool_)) or not isinstance(
            max_lattice_points, Integral
        ):
            raise TypeError(
                "max_lattice_points must be a positive integer or None."
            )
        max_lattice_points = int(max_lattice_points)
        if max_lattice_points <= 0:
            raise ValueError("max_lattice_points must be positive or None.")

    lattice_point_count = (2 * lattice_cutoff + 1) ** dimension
    if (
        max_lattice_points is not None
        and lattice_point_count > max_lattice_points
    ):
        raise ValueError(
            "Winding sum requires "
            f"{lattice_point_count:,} lattice points, exceeding "
            f"max_lattice_points={max_lattice_points:,}. Increase the bound "
            "explicitly after assessing the computational cost."
        )
    return (
        radius,
        lattice_cutoff,
        chunk_size,
        max_lattice_points,
        lattice_point_count,
    )


def _truncated_winding_sum(
    reduced_form: FloatArray,
    radius: float,
    lattice_cutoff: int,
    chunk_size: int,
) -> float:
    r"""Evaluate the truncated compact boson winding sum

    .. math::

       \Theta_N(R,T)
       =
       \sum_{\boldsymbol{n}\in\{-N,\ldots,N\}^{2g}}
       \exp\!\left(
           -4\pi R^2\boldsymbol{n}^{T}T\boldsymbol{n}
       \right).

    Here :math:`N` is ``lattice_cutoff``, :math:`R` is ``radius``, and
    :math:`T` is ``reduced_form``.

    Parameters
    ----------
    reduced_form : numpy.ndarray
        Real ``float64`` symmetric positive-definite winding form of shape
        ``(2g, 2g)``.
    radius : float
        Positive compactification radius.
    lattice_cutoff : int
        Nonnegative component-wise cutoff for winding vectors.
    chunk_size : int
        Positive number of winding vectors evaluated simultaneously.

    Returns
    -------
    float
        Positive truncated winding sum.
    """
    dimension = reduced_form.shape[0]
    if dimension == 0:
        return 1.0
    _cholesky_factor_and_log_det(reduced_form, symmetrize=True)

    coordinates = range(-lattice_cutoff, lattice_cutoff + 1)
    vectors = product(coordinates, repeat=dimension)
    total = 0.0
    while True:
        chunk = list(islice(vectors, chunk_size))
        if not chunk:
            break
        points = np.asarray(chunk, dtype=np.float64)
        quadratic = np.einsum(
            "ni,ij,nj->n",
            points,
            reduced_form,
            points,
            optimize=True,
        )
        total += float(
            np.sum(np.exp(-4.0 * math.pi * radius**2 * quadratic))
        )
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError(f"Invalid winding sum {total}.")
    return total


def identity_wavefunctional_kernel(boundary_length: int) -> FloatArray:
    r"""Return the quadratic kernel of the identity wavefunctional of the free boson.

    Define the boundary Fourier modes as

    .. math::

       X_m
       =
       \frac{1}{2\pi}
       \int_0^{2\pi}
       X(\sigma)e^{-im\sigma}\,d\sigma,

    In terms of the boundary Fourier modes, the identity wavefunctional is

    .. math::

       \Psi_1[X(\sigma)]
       \propto
       \exp\!\left[
           -\frac{1}{\alpha'}
           \sum_{m\geq 1}mX_mX_{-m}
       \right].

    On a boundary discretized into ``L`` sites, this becomes

    .. math::

       \Psi_{1,L}[X]
       \propto
       \exp\!\left[
           -\frac{1}{2\pi\alpha'}
           \sum_{k,\ell=1}^{L}
           X_k K_{(k-\ell)\bmod L}X_\ell
       \right],

    where the kernel is defined as

    .. math::

       K_d
       =
       \frac{\sin(\pi/L)}
            {L\left[
                \cos(2\pi d/L)-\cos(\pi/L)
            \right]},
       \qquad
       d\in\mathbb{Z}/L\mathbb{Z}.

    This function returns this kernel.

    Parameters
    ----------
    boundary_length : int
        Number of discretized sites on the full disc boundary.

    Returns
    -------
    numpy.ndarray
        Real ``float64`` kernel vector of shape ``(boundary_length,)``.
    """
    if isinstance(boundary_length, bool) or not isinstance(boundary_length, Integral):
        raise TypeError("boundary_length must be an integer, not a Boolean.")
    boundary_length = int(boundary_length)
    if boundary_length <= 2:
        raise ValueError(
            f"boundary_length must be greater than two, got {boundary_length}."
        )

    angle = math.pi / boundary_length
    theta = (
        2.0 * math.pi / boundary_length
    ) * np.arange(boundary_length, dtype=np.float64)
    denominator = (
        2.0
        * boundary_length
        * np.sin(0.5 * (theta + angle))
        * np.sin(0.5 * (angle - theta))
    )
    kernel = np.sin(angle) / denominator
    kernel[1:] = 0.5 * (kernel[1:] + kernel[:0:-1])
    return np.asarray(kernel, dtype=np.float64)


def _one_face_gluing_data(
    ribbon_graph,
    edge_lengths: Sequence[int],
    ) -> _OneFaceGluingData:
    r"""First uses :func:`~string_amplitudes.ribbon_graph_generator.get_boundary_data` to convert a ribbon graph
    structure into the boundary data of a disc. The boundary is discretized by
    first dividing each edge :math:`e_i` into ``edge_lengths[i]``, or
    :math:`\{\ell_i\}`, number of sites (the exact number can vary between
    each site, but sewn together edges must have the same number of sites).
    The function then constructs the explicit sewing map between discretized
    sites. The sites on paired occurrences of an edge are matched in reverse
    boundary order, as required by the orientation-reversing sewing map.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected one-face ribbon graph ``(edges, vertices, rotation)``.
    edge_lengths : sequence of int
        Positive integer number of discretized sites assigned to each edge.

    Returns
    -------
    _OneFaceGluingData
        Sewing data associated with the discretized ribbon graph.
    """
    boundary_data = get_boundary_data(ribbon_graph, edge_lengths)
    edges = ribbon_graph[0]
    n_edges = len(edges)
    if boundary_data["n_faces"] != 1:
        raise NotImplementedError(
            "Partition-function sewing currently requires a one-face ribbon "
            f"graph; received {boundary_data['n_faces']} faces."
        )
    lengths = np.asarray(boundary_data["edge_lengths"], dtype=np.int64)
    boundary = boundary_data["boundaries"][0]
    sewing = boundary_data["sewing"]
    genus = int(boundary_data["genus"])
    boundary_length = int(boundary_data["boundary_lengths"][0])
    segment_starts = np.asarray(
        (*boundary_data["boundary_segment_starts"][0], boundary_length),
        dtype=np.int64,
    )

    prime = np.full(boundary_length, -1, dtype=np.int64)
    edge_points = []
    oriented_edges = []
    for edge_index in range(n_edges):
        first_segment, second_segment = sorted(
            position for _, position in sewing[edge_index]
        )
        edge_length = int(lengths[edge_index])
        local = np.arange(edge_length, dtype=np.int64)
        first_sites = segment_starts[first_segment] + local
        second_sites = segment_starts[second_segment + 1] - 1 - local
        prime[first_sites] = second_sites
        prime[second_sites] = first_sites
        edge_points.append(first_sites)

        start, end, _ = boundary[first_segment]
        oriented_edges.append((start, end))

    if np.any(prime < 0) or not np.array_equal(
        prime[prime], np.arange(boundary_length)
    ):
        raise ValueError("Failed to construct an involutive boundary sewing map.")

    independent_sites = np.sort(np.concatenate(edge_points))
    if 2 * independent_sites.size != boundary_length:
        raise ValueError(
            "The independent-site count is inconsistent with pairwise sewing."
        )
    independent_lookup = np.full(boundary_length, -1, dtype=np.int64)
    independent_lookup[independent_sites] = np.arange(independent_sites.size)
    edge_columns = tuple(independent_lookup[sites] for sites in edge_points)
    if any(np.any(columns < 0) for columns in edge_columns):
        raise ValueError("An edge representative is missing from the independent sites.")

    return _OneFaceGluingData(
        genus=genus,
        edge_lengths=lengths,
        boundary_length=boundary_length,
        independent_sites=independent_sites,
        partner_sites=prime[independent_sites],
        prime=prime,
        edge_points=tuple(edge_points),
        edge_independent_columns=edge_columns,
        oriented_edges=tuple(oriented_edges),
    )


def _kernel_block(kernel: FloatArray, rows, columns) -> FloatArray:
    r"""For a given set of :math:`a` rows and :math:`b` columns, construct the
    full :math:`a\times b` kernel matrix from a compressed representation of the
    kernel (the more compressed representation is due to the kernel only
    depending on the relative difference between a row and column index).

    Parameters
    ----------
    kernel : numpy.ndarray
        Real ``float64`` circulant-kernel vector of shape ``(L,)``.
    rows : array-like of int
        Boundary-site indices selecting matrix rows.
    columns : array-like of int
        Boundary-site indices selecting matrix columns.

    Returns
    -------
    numpy.ndarray
        Real ``float64`` matrix of shape ``(len(rows), len(columns))``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    return kernel[(rows[:, None] - columns[None, :]) % kernel.size]


def _sewn_kernel_block(
    kernel: FloatArray,
    row_sites,
    row_partners,
    column_sites,
    column_partners,
) -> FloatArray:
    r"""For a given set of :math:`a` rows and :math:`b` columns, construct the
    full :math:`a\times b` kernel matrix using :func:`_kernel_block` and
    add together the matrices of sewn together sites.

    Parameters
    ----------
    kernel : numpy.ndarray
        Real ``float64`` circulant-kernel vector of shape ``(L,)``.
    row_sites : array-like of int
        Independent boundary-site indices selecting rows.
    row_partners : array-like of int
        Sewn partners corresponding elementwise to ``row_sites``.
    column_sites : array-like of int
        Independent boundary-site indices selecting columns.
    column_partners : array-like of int
        Sewn partners corresponding elementwise to ``column_sites``.

    Returns
    -------
    numpy.ndarray
        Real ``float64`` sewn-kernel matrix of shape
        ``(len(row_sites), len(column_sites))``.
    """
    return (
        _kernel_block(kernel, row_sites, column_sites)
        + _kernel_block(kernel, row_sites, column_partners)
        + _kernel_block(kernel, row_partners, column_sites)
        + _kernel_block(kernel, row_partners, column_partners)
    )


def _reduced_matter_matrix(
    gluing: _OneFaceGluingData,
    kernel: FloatArray,
    *,
    eliminated_index: int,
    block_size: int,
) -> FloatArray:
    r"""Construct the full kernel of a wavefunctional in the sewn disc representation.
    Use :func:`_sewn_kernel_block` to add the contributions from sewn sites, then eliminate
    the constant zero mode. One reduces :math:`x_i` to :math:`y_i` via
    :math:`x_N=-\sum_{i=1}^{N-1}y_i`. This is represented as the matrix
    :math:`\vec{x}=C\vec{y}`.
    Then, the kernel becomes :math:`A'=C^TKC`.

    Parameters
    ----------
    gluing : _OneFaceGluingData
        Validated one-face boundary sewing data.
    kernel : numpy.ndarray
        Real ``float64`` circulant-kernel vector for the full boundary.
    eliminated_index : int
        Index, among the independent sites, used to eliminate the constant mode.
    block_size : int
        Positive number of reduced-matrix rows constructed simultaneously.

    Returns
    -------
    numpy.ndarray
        Real ``float64`` symmetric reduced matter matrix of shape
        ``(N - 1, N - 1)``.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise TypeError("block_size must be an integer, not a Boolean.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    sites = gluing.independent_sites
    partners = gluing.partner_sites
    n_sites = sites.size
    if n_sites < 2:
        raise ValueError("At least two independent boundary sites are required.")

    eliminated_index = int(eliminated_index) % n_sites
    active_indices = np.delete(np.arange(n_sites), eliminated_index)
    active_sites = sites[active_indices]
    active_partners = partners[active_indices]
    star_site = sites[eliminated_index : eliminated_index + 1]
    star_partner = partners[eliminated_index : eliminated_index + 1]

    active_star = _sewn_kernel_block(
        kernel,
        active_sites,
        active_partners,
        star_site,
        star_partner,
    )[:, 0]
    star_active = _sewn_kernel_block(
        kernel,
        star_site,
        star_partner,
        active_sites,
        active_partners,
    )[0]
    star_star = float(
        _sewn_kernel_block(
            kernel,
            star_site,
            star_partner,
            star_site,
            star_partner,
        )[0, 0]
    )

    size = active_sites.size
    reduced = np.empty((size, size), dtype=np.float64, order="F")
    for lower in range(0, size, block_size):
        upper = min(size, lower + block_size)
        sewn = _sewn_kernel_block(
            kernel,
            active_sites[lower:upper],
            active_partners[lower:upper],
            active_sites,
            active_partners,
        )
        reduced[lower:upper] = (
            sewn
            - active_star[lower:upper, None]
            - star_active[None, :]
            + star_star
        )
    return np.asfortranarray(0.5 * (reduced + reduced.T))


def identity_wavefunctional_kernel_reduced(
    ribbon_graph,
    edge_lengths: Sequence[int],
    *,
    block_size: int = 1024,
) -> FloatArray:
    r"""Construct the identity wavefunctional (with zero mode removed)
       :math:`A'` on the boundary of the disc of the one face genus :math:`g`
       ribbon graph, in the case of a free boson.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected one-face ribbon graph ``(edges, vertices, rotation)``.
    edge_lengths : sequence of int
        Positive integer length assigned to every graph edge.
    block_size : int, optional
        Positive number of matrix rows constructed simultaneously.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix of shape ``(sum(edge_lengths)-1,)`` squared.
    """
    gluing = _one_face_gluing_data(ribbon_graph, edge_lengths)
    kernel = identity_wavefunctional_kernel(gluing.boundary_length)
    return _reduced_matter_matrix(
        gluing,
        kernel,
        eliminated_index=-1,
        block_size=block_size,
    )


def matter_log_determinant(
    ribbon_graph,
    edge_lengths: Sequence[int],
    *,
    block_size: int = 1024,
) -> float:
    r"""Given a discretized genus :math:`g` ribbon graph with one face, construct the
       logarithm of the determinant of the quadratic kernel of the identity
       wavefunctional :math:`\det A'` using :func:`identity_wavefunctional_kernel_reduced`
       and :func:`_cholesky_factor_and_log_det`.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected one-face ribbon graph ``(edges, vertices, rotation)``.
    edge_lengths : sequence of int
        Positive integer length assigned to every graph edge.
    block_size : int, optional
        Positive number of matrix rows constructed simultaneously.

    Returns
    -------
    float
        Natural logarithm of the positive determinant of the reduced matter
        matrix.
    """
    matrix = identity_wavefunctional_kernel_reduced(
        ribbon_graph,
        edge_lengths,
        block_size=block_size,
    )
    _, logdet = _cholesky_factor_and_log_det(matrix, symmetrize=True)
    return logdet


def _fundamental_cycle_basis(oriented_edges) -> IntArray:
    r"""Given a set of oriented edges (an oriented edge is a collection of
    ordered vertices), construct a matrix :math:`B\in \mathbb{Z}^{E,2g}`,
    with :math:`E` the number of edges, that converts arbitrary integers into
    an allowed set of winding integers across each edge. For a vector
    :math:`\vec{v}\in\mathbb{Z}^{2g}` with genus g, B outputs the winding
    integers :math:`\vec{s}` across each edge. The matrix ensures that the
    winding integers of edges incident at a vertex add up to zero. The
    construction first chooses a spanning tree, i.e. a set of edges that
    contains and connects every vertex, but contains no closed loops.
    A spanning tree of :math:`V` vertices contains :math:`V-1` edges. Assign a column of
    :math:`B\in \mathbb{Z}^{E,2g}` for each edge not in the spanning tree. For
    each edge :math:`e_i` not in the spanning tree, there exists a unique
    oriented closed cycle in the spanning tree that connects the endpoints of
    :math:`e_i`. For each edge in the closed cycle connecting the endpoints of
    :math:`e_i`, assign +1 to :math:`B_{ai}` if the cycle associated to
    :math:`e_a` traversess :math:`e_i` along its orientation, -1 if against its
    orientation, and 0 if the cycle associated to :math:`e_a` does not contain
    :math:`e_i`.

    Parameters
    ----------
    oriented_edges : sequence of tuple of (int, int)
        Oriented endpoint pair for each graph edge.

    Returns
    -------
    numpy.ndarray
        Integer ``int64`` cycle-basis matrix of shape
        ``(n_edges, n_edges - n_vertices + 1)``.
    """
    n_edges = len(oriented_edges)
    vertices = sorted({vertex for edge in oriented_edges for vertex in edge})
    parent = {vertex: vertex for vertex in vertices}

    def find(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    tree_edges = []
    non_tree_edges = []
    for edge_index, (start, end) in enumerate(oriented_edges):
        root_start = find(start)
        root_end = find(end)
        if root_start != root_end:
            parent[root_end] = root_start
            tree_edges.append(edge_index)
        else:
            non_tree_edges.append(edge_index)

    tree_adjacency = {vertex: [] for vertex in vertices}
    for edge_index in tree_edges:
        start, end = oriented_edges[edge_index]
        tree_adjacency[start].append((end, edge_index))
        tree_adjacency[end].append((start, edge_index))

    def tree_path(start, end):
        previous = {start: (None, None)}
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            if vertex == end:
                break
            for neighbor, edge_index in tree_adjacency[vertex]:
                if neighbor not in previous:
                    previous[neighbor] = (vertex, edge_index)
                    queue.append(neighbor)
        if end not in previous:
            raise ValueError(f"No tree path connects vertices {start} and {end}.")
        path = []
        current = end
        while current != start:
            previous_vertex, edge_index = previous[current]
            path.append((previous_vertex, current, edge_index))
            current = previous_vertex
        path.reverse()
        return path

    cycles = []
    for edge_index in non_tree_edges:
        start, end = oriented_edges[edge_index]
        cycle = np.zeros(n_edges, dtype=np.int64)
        cycle[edge_index] = 1
        for path_start, path_end, tree_edge in tree_path(end, start):
            source, target = oriented_edges[tree_edge]
            cycle[tree_edge] = (
                1 if (path_start, path_end) == (source, target) else -1
            )
        cycles.append(cycle)
    if not cycles:
        return np.zeros((n_edges, 0), dtype=np.int64)
    return np.column_stack(cycles)


def _compact_boson_reduced_form(
    ribbon_graph,
    edge_lengths: Sequence[int],
    *,
    block_size: int = 1024,
) -> tuple[float, FloatArray]:
    r"""Construct the components of the compact boson partition function that are
    used when summing over winding. First, :math:`\log \det A'`, the logarithm of the
    of free boson partition function quadratic form, is constructed.
    The compact boson partition function can be expressed as

    .. math::

       Z_{R,N}^{\mathrm{code}}
       =
       \frac{(2\pi)^{1-g}R}{\sqrt{\det A'}}
       \sum_{\boldsymbol{n}\in\{-N,\ldots,N\}^{2g}}
       \exp\!\left(
           -4\pi R^2
           \boldsymbol{n}^{T}T\boldsymbol{n}
       \right).

    In this function, :math:`T` is constructed as well as :math:`\log\det A'`.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected one-face ribbon graph.
    edge_lengths : sequence of int
        Positive integer length assigned to every graph edge.
    block_size : int, optional
        Positive number of matter-matrix rows constructed simultaneously.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        Natural logarithm of the compact matter determinant and the positive
        ``2g`` by ``2g`` reduced holonomy form.
    """
    gluing = _one_face_gluing_data(ribbon_graph, edge_lengths)
    # With alpha'=1, the compact derivation uses M=K/(2 alpha')=K/2.
    kernel = 0.5 * identity_wavefunctional_kernel(gluing.boundary_length)
    matter = _reduced_matter_matrix(
        gluing,
        kernel,
        eliminated_index=0,
        block_size=block_size,
    )
    independent = gluing.independent_sites
    prime = gluing.prime
    eliminated = independent[0]
    eliminated_partner = prime[eliminated]
    active = independent[1:]
    all_partners = prime[independent]

    matter_factor, logdet_matter = _cholesky_factor_and_log_det(
        matter,
        symmetrize=True,
    )
    mixed = (
        _kernel_block(kernel, active, all_partners)
        + _kernel_block(kernel, prime[active], all_partners)
        - _kernel_block(kernel, [eliminated], all_partners)
        - _kernel_block(kernel, [eliminated_partner], all_partners)
    )

    n_edges = len(gluing.edge_points)
    first_term = np.empty((n_edges, n_edges), dtype=np.float64)
    primed_edge_points = [prime[points] for points in gluing.edge_points]
    for row_edge, rows in enumerate(primed_edge_points):
        for column_edge, columns in enumerate(primed_edge_points):
            first_term[row_edge, column_edge] = float(
                np.sum(_kernel_block(kernel, rows, columns))
            )

    incidence = np.zeros((active.size, n_edges), dtype=np.float64)
    for edge_index, columns in enumerate(gluing.edge_independent_columns):
        incidence[:, edge_index] = np.sum(mixed[:, columns], axis=1)
    edge_form = first_term - incidence.T @ _cholesky_solve(
        matter_factor,
        incidence,
    )
    edge_form = 0.5 * (edge_form + edge_form.T)

    cycle_basis = _fundamental_cycle_basis(gluing.oriented_edges)
    expected_cycles = 2 * gluing.genus
    if cycle_basis.shape != (n_edges, expected_cycles):
        raise ValueError(
            "Cycle-basis dimension is inconsistent with the ribbon-graph genus: "
            f"got {cycle_basis.shape[1]}, expected {expected_cycles}."
        )
    reduced_form = cycle_basis.T @ edge_form @ cycle_basis
    reduced_form = 0.5 * (reduced_form + reduced_form.T)
    if expected_cycles:
        _cholesky_factor_and_log_det(reduced_form, symmetrize=True)

    return logdet_matter, reduced_form


def compact_boson_partition_function(
    ribbon_graph,
    edge_lengths: Sequence[int],
    radius: float,
    *,
    lattice_cutoff: int = 6,
    chunk_size: int = 100_000,
    block_size: int = 1024,
    max_lattice_points: int | None = DEFAULT_MAX_LATTICE_POINTS,
) -> float:
    r"""Given a discretized ribbon graph, return the
    compact boson partition function.

    .. math::

       \log Z_R=(1-g)\log(2\pi)+\log R-\frac12\log\det A'
       +\log\!\sum_n e^{-4\pi R^2 n^T T n}.

    The matrix :math:`T` is constructed as

    .. math::

       T=B^T[Q^T KQ-Q^TKPC_0(A')^{-1}C_0^T P^T KQ]B.

    :math:`B`, generated using :func:`_fundamental_cycle_basis`, converts arbitrary
    integers :math:`\vec{n}\in \mathbb{Z}^{2g}`, to allowed winding integers
    while :math:`K` is the quadratic kernel of the discretized identity wavefunctional
    of the noncompact free boson.
    Consider matrices :math:`C_0, P,Q`. These matrices parameterize the reduction
    of the full unconstrained field :math:`X[i]` on the disc to the constrained
    field :math:`x[i]`. The field :math:`x[i]` is constrained because we are not constructing
    the unconstrained wavefunctional on the disc, but rather the partition function on
    the ribbon graph. The precise relation is :math:`\vec{X}=PC_0 \vec{x}+2\pi RQ \vec{s}`.
    :math:`C_0` removes the constant mode, i.e. constraints :math:`X[l]=-\sum_{i\neq L}x_i`.
    :math:`P` constrains :math:`X[i]=X[s(i)]`, where :math:`s(i)` is the sewing map inherited
    from the ribbon graph. :math:`Q` implements the winding shifts to only entry of each pair
    of sewed edges. When the integration over the boundary variables :math:`X[i]` is performed,
    one recovers the above form of :math:`T`.

    Parameters
    ----------
    ribbon_graph : tuple
        genus g ribbon graph with one face.
    edge_lengths : sequence of int
        Number of discretized points for each edge.
    radius : float
        radius of compact boson.
    lattice_cutoff : int, optional
        Truncates every winding integer to be less than or equal to
        ``lattice_cutoff`` in absolute value.
    chunk_size : int, optional
        The winding sum is split up into components, each of ``chunk_size``, with each component a
        vectorized computation. This reduces the memory load. Defaults to 100,000.
    block_size : int, optional
        Positive number of rows of :math:`A'`, the free boson identity wavefunctional kernel,
        to construct in a vectorized computation. This reduces memory load. Defaults to 1,024.
    max_lattice_points : int or None, optional
        Work guard for the holonomy sum.  ``None`` explicitly disables it.

    Returns
    -------
    float
        Compact-boson partition function.  An exception is raised if its value
        is outside the representable range of a double-precision float.
    """
    try:
        edges, vertices, _ = ribbon_graph
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ribbon_graph must be an (edges, vertices, rotation) tuple."
        ) from error
    inferred_dimension = 1 - len(vertices) + len(edges)
    (
        radius,
        lattice_cutoff,
        chunk_size,
        max_lattice_points,
        _,
    ) = _validate_winding_sum_parameters(
        radius,
        lattice_cutoff,
        chunk_size,
        max_lattice_points,
        dimension=inferred_dimension,
    )

    logdet_A_prime, reduced_form = _compact_boson_reduced_form(
        ribbon_graph,
        edge_lengths,
        block_size=block_size,
    )
    if inferred_dimension != reduced_form.shape[0]:
        raise ValueError("Inferred holonomy dimension changed during validation.")
    genus = inferred_dimension // 2
    theta_sum = _truncated_winding_sum(
        reduced_form,
        radius,
        lattice_cutoff,
        chunk_size,
    )
    log_Z = (
        math.log(2.0 * math.pi * radius)
        - genus * math.log(2.0 * math.pi)
        - 0.5 * logdet_A_prime
        + math.log(theta_sum)
    )
    smallest_positive = np.nextafter(0.0, 1.0)
    minimum_log = math.log(float(smallest_positive))
    maximum_log = math.log(float(np.finfo(np.float64).max))
    if not minimum_log <= log_Z <= maximum_log:
        raise FloatingPointError(
            "The compact-boson partition function is outside the "
            "representable float64 range."
        )
    return float(math.exp(log_Z))


__all__ = (
    "compact_boson_partition_function",
    "matter_log_determinant",
    "identity_wavefunctional_kernel",
    "identity_wavefunctional_kernel_reduced",
)
