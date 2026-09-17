r"""Evaluate the analytic bosonic string integrand at genus two and three in terms of
the Igusa cusp forms.
This allows us to confirm that our numerics are correct.

The public functions are:

* :func:`genus_two_period_matrix_integrand`: Evaluates the genus two string integrand
  in terms of the period matrix.
* :func:`genus_three_period_matrix_integrand`: Evaluates the genus three string integrand
  in terms of the period matrix.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .riemann_surface_holomorphic_data import (
    RiemannSurfaceData,
    abel_jacobi_map,
    holomorphic_one_form_antiderivatives,
    igusa_cusp_form_genus_three,
    igusa_cusp_form_genus_two,
)
from .ribbon_graph_generator import get_boundary_data
from .ribbon_graph_to_period_matrix import compute_period_map


ComplexArray = NDArray[np.complex128]


def _canonical_divisor_zeros_genus_two(one_form) -> ComplexArray:
    r"""Finds the zeros of a genus two holomorphic one form

    Parameters
    ----------
    one_form : callable
        Genus-two disc holomorphic one form with a callable dimensional
        ``coefficients`` array that gives the coefficients which can be used
        to reconstruct the holomoprhic one form.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(2,)`` containing the two interior zeros,
        with a repeated entry when the polynomial has a double zero.

    Raises
    ------
    TypeError
        If ``one_form`` does not expose its polynomial coefficients.
    ValueError
        If the polynomial is numerically zero or does not have exactly two
        roots inside the unit disc.
    """

    coefficients = getattr(one_form, "coefficients", None)
    if coefficients is None:
        raise TypeError(
            "the normalized one-form must expose polynomial coefficients"
        )
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
    if nonzero.size == 0:
        raise ValueError("the normalized one-form polynomial is numerically zero")
    roots = np.roots(coefficients[: int(nonzero[-1]) + 1][::-1])
    zeros = np.asarray(roots[np.abs(roots) < 0.99], dtype=np.complex128)
    if zeros.size != 2:
        raise ValueError(
            "a genus-two holomorphic one-form must have two interior zeros; "
            f"found {zeros.size}"
        )
    return zeros


def _track_disc_one_form_zero(one_form, seed: complex) -> np.complex128:
    r"""Takes a given zero of a holomorphic one form, and when one of the ribbon graph
    lengths is changed infinitesimally, find the position of the zero of the infinitesimally
    changed holomorphic one form.

    Newton iteration starts at ``seed`` and follows the nearby root of the
    polynomial factor of ``one_form``.  If the iteration does not produce an
    interior root with a sufficiently small residual, both canonical-divisor
    zeros are computed and the one nearest ``seed`` is returned.

    Parameters
    ----------
    one_form : callable
        Disc holomorphic one-form on the neighboring genus-two surface,
        exposing a one-dimensional ``coefficients`` array in
        increasing-power order.
    seed : complex
        Position of the selected zero on the unperturbed surface.

    Returns
    -------
    numpy.complex128
        Position of the locally continued zero on the neighboring surface.

    Raises
    ------
    TypeError
        If ``one_form`` does not expose its polynomial coefficients.
    ValueError
        If the polynomial is numerically zero or its genus-two canonical
        divisor cannot be identified.

    Notes
    -----
    The continuation assumes that the surface perturbation is small and that
    the selected zero remains simple.  Convergence of the resulting Jacobian
    must still be checked.
    """

    coefficients = getattr(one_form, "coefficients", None)
    if coefficients is None:
        raise TypeError(
            "the normalized one-form must expose polynomial coefficients"
        )
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
    if nonzero.size == 0:
        raise ValueError("the normalized one-form polynomial is numerically zero")
    coefficients = coefficients[: int(nonzero[-1]) + 1]
    derivative = np.arange(1, coefficients.size) * coefficients[1:]
    zero = np.complex128(seed)
    for _ in range(30):
        value = np.polynomial.polynomial.polyval(zero, coefficients)
        slope = np.polynomial.polynomial.polyval(zero, derivative)
        if abs(slope) <= np.finfo(float).tiny:
            break
        step = value / slope
        zero = np.complex128(zero - step)
        if abs(step) <= 1e-13 * max(1.0, abs(zero)):
            break

    residual = abs(np.polynomial.polynomial.polyval(zero, coefficients))
    scale = max(1.0, float(np.linalg.norm(coefficients, ord=1)))
    if abs(zero) < 0.99 and residual <= 1e-8 * scale:
        return zero

    zeros = _canonical_divisor_zeros_genus_two(one_form)
    return np.complex128(zeros[np.argmin(np.abs(zeros - seed))])


def genus_two_period_matrix_integrand(
    ribbon_graph,
    edge_lengths: Sequence[int],
    *,
    c_point: complex = 0.0j,
    dependent_edge: int | None = None,
    period_quadrature_order: int = 128,
    antiderivative_quadrature_order: int = 96,
    theta_lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
    jacobian_step: int = 1,
    richardson_step: int | None = 2,
) -> float:
    r"""Evaluate the genus two critical bosonic string integrand.

    One then multiplies by the Jacobian of ribbon graph parameters in terms of the period
    matrix to evaluate the critical bosonic string integrand in terms of the ribbon graph
    coordinates. The final result includes the normalization.

    With :math:`u=\int_q^{z_c}\widehat\omega_1`, where :math:`q` is a zero of
    :math:`\widehat\omega_1`, it is

    .. math::

       2^{28}(2\pi)^{-50}
       \left|\det_{\mathbb R}
       \frac{\partial(\Omega_{11},\Omega_{12},\Omega_{22},u)}
       {\partial(\ell_1,\ldots,\ell_8)}\right|
       \frac{1}{
       (\det\operatorname{Im}\Omega)^{13}
       |\widehat\omega_1(z_c)|^2
       |\chi_{10}(\Omega)|^2}.

    Here :math:`\chi_{10}` uses convention implemented
    by :func:`~string_amplitudes.riemann_surface_holomorphic_data.igusa_cusp_form_genus_two`.
    The factor :math:`2^4` converting the complex coordinate Jacobian to the
    real coordinate Jacobian is included.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected trivalent genus two ribbon graph with one face.
    edge_lengths : sequence of int
        Nine positive discretized edge lengths.
    c_point : complex, optional
        Disc coordinate of the puncture insertion.
    dependent_edge : int or None, optional
        Edge adjusted to keep the total boundary length fixed. The last edge
        is used when this is ``None``.
    period_quadrature_order : int, optional
        Number of Gauss-Legendre nodes used for period integrals.
    antiderivative_quadrature_order : int, optional
        Number of Gauss-Legendre nodes used for Abel-Jacobi integrals.
    theta_lattice_cutoff : int or None, optional
        Cutoff used in each theta constant defining :math:`\chi_{10}`.
    tolerance : float, optional
        Theta-sum tolerance used when ``theta_lattice_cutoff`` is ``None``.
    jacobian_step : int, optional
        Positive edge-length step in the central finite difference.
    richardson_step : int or None, optional
        Second positive step used for Richardson extrapolation. If ``None``,
        only ``jacobian_step`` is used.

    Returns
    -------
    float
        Positive period-matrix density in the independent edge-length
        coordinates.
    """

    boundary_data = get_boundary_data(ribbon_graph, edge_lengths)
    if boundary_data["n_faces"] != 1 or boundary_data["genus"] != 2:
        raise ValueError(
            "genus_two_period_matrix_integrand requires a genus-two, one-face graph"
        )
    lengths = np.asarray(boundary_data["edge_lengths"], dtype=np.int64)
    if lengths.shape != (9,):
        raise ValueError("a trivalent genus-two, one-face graph must have nine edges")

    if dependent_edge is None:
        dependent_edge = lengths.size - 1
    if isinstance(dependent_edge, bool) or not isinstance(dependent_edge, Integral):
        raise TypeError("dependent_edge must be an integer or None")
    dependent_edge = int(dependent_edge)
    if not 0 <= dependent_edge < lengths.size:
        raise ValueError("dependent_edge is outside the edge index range")
    independent_edges = tuple(
        edge for edge in range(lengths.size) if edge != dependent_edge
    )

    def validate_step(value: int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer or None")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value >= min(lengths):
            raise ValueError(f"{name} must be smaller than every edge length")
        return value

    first_step = validate_step(jacobian_step, "jacobian_step")
    second_step = validate_step(richardson_step, "richardson_step")
    assert first_step is not None
    if second_step == first_step:
        raise ValueError("the two Jacobian steps must be distinct")

    def surface_at(surface_lengths: Sequence[int]) -> RiemannSurfaceData:
        period_data = compute_period_map(
            ribbon_graph,
            surface_lengths,
            period_quadrature_order=period_quadrature_order,
        )
        return RiemannSurfaceData(
            genus=2,
            Omega=period_data.period_matrix,
            normalized_one_forms=period_data.normalized_one_forms,
            antiderivatives_normalized_forms=holomorphic_one_form_antiderivatives(
                period_data.normalized_one_forms,
                quadrature_order=antiderivative_quadrature_order,
            ),
        )

    surface = surface_at(lengths)
    zeros = _canonical_divisor_zeros_genus_two(surface.normalized_one_forms[0])
    ordering = np.lexsort((zeros.imag, zeros.real))
    zero_seed = np.complex128(zeros[ordering[0]])
    c_point = complex(np.complex128(c_point))

    def period_puncture_coordinates(
        perturbed_surface: RiemannSurfaceData,
    ) -> NDArray[np.float64]:
        zero = _track_disc_one_form_zero(
            perturbed_surface.normalized_one_forms[0],
            zero_seed,
        )
        puncture_coordinate = abel_jacobi_map(
            c_point,
            perturbed_surface,
            basepoint=zero,
        )[0]
        coordinates = (
            perturbed_surface.Omega[0, 0],
            perturbed_surface.Omega[0, 1],
            perturbed_surface.Omega[1, 1],
            puncture_coordinate,
        )
        return np.asarray(
            [value.real for value in coordinates]
            + [value.imag for value in coordinates],
            dtype=np.float64,
        )

    def derivative_matrix(step: int) -> NDArray[np.float64]:
        derivative = np.empty((8, 8), dtype=np.float64)
        for column, edge in enumerate(independent_edges):
            plus = lengths.copy()
            minus = lengths.copy()
            plus[edge] += step
            plus[dependent_edge] -= step
            minus[edge] -= step
            minus[dependent_edge] += step
            plus_surface = surface_at(plus)
            minus_surface = surface_at(minus)
            derivative[:, column] = (
                period_puncture_coordinates(plus_surface)
                - period_puncture_coordinates(minus_surface)
            ) / (2.0 * step)
        return derivative

    derivative = derivative_matrix(first_step)
    if second_step is not None:
        coarse_derivative = derivative_matrix(second_step)
        ratio_squared = (second_step / first_step) ** 2
        derivative = (
            ratio_squared * derivative - coarse_derivative
        ) / (ratio_squared - 1.0)

    jacobian = abs(float(np.linalg.det(derivative)))
    det_im_omega = float(np.linalg.det(np.imag(surface.Omega)))
    if det_im_omega <= 0.0:
        raise ValueError(
            "the imaginary part of the period matrix is not positive definite"
        )
    puncture_one_form = surface.normalized_one_forms[0](c_point)
    if abs(puncture_one_form) <= np.finfo(float).tiny:
        raise ZeroDivisionError("the puncture lies at a zero of the first one-form")
    chi10 = igusa_cusp_form_genus_two(
        surface.Omega,
        lattice_cutoff=theta_lattice_cutoff,
        tolerance=tolerance,
    )
    if abs(chi10) <= np.finfo(float).tiny:
        raise ZeroDivisionError("the Igusa cusp form vanished")

    return float(
        2.0**28
        * (2.0 * math.pi) ** -50
        * jacobian
        * det_im_omega**-13
        * abs(puncture_one_form) ** -2
        * abs(chi10) ** -2
    )


def genus_three_period_matrix_integrand(
    period_matrix: Sequence[Sequence[complex]],
    *,
    theta_lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> float:
    r"""Evaluate the genmus three string integrand in terms of the period matrix.
    Unlike the genus two result, this not multiplied by the jacobian converting to
    the ribbon graph coordinates.

    We use the following convention for the Igusa cusp form:

    .. math::

       \chi_{18}(\Omega)
       =
       \prod_{\delta\,\mathrm{even}}
       \vartheta[\delta](0\mid\Omega).

    The function then treturns

    .. math::

       \frac{1}{
       (\det\operatorname{Im}\Omega)^{13}
       |\chi_{18}(\Omega)|}

    Parameters
    ----------
    period_matrix : sequence of sequence of complex
        Genus three period matrix in the Siegel upper half space.
    theta_lattice_cutoff : int or None, optional
        Cutoff used in each theta constant defining :math:`\chi_{18}`. If
        ``None``, :func:`~string_amplitudes.riemann_surface_holomorphic_data.theta_truncation`
        chooses the cutoff.
    tolerance : float, optional
        Theta sum tolerance used when ``theta_lattice_cutoff`` is ``None``.

    Returns
    -------
    float
        Genus three bosonic string integrand.
    """

    omega = np.asarray(period_matrix, dtype=np.complex128)
    chi18 = igusa_cusp_form_genus_three(
        omega,
        lattice_cutoff=theta_lattice_cutoff,
        tolerance=tolerance,
    )
    if abs(chi18) <= np.finfo(float).tiny:
        raise ZeroDivisionError(
            "the genus-three Igusa cusp form vanished; period coordinates "
            "are singular on the hyperelliptic locus"
        )
    det_im_omega = float(np.linalg.det(np.imag(omega)))
    if det_im_omega <= 0.0:
        raise ValueError(
            "the imaginary part of the period matrix is not positive definite"
        )
    return float(det_im_omega**-13 * abs(chi18) ** -1)


__all__ = (
    "genus_three_period_matrix_integrand",
    "genus_two_period_matrix_integrand",
)
