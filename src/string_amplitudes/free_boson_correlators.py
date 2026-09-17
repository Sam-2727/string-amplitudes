r"""Normalized free boson one point functions on Riemann surfaces.

The public objects are:

* :class:`FreeBosonOnePointFunctions`: Store the one point functions.
* :func:`free_boson_one_point_functions`: evaluate the correlators
.. math::

   \mathrm{holomorphic}
   =
   \left\langle
   :\!\partial_z X\,\partial_z X\!:(0)
   \right\rangle,

and

.. math::

   \mathrm{mixed}
   =
   \left\langle
   :\!\partial_z X\,\bar{\partial}_{\bar z}X\!:(0)
   \right\rangle.

* :func:`genus_one_period_matrix_one_point_functions`: Evaluate the corresponding flat
space period matrix expressions.
* :func:`genus_one_disk_frame_one_point_functions`: Transform the flat space genus one
expression to the ribbon graph coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .partition_function import (
    _cholesky_factor_and_log_det,
    _cholesky_solve,
    _one_face_gluing_data,
    _reduced_matter_matrix,
    identity_wavefunctional_kernel,
)
from .riemann_surface_holomorphic_data import theta_truncation
from .ribbon_graph_to_period_matrix import DiscHolomorphicOneForm, PeriodMapResult


@dataclass(frozen=True, slots=True)
class FreeBosonOnePointFunctions:
    r"""Class to store the results from the one point functions.

    Parameters
    ----------
    holomorphic : complex
        The normalized correlator
        :math:`\langle:\!\partial X\partial X\!:\rangle`.
    mixed : complex
        The normalized correlator
        :math:`\langle:\!\partial X\bar\partial X\!:\rangle`.
    """

    holomorphic: complex
    mixed: complex


def _reduced_boundary_mode(gluing, mode: int) -> np.ndarray:
    r"""Return the coefficients of the fourier modes of a boundary variable
    after the zero mode is removed.

    Parameters
    ----------
    gluing : _OneFaceGluingData
        Sewing data for the ribbon graph with one face containing the independent boundary
        sites, their identified partners, and the full boundary length.
    mode : int
        Integer Fourier-mode label.

    Returns
    -------
    numpy.ndarray
        Complex coefficient vector of shape ``(N - 1,)``, where ``N`` is the
        number of independent sewn boundary sites. Its contraction with the
        reduced boundary variables gives the Fourier coefficient ``X_mode``.
    """

    boundary_length = gluing.boundary_length
    angles = (2.0 * math.pi / boundary_length) * (
        np.arange(boundary_length, dtype=np.float64) + 1.0
    )
    phase = np.exp(-1j * mode * angles) / boundary_length
    sewn_mode = phase[gluing.independent_sites] + phase[gluing.partner_sites]
    return np.asarray(sewn_mode[:-1] - sewn_mode[-1], dtype=np.complex128)


def free_boson_one_point_functions(
    ribbon_graph,
    edge_lengths: Sequence[int],
    *,
    block_size: int = 1024,
) -> FreeBosonOnePointFunctions:
    r"""Evaluate normalized noncompact free boson one point functions.

    The operator is inserted at the origin of the unit disc coordinate.  The
    calculation sews the wavefunctionals

    .. math::

       \Psi_{:\!\partial X\partial X\!:}[X]
       =X_1^2\Psi_1[X],
       \qquad
       \Psi_{:\!\partial X\bar\partial X\!:}[X]
       =\frac{1}{\pi}\left(\frac12-X_1X_{-1}\right)\Psi_1[X],

    and divides by the identity path integral. Although the Weyl anomaly factor cancels between
    the numerator and denominator, the result still depends on the local coordinates where the operator
    is inserted.

    Parameters
    ----------
    ribbon_graph : tuple
        Connected one face ribbon graph ``(edges, vertices, rotation)``.
    edge_lengths : sequence of int
        Positive number of discretized sites assigned to every graph edge.
    block_size : int, optional
        Positive number of rows of the matrix used in the computation of the free 
        boson partition function to construct at once. More rows requires more memory.

    Returns
    -------
    FreeBosonOnePointFunctions
        The pair of one point functions in the unit 
        disc coordinate.
    """

    gluing = _one_face_gluing_data(ribbon_graph, edge_lengths)
    kernel = identity_wavefunctional_kernel(gluing.boundary_length)
    matter = _reduced_matter_matrix(
        gluing,
        kernel,
        eliminated_index=-1,
        block_size=block_size,
    )
    factor, _ = _cholesky_factor_and_log_det(matter, symmetrize=True)
    mode_one = _reduced_boundary_mode(gluing, 1)
    mode_minus_one = np.conjugate(mode_one)
    covariance_mode_one = _cholesky_solve(factor, mode_one)

    holomorphic = math.pi * np.dot(mode_one, covariance_mode_one)
    mixed = 1.0 / (2.0 * math.pi) - np.dot(
        mode_minus_one,
        covariance_mode_one,
    )
    return FreeBosonOnePointFunctions(
        holomorphic=complex(holomorphic),
        mixed=complex(mixed),
    )


def genus_one_period_matrix_one_point_functions(
    period_matrix,
    *,
    theta_lattice_cutoff: int | None = None,
    theta_tolerance: float = 1.0e-14,
) -> FreeBosonOnePointFunctions:
    r"""Evaluate the pair of one point functions in the flat torus coordinate.

    For :math:`u\sim u+1\sim u+\tau`, this function evaluates

    .. math::

       \left\langle:\!\partial_uX\bar\partial_{\bar u}X\!:\right\rangle
       =\frac{1}{2\tau_2},
       \qquad
       \left\langle:\!\partial_uX\partial_uX\!:\right\rangle
       =\frac{\theta_1'''(0\mid\tau)}{6\theta_1'(0\mid\tau)}
       +\frac{\pi}{2\tau_2}.

    The theta derivatives are evaluated from the truncated odd theta lattice
    series in the convention where its first argument has periods
    :math:`1` and :math:`\tau`.

    Parameters
    ----------
    period_matrix : complex or array-like of complex
        Genus one modulus or period matrix of shape ``(1, 1)``.
    theta_lattice_cutoff : int or None, optional
        Component-wise cutoff in the theta-derivative sums.  If ``None``, a
        cutoff is selected from ``theta_tolerance``.
    theta_tolerance : float, optional
        Positive target used to select the cutoff when
        ``theta_lattice_cutoff`` is ``None``.

    Returns
    -------
    FreeBosonOnePointFunctions
        Exact holomorphic and mixed correlators in the flat coordinate.
    """

    matrix = np.asarray(period_matrix, dtype=np.complex128)
    if matrix.shape == ():
        modulus = complex(matrix)
    elif matrix.shape == (1, 1):
        modulus = complex(matrix[0, 0])
    else:
        raise ValueError(
            "period_matrix must be a scalar or an array of shape (1, 1)."
        )
    if not math.isfinite(modulus.real) or not math.isfinite(modulus.imag):
        raise ValueError("period_matrix must be finite.")
    if modulus.imag <= 0.0:
        raise ValueError("the genus-one modulus must have positive imaginary part.")

    if theta_lattice_cutoff is None:
        if isinstance(theta_tolerance, (bool, np.bool_)) or not isinstance(
            theta_tolerance,
            Real,
        ):
            raise TypeError("theta_tolerance must be a real number.")
        theta_tolerance = float(theta_tolerance)
        if not 0.0 < theta_tolerance < 1.0:
            raise ValueError("theta_tolerance must lie strictly between zero and one.")
        theta_lattice_cutoff = theta_truncation(
            np.asarray([[modulus]], dtype=np.complex128),
            tolerance=theta_tolerance,
        )
    elif isinstance(theta_lattice_cutoff, (bool, np.bool_)) or not isinstance(
        theta_lattice_cutoff,
        Integral,
    ):
        raise TypeError("theta_lattice_cutoff must be an integer or None.")
    else:
        theta_lattice_cutoff = int(theta_lattice_cutoff)
        if theta_lattice_cutoff < 1:
            raise ValueError("theta_lattice_cutoff must be positive.")

    integers = np.arange(
        -theta_lattice_cutoff,
        theta_lattice_cutoff + 1,
        dtype=np.float64,
    )
    shifted = integers + 0.5
    odd_theta_terms = np.exp(
        1j * math.pi * (shifted * shifted * modulus + shifted)
    )
    first_derivative = np.sum((2j * math.pi * shifted) * odd_theta_terms)
    third_derivative = np.sum((2j * math.pi * shifted) ** 3 * odd_theta_terms)
    if abs(first_derivative) <= np.finfo(np.float64).tiny:
        raise FloatingPointError("the truncated theta derivative is numerically zero.")

    holomorphic = (
        third_derivative / (6.0 * first_derivative)
        + math.pi / (2.0 * modulus.imag)
    )
    mixed = 1.0 / (2.0 * modulus.imag)
    return FreeBosonOnePointFunctions(
        holomorphic=complex(holomorphic),
        mixed=complex(mixed),
    )


def _abelian_primitive_schwarzian(
    one_form: DiscHolomorphicOneForm,
    point: complex,
) -> complex:
    r"""Evaluate the Schwarzian derivative of the coordinate :math:`u(z)` obtained by
    integrating a holomoprhic one form.

    Parameters
    ----------
    one_form : DiscHolomorphicOneForm
        Holomorphic one-form :math:`\omega(z)=f(z)\,dz` whose local primitive
        satisfies :math:`u'(z)=f(z)`.
    point : complex
        Point in the unit-disc coordinate at which to evaluate the
        Schwarzian derivative.

    Returns
    -------
    complex
        Schwarzian derivative :math:`\{u,z\}` evaluated at ``point``.
    """

    point = complex(point)
    offsets = one_form.prevertices - point
    if np.any(np.isclose(offsets, 0.0, rtol=0.0, atol=1.0e-14)):
        raise ValueError("point coincides with a prevertex of the disc one-form.")

    coefficients = one_form.coefficients
    polynomial = np.polynomial.polynomial.polyval(point, coefficients)
    if abs(polynomial) <= np.finfo(np.float64).tiny:
        raise ValueError("the normalized one-form vanishes at point.")
    first_coefficients = np.arange(1, coefficients.size) * coefficients[1:]
    second_coefficients = (
        np.arange(2, coefficients.size)
        * np.arange(1, coefficients.size - 1)
        * coefficients[2:]
    )
    polynomial_first = (
        np.polynomial.polynomial.polyval(point, first_coefficients)
        if first_coefficients.size
        else 0.0j
    )
    polynomial_second = (
        np.polynomial.polynomial.polyval(point, second_coefficients)
        if second_coefficients.size
        else 0.0j
    )

    logarithmic_singular_first = one_form.singularity_power * np.sum(
        1.0 / offsets
    )
    singular_second_ratio = (
        logarithmic_singular_first**2
        + one_form.singularity_power * np.sum(1.0 / offsets**2)
    )
    polynomial_first_ratio = polynomial_first / polynomial
    first_ratio = logarithmic_singular_first + polynomial_first_ratio
    second_ratio = (
        singular_second_ratio
        + 2.0 * logarithmic_singular_first * polynomial_first_ratio
        + polynomial_second / polynomial
    )
    return complex(second_ratio - 1.5 * first_ratio**2)


def genus_one_disk_frame_one_point_functions(
    period_data: PeriodMapResult,
    *,
    point: complex = 0.0j,
    theta_lattice_cutoff: int | None = None,
    theta_tolerance: float = 1.0e-14,
) -> FreeBosonOnePointFunctions:
    r"""Transform exact genus-one one-point functions to the disc frame.

    Let :math:`u(z)=\int^z\omega` for the A-normalized holomorphic one-form.
    The returned values are

    .. math::

       \left\langle:\!\partial_zX\bar\partial_{\bar z}X\!:\right\rangle
       =|u'(z)|^2
       \left\langle:\!\partial_uX\bar\partial_{\bar u}X\!:\right\rangle,

    .. math::

       \left\langle:\!\partial_zX\partial_zX\!:\right\rangle
       =u'(z)^2
       \left\langle:\!\partial_uX\partial_uX\!:\right\rangle
       -\frac{1}{12}\{u,z\}.

    The returned value is the Schwarzian derivative

    .. math::

    \{u,z\}
    =
    \frac{u'''(z)}{u'(z)}
    -\frac{3}{2}
    \left(\frac{u''(z)}{u'(z)}\right)^2
    =
    \frac{f''(z)}{f(z)}
    -\frac{3}{2}
    \left(\frac{f'(z)}{f(z)}\right)^2,

    where :math:`f(z)` is the coefficient of the holomorphic one form in the disc
    coordinate system.   

    Parameters
    ----------
    period_data : PeriodMapResult
        Genus one period matrix and its A-normalized disc one form.
    point : complex, optional
        Point in the unit disc coordinate where the operators are inserted.
    theta_lattice_cutoff : int or None, optional
        Cutoff in the theta derivative sums.  If ``None``, a
        cutoff is selected from ``theta_tolerance``.
    theta_tolerance : float, optional
        Positive target used to select the cutoff when
        ``theta_lattice_cutoff`` is ``None``.

    Returns
    -------
    FreeBosonOnePointFunctions
        Exact holomorphic and mixed correlators in the unit-disc coordinate.
    """

    if not isinstance(period_data, PeriodMapResult):
        raise TypeError("period_data must be a PeriodMapResult.")
    if period_data.genus != 1:
        raise ValueError("period_data must describe a genus-one surface.")
    flat = genus_one_period_matrix_one_point_functions(
        period_data.period_matrix,
        theta_lattice_cutoff=theta_lattice_cutoff,
        theta_tolerance=theta_tolerance,
    )
    one_form = period_data.normalized_one_forms[0]
    derivative = complex(one_form(point))
    schwarzian = _abelian_primitive_schwarzian(one_form, point)
    return FreeBosonOnePointFunctions(
        holomorphic=derivative**2 * flat.holomorphic - schwarzian / 12.0,
        mixed=abs(derivative) ** 2 * flat.mixed,
    )


__all__ = (
    "FreeBosonOnePointFunctions",
    "free_boson_one_point_functions",
    "genus_one_disk_frame_one_point_functions",
    "genus_one_period_matrix_one_point_functions",
)
