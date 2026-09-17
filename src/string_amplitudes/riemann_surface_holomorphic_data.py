r"""Evaluate holomorphic data on a compact Riemann surface from a basis of
:math:`\alpha`-cycle normalized holomorphic one-forms, their antiderivatives, the associated A and B cycles,
and the period matrix.

The bc ghost correlation function is also computed using the formula of Verlinde and Verlinde.

The public objects are:

* :class:`RiemannSurfaceData`: stores a period matrix, normalized
  one-forms, and their antiderivatives.
* :class:`BCGhostCorrelatorData`: stores the fixed data on a given Riemann surface
  that is used in the evaluation the bc ghost correlation function (i.e., data that doesn't
  depend on the point the correlation function is evaluated at).
* :func:`holomorphic_one_form_antiderivatives`: constructs antiderivatives of holomorphic one forms
  in a specified analytic coordinate chart.
* :func:`abel_jacobi_map`: evaluates the Abel-Jacobi
  map.
* :func:`theta_characteristics`: enumerates all genus-:math:`g`
  half-characteristics, optionally restricted to even or odd parity.
* :func:`characteristic_parity`: classifies a half-characteristic as even or
  odd.
* :func:`riemann_theta`: evaluates the truncated Riemann theta sum for a
  specified argument, period matrix, and half-characteristic.
* :func:`riemann_theta_gradient`: evaluates the gradient of the Riemann theta
  function with respect to its argument.
* :func:`igusa_cusp_form_genus_two` and
  :func:`igusa_cusp_form_genus_three`: evaluate the genus two and genu -three
  Igusa cusp forms in the conventions used in our paper.
* :func:`prime_form`: constructs the prime form from an odd
  characteristic.
* :func:`riemann_constant_vector`:  computes the Riemann constant.
* :func:`sigma_ratio`: constructs the sigma function appearing in the Verlinde Verlinde formula.
* :func:`prepare_bc_correlator`: prepares a reusable :math:`bc` correlator
  for fixed surface and normalization data.
* :func:`bc_correlator`: evaluates the bc holomorphic correlator with b-ghost weight :math:`\lambda`.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
HolomorphicForm = Callable[[complex], complex]
AbelianPrimitive = Callable[[complex], complex]
Characteristic = tuple[Sequence[float], Sequence[float]]


@dataclass(frozen=True, slots=True)
class RiemannSurfaceData:
    r"""A collection of the main data used in the computations on a given Riemann surface.

    Parameters
    ----------
    genus : int
        Positive genus of the Riemann surface.
    Omega : numpy.ndarray
        Period matrix of shape ``(genus, genus)`` with positive
        definite imaginary part.
    normalized_one_forms : tuple of callable
        :math:`\alpha`-cycle-normalized holomorphic one-forms satisfying
        :math:`\oint_{\alpha_J}\omega_I=\delta_{IJ}`.
    antiderivatives_normalized_forms : tuple of callable
        Antiderivatives of ``normalized_one_forms`` used as input to the
        Abel--Jacobi map:
        
        .. math::

           A_{z_0}(z)
           =
           \left(
           \int_{z_0}^{z}\omega_1,\ldots,
           \int_{z_0}^{z}\omega_g
           \right).

    """

    genus: int
    Omega: ComplexArray
    normalized_one_forms: tuple[HolomorphicForm, ...]
    antiderivatives_normalized_forms: tuple[AbelianPrimitive, ...]

    def __post_init__(self) -> None:
        if isinstance(self.genus, bool) or not isinstance(self.genus, Integral):
            raise TypeError("genus must be an integer")
        genus = int(self.genus)
        if genus < 1:
            raise ValueError("genus must be positive")

        omega = _validate_period_matrix(self.Omega)
        if omega.shape != (genus, genus):
            raise ValueError(
                f"Omega must have shape ({genus}, {genus}), got {omega.shape}"
            )
        forms = tuple(self.normalized_one_forms)
        primitives = tuple(self.antiderivatives_normalized_forms)
        if len(forms) != genus or len(primitives) != genus:
            raise ValueError(
                "normalized_one_forms and antiderivatives_normalized_forms "
                "must each contain "
                f"genus={genus} callables"
            )
        if not all(callable(value) for value in (*forms, *primitives)):
            raise TypeError(
                "all normalized forms and their antiderivatives must be callable"
            )

        object.__setattr__(self, "genus", genus)
        object.__setattr__(self, "Omega", omega)
        object.__setattr__(self, "normalized_one_forms", forms)
        object.__setattr__(self, "antiderivatives_normalized_forms", primitives)

    @property
    def tau(self) -> np.complex128 | None:
        """Return the scalar modulus for genus one, and ``None`` otherwise."""

        if self.genus == 1:
            return np.complex128(self.Omega[0, 0])
        return None


@dataclass(frozen=True, slots=True)
class BCGhostCorrelatorData:
    r"""Fixed geometric and normalization data for a :math:`bc` correlator.

    This object composes the intrinsic holomorphic surface data with the
    additional choices needed to evaluate the Verlinde--Verlinde formula.
    The auxiliary divisor points and sigma reference data are not invariants
    of the abstract Riemann surface.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Period matrix, normalized holomorphic one-forms, and their
        antiderivatives.
    riemann_constant : numpy.ndarray
        Riemann constant vector :math:`\Delta` of shape ``(genus,)`` in the
        Strebel sign convention.
    divisor_points : tuple of complex
        The :math:`g` auxiliary points entering the sigma-function ratio.
    normalization_point : complex
        Reference point at which ``sigma_normalization`` is specified.
    sigma_normalization : complex
        Nonzero value assigned to sigma at ``normalization_point``.
    chiral_z1 : complex
        Nonzero chiral determinant-line quantity :math:`Z_1`.
    """

    surface: RiemannSurfaceData
    riemann_constant: ComplexArray
    divisor_points: tuple[complex, ...]
    normalization_point: complex
    sigma_normalization: complex
    chiral_z1: complex

    def __post_init__(self) -> None:
        if not isinstance(self.surface, RiemannSurfaceData):
            raise TypeError("surface must be RiemannSurfaceData")

        riemann_constant = np.array(
            self.riemann_constant,
            dtype=np.complex128,
            copy=True,
        )
        if riemann_constant.shape != (self.surface.genus,):
            raise ValueError(
                "riemann_constant must have shape "
                f"({self.surface.genus},)"
            )
        if not np.all(np.isfinite(riemann_constant)):
            raise ValueError("riemann_constant contains a non-finite entry")
        riemann_constant.setflags(write=False)

        divisor_points = tuple(
            complex(np.complex128(point)) for point in self.divisor_points
        )
        if len(divisor_points) != self.surface.genus:
            raise ValueError(
                "divisor_points must contain "
                f"genus={self.surface.genus} points"
            )
        if not all(np.isfinite(point) for point in divisor_points):
            raise ValueError("divisor_points contains a non-finite entry")

        normalization_point = complex(np.complex128(self.normalization_point))
        if not np.isfinite(normalization_point):
            raise ValueError("normalization_point must be finite")
        if any(abs(point - normalization_point) < 1e-12 for point in divisor_points):
            raise ValueError(
                "divisor points must not coincide with normalization_point"
            )

        sigma_normalization = complex(np.complex128(self.sigma_normalization))
        if not np.isfinite(sigma_normalization):
            raise ValueError("sigma_normalization must be finite")
        if abs(sigma_normalization) <= np.finfo(float).tiny:
            raise ValueError("sigma_normalization must be nonzero")

        chiral_z1 = complex(np.complex128(self.chiral_z1))
        if not np.isfinite(chiral_z1):
            raise ValueError("chiral_z1 must be finite")
        if abs(chiral_z1) <= np.finfo(float).tiny:
            raise ValueError("chiral_z1 must be nonzero")

        object.__setattr__(self, "riemann_constant", riemann_constant)
        object.__setattr__(self, "divisor_points", divisor_points)
        object.__setattr__(self, "normalization_point", normalization_point)
        object.__setattr__(self, "sigma_normalization", sigma_normalization)
        object.__setattr__(self, "chiral_z1", chiral_z1)


def _validate_period_matrix(period_matrix: np.ndarray) -> ComplexArray:
    """Double check that the provided period matrix is finite, is a square matrix,
       and that the imaginary part is positive definite."""

    omega = np.array(period_matrix, dtype=np.complex128, copy=True)
    if omega.ndim != 2 or omega.shape[0] != omega.shape[1] or not omega.size:
        raise ValueError("the period matrix must be nonempty and square")
    if not np.all(np.isfinite(omega)):
        raise ValueError("the period matrix contains a non-finite entry")
    if not np.allclose(omega, omega.T, rtol=1e-11, atol=1e-12):
        raise ValueError("the period matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(np.imag(omega))
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("the imaginary part of the period matrix must be positive definite")
    omega.setflags(write=False)
    return omega


def holomorphic_one_form_antiderivatives(
    normalized_one_forms: Sequence[HolomorphicForm],
    *,
    basepoint: complex = 0.0j,
    quadrature_order: int = 128,
) -> tuple[AbelianPrimitive, ...]:
    r"""Construct antiderivatives of holomorpohic one forms by integrating along
    straight segments from a given basepoint (defaults to the origin):

    .. math::

       F_I(z)=\int_{z_0}^z \omega_I

    This function should only be used when the paths chosen for the integration remain
    in the disc coordinate system where :math:`\omega_I` is holomorphic.

    Parameters
    ----------
    normalized_one_forms : sequence of callable
        A-normalized holomorphic one-forms in one coordinate chart.
    basepoint : complex, optional
        Common lower endpoint of every primitive.
    quadrature_order : int, optional
        Positive Gauss--Legendre quadrature order.

    Returns
    -------
    tuple of callable
        One cached primitive for each supplied one-form.
    """

    forms = tuple(normalized_one_forms)
    if not forms or not all(callable(form) for form in forms):
        raise TypeError("normalized_one_forms must be a nonempty sequence of callables")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral):
        raise TypeError("quadrature_order must be an integer")
    quadrature_order = int(quadrature_order)
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")
    basepoint = np.complex128(basepoint)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    parameters = 0.5 * (nodes + 1.0)
    scaled_weights = 0.5 * weights
    primitives = []
    for form in forms:
        cache: dict[tuple[float, float], np.complex128] = {}

        def primitive(point, *, _form=form, _cache=cache):
            point = np.complex128(point)
            key = (float(point.real), float(point.imag))
            if key not in _cache:
                displacement = point - basepoint
                values = np.asarray(
                    [
                        np.complex128(
                            _form(basepoint + parameter * displacement)
                        )
                        for parameter in parameters
                    ],
                    dtype=np.complex128,
                )
                _cache[key] = np.complex128(
                    displacement * np.dot(scaled_weights, values)
                )
            return _cache[key]

        primitives.append(primitive)
    return tuple(primitives)


def abel_jacobi_map(
    point: complex,
    surface: RiemannSurfaceData,
    *,
    basepoint: complex = 0.0j,
) -> ComplexArray:
    r"""Evaluate the Abel-Jacobi map by computing antiderivatives of 
        basis of A cycle normalized holomorphic one forms relative to
        a given basepoint.

    Parameters
    ----------
    point, basepoint : complex
        Endpoints in the coordinate chart used by the supplied primitives.
    surface : RiemannSurfaceData
        Data of the Riemann surface.

    Returns
    -------
    numpy.ndarray
        Complex vector of shape ``(genus,)`` representing
        :math:`\int_{\mathrm{basepoint}}^{\mathrm{point}}\boldsymbol\omega`.
    """

    point = np.complex128(point)
    basepoint = np.complex128(basepoint)
    return np.asarray(
        [
            primitive(point) - primitive(basepoint)
            for primitive in surface.antiderivatives_normalized_forms
        ],
        dtype=np.complex128,
    )


@lru_cache(maxsize=None)
def theta_characteristics(
    genus: int,
    *,
    parity: str | None = None,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    r"""Enumerates theta characteristics associated with a given genus.
       Specifically, it enumerates all :math:`2^{2g}` combinations
       :math:`(\vec{\epsilon},\vec{\eta})`, where
       :math:`\vec{\epsilon},\vec{\eta}\in\{0,\frac{1}{2}\}^{g}`.

    Parameters
    ----------
    genus : int
        Positive genus.
    parity : {``None``, ``"even"``, ``"odd"``}, optional
        Returns only even or odd spin structures. A spin structure is even
        if :math:`4\epsilon\cdot \eta=0\mod 2` and odd otherwise.
    Returns
    -------
    tuple
        Characteristics represented as ``(epsilon, delta)`` pairs whose
        entries are ``0.0`` or ``0.5``.
    """

    if isinstance(genus, bool) or not isinstance(genus, Integral):
        raise TypeError("genus must be an integer")
    genus = int(genus)
    if genus < 1:
        raise ValueError("genus must be positive")
    if parity not in {None, "even", "odd"}:
        raise ValueError("parity must be None, 'even', or 'odd'")

    result = []
    for epsilon in product((0.0, 0.5), repeat=genus):
        for delta in product((0.0, 0.5), repeat=genus):
            characteristic = (epsilon, delta)
            if parity is not None:
                expected = 0 if parity == "even" else 1
                if characteristic_parity(characteristic) != expected:
                    continue
            result.append((tuple(epsilon), tuple(delta)))
    return tuple(result)


def _validate_characteristic(
    characteristic: Characteristic | None,
    genus: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate that a provided candidate theta characteristic has shape (2,g),
       every entry is finite and either 0,1/2. None converts to the trivial characteristic
       (0,0)."""

    if characteristic is None:
        return np.zeros(genus), np.zeros(genus)
    if not isinstance(characteristic, (tuple, list)) or len(characteristic) != 2:
        raise ValueError("a characteristic must be a pair (epsilon, delta)")
    epsilon = np.asarray(characteristic[0], dtype=np.float64)
    delta = np.asarray(characteristic[1], dtype=np.float64)
    if epsilon.shape != (genus,) or delta.shape != (genus,):
        raise ValueError(f"each characteristic vector must have shape ({genus},)")

    for vector in (epsilon, delta):
        if not np.all(np.isfinite(vector)):
            raise ValueError("characteristic entries must be finite")
        valid = np.isclose(vector, 0.0, rtol=0.0, atol=1e-12) | np.isclose(
            vector,
            0.5,
            rtol=0.0,
            atol=1e-12,
        )
        if not np.all(valid):
            raise ValueError("characteristic entries must be 0 or 1/2")
    return epsilon, delta


def characteristic_parity(characteristic: Characteristic) -> int:
    r"""Return the Arf parity of a provided theta characteristic.

       If :math:`\vec{\eta}\cdot \vec{\eta}=0\mod 2`, returns 0 (even),
       otherwise returns 1 (odd).

    Parameters
    ----------
    characteristic : pair of sequences
        Theta characteristic with entries ``0.0`` or ``0.5``.

    Returns
    -------
    int
        Zero for an even characteristic and one for an odd characteristic.
    """

    if not isinstance(characteristic, (tuple, list)) or len(characteristic) != 2:
        raise ValueError("a characteristic must be a pair (epsilon, delta)")
    genus = len(characteristic[0])
    epsilon, delta = _validate_characteristic(characteristic, genus)
    return int(np.rint(4.0 * float(epsilon @ delta))) % 2


def theta_truncation(period_matrix: np.ndarray, *, tolerance: float = 1e-12) -> int:
    r"""For a given period matrix, heuristically choose an upper cutoff for the lattice sum
       in the Riemann theta function evaluation.

       Define :math:`Y=\Im \Omega`. For a given lattice vector :math:`\vec{n}`, the Gaussian
       part of the theta function decays as :math:`\exp(-\pi \lambda_{\text{min}}||n||^2)`,
       where :math:`\lambda_{\text{min}}` is the minimum eigenvalue of Y. The desired maximum n,
       denoted as N, is heuristically chosen such that the asymptotic decay is less than the
       provided tolerance. Specifically,

       .. math::

        N
        =
        \max\!\left[
            4,\,
            \left\lceil
            \sqrt{
                \frac{-\log(\mathrm{tolerance})}
                        {\pi\lambda_{\min}}
            }
            \right\rceil+2
        \right].

    Parameters
    ----------
    period_matrix : numpy.ndarray
        Period matrix of Riemann surface.
    tolerance : float, optional
        Positive number which is heuristically used as the acceptable asymptotic error in the
        theta function.

    Returns
    -------
    int
        Nonnegative Riemann theta cutoff.
    """

    omega = _validate_period_matrix(period_matrix)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    smallest = float(np.min(np.linalg.eigvalsh(np.imag(omega))))
    return max(
        4,
        int(math.ceil(math.sqrt(-math.log(tolerance) / (math.pi * smallest)))) + 2,
    )


def riemann_theta(
    argument: Sequence[complex],
    period_matrix: np.ndarray,
    *,
    characteristic: Characteristic | None = None,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate a Riemann theta function:

    .. math::

       \vartheta
       \begin{bmatrix}
       \epsilon \\
       \delta
       \end{bmatrix}
       (z\mid\Omega)
       =
       \sum_{\boldsymbol{n}\in\{-N,\ldots,N\}^{g}}
       \exp\!\left\{
       i\pi\left[
       (\boldsymbol{n}+\epsilon)^{T}
       \Omega
       (\boldsymbol{n}+\epsilon)
       +
       2(\boldsymbol{n}+\epsilon)^{T}(z+\delta)
       \right]
       \right\}.

    If the theta characteristic :math:`(\epsilon,\delta)`
    is not provided, the formula uses :math:`(\epsilon,\delta)=(0,0)`.

    Parameters
    ----------
    argument : sequence of complex
        Vector :math:`z` of length ``genus``.
    period_matrix : numpy.ndarray
        Matrix in the Siegel upper half space.
    characteristic : pair of sequences or None, optional
        Half-integer theta characteristic with entries ``0.0`` or ``0.5`` and
        shape ``(2, genus)``. If ``None``, all entries are set to zero.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N`. If omitted, :func:`theta_truncation` determines
        :math:`N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff.

    Returns
    -------
    numpy.complex128
        Truncated theta sum.
    """

    theta, _ = _riemann_theta_with_gradient(
        argument,
        period_matrix,
        characteristic=characteristic,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
        compute_gradient=False,
    )
    return theta


def riemann_theta_gradient(
    argument: Sequence[complex],
    period_matrix: np.ndarray,
    *,
    characteristic: Characteristic | None = None,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> ComplexArray:
    r"""Evaluate the gradient of a Riemann theta function with respect to :math:`z`:

    .. math::

       \left(\nabla_z\vartheta\right)_I
       =
       \frac{\partial\vartheta}{\partial z_I}
       =
       2\pi i
       \sum_{\boldsymbol n}
       (n_I+\epsilon_I)
       \exp\!\left\{
       i\pi\left[
       (\boldsymbol n+\epsilon)^T
       \Omega
       (\boldsymbol n+\epsilon)
       +
       2(\boldsymbol n+\epsilon)^T(z+\delta)
       \right]
       \right\}.

    If the theta characteristic :math:`(\epsilon,\delta)` is not provided,
    the formula uses :math:`(\epsilon,\delta)=(0,0)`.

    Parameters
    ----------
    argument : sequence of complex
        Vector :math:`z` of length ``genus``.
    period_matrix : numpy.ndarray
        Matrix in the Siegel upper half space.
    characteristic : pair of sequences or None, optional
        Half-integer theta characteristic with entries ``0.0`` or ``0.5`` and
        shape ``(2, genus)``. If ``None``, all entries are set to zero.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N`. If omitted, :func:`theta_truncation` determines
        :math:`N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff.


    Returns
    -------
    numpy.ndarray
        Gradient with respect to the theta argument, of shape ``(genus,)``.
    """

    _, gradient = _riemann_theta_with_gradient(
        argument,
        period_matrix,
        characteristic=characteristic,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
        compute_gradient=True,
    )
    assert gradient is not None
    return gradient


def igusa_cusp_form_genus_two(
    period_matrix: Sequence[Sequence[complex]],
    *,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate the genus-two Igusa cusp form in the product convention.

    This function uses the convention

    .. math::

       \chi_{10}(\Omega)
       =
       \prod_{\delta\,\mathrm{even}}
       \vartheta[\delta](0\mid\Omega)^2.

    Parameters
    ----------
    period_matrix : sequence of sequence of complex
        Genus-two period matrix in the Siegel upper half-space.
    lattice_cutoff : int or None, optional
        Cutoff for each integer component in the theta sums. If ``None``,
        :func:`theta_truncation` chooses the cutoff.
    tolerance : float, optional
        Tolerance used by :func:`theta_truncation` when ``lattice_cutoff`` is
        not supplied.

    Returns
    -------
    numpy.complex128
        Product-normalized value of :math:`\chi_{10}(\Omega)`.
    """

    omega = _validate_period_matrix(period_matrix)
    if omega.shape != (2, 2):
        raise ValueError(
            "igusa_cusp_form_genus_two requires a period matrix of shape (2, 2)"
        )
    argument = np.zeros(2, dtype=np.complex128)
    value = np.complex128(1.0)
    for characteristic in theta_characteristics(2, parity="even"):
        theta_constant = riemann_theta(
            argument,
            omega,
            characteristic=characteristic,
            lattice_cutoff=lattice_cutoff,
            tolerance=tolerance,
        )
        value *= theta_constant**2
    return np.complex128(value)


def igusa_cusp_form_genus_three(
    period_matrix: Sequence[Sequence[complex]],
    *,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate the genus-three Igusa cusp form in the product convention.

    This function uses the convention

    .. math::

       \chi_{18}(\Omega)
       =
       \prod_{\delta\,\mathrm{even}}
       \vartheta[\delta](0\mid\Omega),

    where the product is over the 36 even genus-three characteristics.

    Parameters
    ----------
    period_matrix : sequence of sequence of complex
        Genus-three period matrix in the Siegel upper half-space.
    lattice_cutoff : int or None, optional
        Cutoff for each integer component in the theta sums. If ``None``,
        :func:`theta_truncation` chooses the cutoff.
    tolerance : float, optional
        Tolerance used by :func:`theta_truncation` when ``lattice_cutoff`` is
        not supplied.

    Returns
    -------
    numpy.complex128
        Product-normalized value of :math:`\chi_{18}(\Omega)`.
    """

    omega = _validate_period_matrix(period_matrix)
    if omega.shape != (3, 3):
        raise ValueError(
            "igusa_cusp_form_genus_three requires a period matrix of shape (3, 3)"
        )
    argument = np.zeros(3, dtype=np.complex128)
    value = np.complex128(1.0)
    even_characteristics = theta_characteristics(3, parity="even")
    if len(even_characteristics) != 36:
        raise RuntimeError("genus three must have exactly 36 even characteristics")
    for characteristic in even_characteristics:
        value *= riemann_theta(
            argument,
            omega,
            characteristic=characteristic,
            lattice_cutoff=lattice_cutoff,
            tolerance=tolerance,
        )
    return np.complex128(value)


def _riemann_theta_with_gradient(
    argument: Sequence[complex],
    period_matrix: np.ndarray,
    *,
    characteristic: Characteristic | None,
    lattice_cutoff: int | None,
    tolerance: float,
    compute_gradient: bool,
) -> tuple[np.complex128, ComplexArray | None]:
    """Helper function to evaluate the Riemann theta function and optionally
    its gradient.

    Parameters
    ----------
    argument : sequence of complex
        Vector :math:`z` of length ``genus``.
    period_matrix : numpy.ndarray
        Matrix in the Siegel upper half space.
    characteristic : pair of sequences or None, optional
        Half integer theta characteristic with entries ``0.0`` or ``0.5``. Has shape (2,genus).
        If None, all entries are set to zero.
    lattice_cutoff : int or None, optional
        Cutoff `N`.  If omitted, :func:`theta_truncation` is used to determine an `N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff.
    compute_gradient : bool
        Whether to evaluate the gradient with respect to ``argument``.

    Returns
    -------
    tuple of (numpy.complex128, numpy.ndarray or None)
        Truncated theta sum and, when ``compute_gradient`` is true, its complex
        argument gradient of shape ``(genus,)``. Otherwise, the second entry is
        ``None``.
    """

    evaluator = _prepare_riemann_theta_evaluator(
        period_matrix,
        characteristic=characteristic,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )
    return evaluator(argument, compute_gradient)


def _prepare_riemann_theta_evaluator(
    period_matrix: np.ndarray,
    *,
    characteristic: Characteristic | None,
    lattice_cutoff: int | None,
    tolerance: float,
) -> Callable[
    [Sequence[complex], bool],
    tuple[np.complex128, ComplexArray | None],
]:
    """Prepare the argument-independent data of a truncated theta sum.

    Parameters
    ----------
    period_matrix : numpy.ndarray
        Matrix in the Siegel upper half space.
    characteristic : pair of sequences or None
        Half-integer theta characteristic with entries ``0.0`` or ``0.5``.
    lattice_cutoff : int or None
        Component-wise cutoff. If omitted, :func:`theta_truncation` determines
        the cutoff.
    tolerance : float
        Tolerance passed to :func:`theta_truncation` when selecting a cutoff.

    Returns
    -------
    callable
        Evaluator accepting a theta argument and a Boolean gradient flag and
        returning the theta value and optional argument gradient.
    """

    omega = _validate_period_matrix(period_matrix)
    genus = omega.shape[0]
    epsilon, delta = _validate_characteristic(characteristic, genus)
    if lattice_cutoff is None:
        lattice_cutoff = theta_truncation(omega, tolerance=tolerance)
    if isinstance(lattice_cutoff, bool) or not isinstance(lattice_cutoff, Integral):
        raise TypeError("lattice_cutoff must be an integer or None")
    lattice_cutoff = int(lattice_cutoff)
    if lattice_cutoff < 0:
        raise ValueError("lattice_cutoff must be nonnegative")
    coordinates = range(-lattice_cutoff, lattice_cutoff + 1)
    lattice = np.asarray(
        tuple(product(coordinates, repeat=genus)),
        dtype=np.float64,
    )
    shifted_lattice = lattice + epsilon
    fixed_exponent = 1j * math.pi * (
        np.einsum(
            "ni,ij,nj->n",
            shifted_lattice,
            omega,
            shifted_lattice,
            optimize=True,
        )
        + 2.0 * (shifted_lattice @ delta)
    )

    def evaluate(
        argument: Sequence[complex],
        compute_gradient: bool,
    ) -> tuple[np.complex128, ComplexArray | None]:
        argument_array = np.asarray(argument, dtype=np.complex128)
        if argument_array.shape != (genus,):
            raise ValueError(f"argument must have shape ({genus},)")
        if not np.all(np.isfinite(argument_array)):
            raise ValueError("argument contains a non-finite entry")
        terms = np.exp(
            fixed_exponent
            + 2j * math.pi * (shifted_lattice @ argument_array)
        )
        theta = np.complex128(np.sum(terms))
        gradient = None
        if compute_gradient:
            gradient = np.asarray(
                (2j * math.pi)
                * np.sum(terms[:, None] * shifted_lattice, axis=0),
                dtype=np.complex128,
            )
        return theta, gradient

    return evaluate


def _prepare_surface_point_evaluators(
    surface: RiemannSurfaceData,
) -> tuple[Callable[[complex], ComplexArray], Callable[[complex], ComplexArray]]:
    """Prepare cached Abel maps and one-form values for points on one surface.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.

    Returns
    -------
    tuple of callable
        Functions for the Abel--Jacobi map and the vector of normalized
        holomorphic one-form values, respectively, for fixed holomorphic data.
    """

    abel_cache: dict[complex, ComplexArray] = {}
    form_cache: dict[complex, ComplexArray] = {}

    def abel(point: complex) -> ComplexArray:
        key = complex(np.complex128(point))
        if key not in abel_cache:
            abel_cache[key] = abel_jacobi_map(key, surface)
        return abel_cache[key]

    def form_values(point: complex) -> ComplexArray:
        key = complex(np.complex128(point))
        if key not in form_cache:
            form_cache[key] = np.asarray(
                [
                    np.complex128(form(np.complex128(key)))
                    for form in surface.normalized_one_forms
                ],
                dtype=np.complex128,
            )
        return form_cache[key]

    return abel, form_values


def _prepare_prime_form_evaluator(
    surface: RiemannSurfaceData,
    *,
    characteristic: Characteristic | None,
    lattice_cutoff: int | None,
    tolerance: float,
    abel_evaluator: Callable[[complex], ComplexArray],
    form_values_evaluator: Callable[[complex], ComplexArray],
) -> Callable[[complex, complex], np.complex128]:
    """Helper function that prepares the fixed theta data used by prime forms on one surface.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.
    characteristic : pair of sequences or None
        Odd half integer theta characteristic. If omitted, the first odd
        characteristic in :func:`theta_characteristics` is used.
    lattice_cutoff : int or None
        Cutoff used in the Riemann theta sum. If omitted, :func:`theta_truncation` determines
        the cutoff.
    tolerance : float
        Tolerance used by :func:`theta_truncation` to select a cutoff.
    abel_evaluator, form_values_evaluator : callable
        Cached evaluators returned by :func:`_prepare_surface_point_evaluators`.

    Returns
    -------
    callable
        Function that accepts two points and and returns their prime form.
    """

    if characteristic is None:
        odd_characteristics = theta_characteristics(surface.genus, parity="odd")
        if not odd_characteristics:
            raise ValueError(f"no odd characteristic exists at genus {surface.genus}")
        characteristic = odd_characteristics[0]
    if characteristic_parity(characteristic) != 1:
        raise ValueError("prime_form requires an odd characteristic")

    theta_evaluator = _prepare_riemann_theta_evaluator(
        surface.Omega,
        characteristic=characteristic,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )
    _, gradient = theta_evaluator(
        np.zeros(surface.genus, dtype=np.complex128),
        True,
    )
    assert gradient is not None

    def evaluate(point: complex, reference_point: complex) -> np.complex128:
        difference = abel_evaluator(point) - abel_evaluator(reference_point)
        numerator, _ = theta_evaluator(difference, False)
        spin_at_point = form_values_evaluator(point) @ gradient
        spin_at_reference = form_values_evaluator(reference_point) @ gradient
        denominator = np.sqrt(spin_at_point) * np.sqrt(spin_at_reference)
        if abs(denominator) <= np.finfo(float).tiny:
            raise ZeroDivisionError("the prime-form denominator vanished")
        return np.complex128(numerator / denominator)

    return evaluate


def prime_form(
    point: complex,
    reference_point: complex,
    surface: RiemannSurfaceData,
    *,
    characteristic: Characteristic | None = None,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate the prime form :math:`E(\mathrm{point},\mathrm{reference})`:

    .. math::

       E(z,w)
       =
       \frac{
           \theta[\delta]\!\left(
               \zeta(z)-\zeta(w)\mid\Omega
           \right)
       }{
           \sqrt{
               \omega[\delta](z)\,
               \omega[\delta](w)
           }
       },

    where :math:`\zeta` is the Abel-Jacobi map evaluated in the function
    :func:`abel_jacobi_map` and
    :math:`\omega[\delta](z)` is a holomorphic one-form defined as

    .. math::

       \omega[\delta](z)
       =
       \left.
       \omega_I(z)
       \frac{\partial}{\partial y_I}
       \theta[\delta](y\mid\Omega)
       \right|_{y=0}.

    The prime form is the unique holomorphic differential of weight :math:`-\frac{1}{2}`
    that vanishes only at :math:`z=w`. It can be shown that it is independent of the odd
    spin structure.

    Parameters
    ----------
    point : complex
        Local coordinate :math:`z` of the first point.
    reference_point : complex
        Local coordinate :math:`w` of the second point.
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.
    characteristic : pair of sequences or None, optional
        Odd half-integer theta characteristic with entries ``0.0`` or ``0.5``
        and shape ``(2, genus)``. If ``None``, the first odd characteristic
        returned by :func:`theta_characteristics` is used.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N`. If omitted, :func:`theta_truncation` determines
        :math:`N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff.

    Returns
    -------
    numpy.complex128
        Prime-form value in the supplied local coordinates.
    """

    abel_evaluator, form_values_evaluator = _prepare_surface_point_evaluators(surface)
    evaluator = _prepare_prime_form_evaluator(
        surface,
        characteristic=characteristic,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
        abel_evaluator=abel_evaluator,
        form_values_evaluator=form_values_evaluator,
    )
    return evaluator(point, reference_point)


def _riemann_constant_vector_genus_one(
    surface: RiemannSurfaceData,
) -> ComplexArray:
    r"""Return the Riemann constant vector :math:`\Delta` for a genus one surface as

    .. math::

       \Delta=\frac{1-\tau}{2}\pmod{\mathbb Z+\tau\mathbb Z}.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Genus one Riemann surface data. The modulus is
        :math:`\tau=\Omega_{11}`.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(1,)`` containing one representative of
        :math:`\Delta` modulo :math:`\mathbb Z+\tau\mathbb Z`.

    Raises
    ------
    ValueError
        If ``surface`` does not have genus one.
    """

    if surface.genus != 1:
        raise ValueError(
            "_riemann_constant_vector_genus_one requires a genus-one surface"
        )
    return np.asarray(
        [0.5 * (1.0 - surface.Omega[0, 0])],
        dtype=np.complex128,
    )


def _riemann_constant_from_canonical_divisor(
    surface: RiemannSurfaceData,
    canonical_divisor_points: Sequence[complex],
    *,
    filter_divisors: Sequence[Sequence[complex]],
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> ComplexArray:
    r"""Helper function that determine the Riemann constant from a canonical divisor,
    for genus greater than 1.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Surface and Abel map data.
    canonical_divisor_points : sequence of complex
        Zeros of a holomorphic one form, including multiplicity.
    filter_divisors : sequence of sequences of complex
        Generic set of points of length ``genus - 1`` to test the Riemann vanishing
        theorem from a set of candidate Riemann constants.

    Returns
    -------
    numpy.ndarray
        Selected representative of shape ``(genus,)``.
    """

    points = tuple(complex(point) for point in canonical_divisor_points)
    expected = 2 * surface.genus - 2
    if len(points) != expected:
        raise ValueError(f"a canonical divisor at genus {surface.genus} has degree {expected}")
    divisors = tuple(tuple(complex(point) for point in divisor) for divisor in filter_divisors)
    if not divisors or any(len(divisor) != surface.genus - 1 for divisor in divisors):
        raise ValueError("each filter divisor must have degree genus - 1")

    canonical_abel = np.sum(
        np.asarray([abel_jacobi_map(point, surface) for point in points], dtype=np.complex128),
        axis=0,
    ) if points else np.zeros(surface.genus, dtype=np.complex128)
    base = 0.5 * canonical_abel
    divisor_abel = tuple(
        np.sum(
            np.asarray([abel_jacobi_map(point, surface) for point in divisor], dtype=np.complex128),
            axis=0,
        ) if divisor else np.zeros(surface.genus, dtype=np.complex128)
        for divisor in divisors
    )

    best_score = math.inf
    best = None
    for integer_half in product((0.0, 0.5), repeat=surface.genus):
        for period_half in product((0.0, 0.5), repeat=surface.genus):
            candidate = (
                base
                + np.asarray(integer_half, dtype=np.complex128)
                + surface.Omega @ np.asarray(period_half, dtype=np.complex128)
            )
            score = math.fsum(
                abs(
                    riemann_theta(
                        divisor - candidate,
                        surface.Omega,
                        lattice_cutoff=lattice_cutoff,
                        tolerance=tolerance,
                    )
                ) ** 2
                for divisor in divisor_abel
            )
            if score < best_score:
                best_score = score
                best = candidate
    assert best is not None
    return np.asarray(best, dtype=np.complex128)


def riemann_constant_vector(
    surface: RiemannSurfaceData,
    canonical_divisor_points: Sequence[complex] | None = None,
    *,
    filter_divisors: Sequence[Sequence[complex]] | None = None,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> ComplexArray:
    r"""Return a Riemann constant vector for a surface of arbitrary genus.

    At genus one, the exact expression

    .. math::

       \Delta=\frac{1-\tau}{2}\pmod{\mathbb Z+\tau\mathbb Z}

    is used.

    At higher genus, given a list of the zeroes :math:`c_r` and their multiplicities
    :math:`m_r` of a given holomorphic one form in ``canonical_divisor_points``, the function
    computes the candidate Riemann constants

    .. math::

           \Delta_{\epsilon,\delta}=\frac{1}{2} \sum_r m_r\zeta(c_r)+\epsilon+\Omega \delta,

    where :math:`\zeta` is the Abel-Jacobi map. :math:`[\epsilon,\delta]` is a characteristic of length
    (2,g), with values 0 or :math:`\frac{1}{2}`. From these candidates, the one that (up to numerical tolerance)
    satisfies the Riemann vanishing theorem for the sets of points in ``filter_divisors`` is chosen.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.
    canonical_divisor_points : sequence of complex or None, optional
        Zeros of a holomorphic one form, repeated according to multiplicity.
        Exactly :math:`2g - 2` points are required when ``genus > 1`` and
        the argument is ignored at genus one.
    filter_divisors : sequence of sequences of complex or None, optional
        Collection of :math:`M` sets of points, each of length :math:`g-1`, to
        test the Riemann vanishing theorem against a set of candidate Riemann
        constants. In our code, we use one set of points.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N` used in the Riemann theta sum. If omitted,
        :func:`theta_truncation` determines :math:`N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff in the Riemann theta sum.
    Returns
    -------
    numpy.ndarray
        Complex Riemann constant vector of shape ``(genus,)``.

    Raises
    ------
    ValueError
        If the required higher-genus divisor data are absent or have invalid
        degrees.
    """

    if surface.genus == 1:
        return _riemann_constant_vector_genus_one(surface)
    if canonical_divisor_points is None:
        raise ValueError("canonical_divisor_points are required when genus > 1")
    if filter_divisors is None:
        raise ValueError("filter_divisors are required when genus > 1")
    return _riemann_constant_from_canonical_divisor(
        surface,
        canonical_divisor_points,
        filter_divisors=filter_divisors,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )


def _prepare_sigma_ratio_evaluator(
    surface: RiemannSurfaceData,
    *,
    reference_point: complex,
    divisor_points: Sequence[complex],
    riemann_constant: Sequence[complex],
    theta_evaluator: Callable[
        [Sequence[complex], bool],
        tuple[np.complex128, ComplexArray | None],
    ],
    prime_form_evaluator: Callable[[complex, complex], np.complex128],
    abel_evaluator: Callable[[complex], ComplexArray],
) -> Callable[[complex], np.complex128]:
    """Helper function to prepare the data used in the sigma function ratio
    that does not depend on the evaluated point :math:`z`.

    Parameters
    ----------
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.
    reference_point : complex
        Fixed denominator point in the sigma ratio.
    divisor_points : sequence of complex
        The :math:`g` auxiliary points entering the theta and prime-form
        factors.
    riemann_constant : sequence of complex
        Riemann constant vector of shape ``(genus,)`` in the Strebel
        convention.
    theta_evaluator : callable
        Function that given z, returns the theta function for fixed 
        holomorphic data.
    prime_form_evaluator : callable
        Function that given z, returns the prime form for fixed holomorphic
        data.
    abel_evaluator : callable
        Function that given z, returns the Abel-Jacobi map for fixed holomorphic
        data.

    Returns
    -------
    callable
        Function accepting a numerator point and returning its sigma ratio
        relative to ``reference_point``.
    """

    divisors = tuple(np.complex128(value) for value in divisor_points)
    if len(divisors) != surface.genus:
        raise ValueError(f"divisor_points must contain genus={surface.genus} points")
    reference_point = np.complex128(reference_point)
    for divisor in divisors:
        if abs(divisor - reference_point) < 1e-12:
            raise ValueError("divisor points must not coincide with ratio endpoints")
    riemann_constant_array = np.asarray(riemann_constant, dtype=np.complex128)
    if riemann_constant_array.shape != (surface.genus,):
        raise ValueError(f"riemann_constant must have shape ({surface.genus},)")

    divisor_sum = np.sum(
        np.asarray([abel_evaluator(value) for value in divisors]),
        axis=0,
    )
    denominator_theta, _ = theta_evaluator(
        divisor_sum - abel_evaluator(reference_point) - riemann_constant_array,
        False,
    )
    if abs(denominator_theta) <= np.finfo(float).tiny:
        raise ZeroDivisionError("the denominator theta factor vanished")
    fixed_prime_product = math.prod(
        (
            prime_form_evaluator(divisor, reference_point)
            for divisor in divisors
        ),
        start=1.0 + 0.0j,
    )

    def evaluate(point: complex) -> np.complex128:
        point = np.complex128(point)
        if any(abs(divisor - point) < 1e-12 for divisor in divisors):
            raise ValueError("divisor points must not coincide with ratio endpoints")
        numerator_theta, _ = theta_evaluator(
            divisor_sum - abel_evaluator(point) - riemann_constant_array,
            False,
        )
        variable_prime_product = math.prod(
            (prime_form_evaluator(divisor, point) for divisor in divisors),
            start=1.0 + 0.0j,
        )
        if abs(variable_prime_product) <= np.finfo(float).tiny:
            raise ZeroDivisionError("the denominator prime-form product vanished")
        return np.complex128(
            numerator_theta
            * fixed_prime_product
            / (denominator_theta * variable_prime_product)
        )

    return evaluate


def sigma_ratio(
    point: complex,
    reference_point: complex,
    surface: RiemannSurfaceData,
    *,
    divisor_points: Sequence[complex],
    riemann_constant: Sequence[complex],
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate the normalized ratio :math:`\frac{\sigma(z)}{\sigma(w)}` for a fixed
    ``reference_point`` :math:`w`.

    The formula computed is

    .. math::

       \frac{\sigma(z)}{\sigma(w)}
       =
       \frac{
           \vartheta\!\left(
               \sum_{i=1}^{g}\zeta(p_i)-\zeta(z)-\Delta
               \,\middle|\,\Omega
           \right)
       }{
           \vartheta\!\left(
               \sum_{i=1}^{g}\zeta(p_i)-\zeta(w)-\Delta
               \,\middle|\,\Omega
           \right)
       }
       \prod_{i=1}^{g}
       \frac{E(p_i,w)}{E(p_i,z)},

    where :math:`\vartheta` is the Riemann theta function, :math:`E(z,w)` is
    the prime form, :math:`\zeta` is the Abel-jacobi map, :math:`\Delta` is the Riemann
    constant, and :math:`p_i` is a set of :math:`g` arbitrary points. The expression is independent of the choice
    of these :math:`p_i`.

    Parameters
    ----------
    point : complex
        Local coordinate :math:`z` at which the numerator sigma function is
        evaluated.
    reference_point : complex
        Local coordinate :math:`w` at which the denominator sigma function is
        evaluated.
    surface : RiemannSurfaceData
        Holomorphic data for the Riemann surface.
    divisor_points : sequence of complex
        The :math:`g` auxiliary points :math:`p_i` entering the theta and
        prime-form factors.
    riemann_constant : sequence of complex
        Riemann constant vector :math:`\Delta` of shape ``(genus,)`` in the
        Strebel convention. Computed using :func:`riemann_constant_vector`.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N` used in the Riemann theta sum. If omitted,
        :func:`theta_truncation` determines :math:`N`.
    tolerance : float, optional
        Tolerance level passed to :func:`theta_truncation` when selecting a cutoff in the Riemann theta sum.

    Returns
    -------
    numpy.complex128
        Sigma ratio in the supplied branch convention.
    """

    abel_evaluator, form_values_evaluator = _prepare_surface_point_evaluators(surface)
    theta_evaluator = _prepare_riemann_theta_evaluator(
        surface.Omega,
        characteristic=None,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )
    prime_form_evaluator = _prepare_prime_form_evaluator(
        surface,
        characteristic=None,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
        abel_evaluator=abel_evaluator,
        form_values_evaluator=form_values_evaluator,
    )
    evaluator = _prepare_sigma_ratio_evaluator(
        surface,
        reference_point=reference_point,
        divisor_points=divisor_points,
        riemann_constant=riemann_constant,
        theta_evaluator=theta_evaluator,
        prime_form_evaluator=prime_form_evaluator,
        abel_evaluator=abel_evaluator,
    )
    return evaluator(point)


def _ghost_number_selection_rule(
    lambda_weight: float,
    genus: int,
    *,
    n_b: int,
    n_c: int,
) -> None:
    r"""Determines whether a provided bc ghost correlator can be nonzero, i.e. it saturates
    the ghost number anomaly condition.

    Parameters
    ----------
    lambda_weight : float
        Conformal weight :math:`\lambda` of the :math:`b_\lambda` ghost.
    genus : int
        Genus of the Riemann surface.
    n_b : int
        Number of :math:`b_\lambda` insertions.
    n_c : int
        Number of :math:`c_{1-\lambda}` insertions.

    Raises
    ------
    ValueError
        If the required net ghost number is not an integer or the supplied
        number of b and c ghosts does not saturate the ghost number anomaly.
    """

    expected = (1.0 - 2.0 * float(lambda_weight)) * (genus - 1)
    rounded = int(round(expected))
    if not math.isclose(expected, rounded, abs_tol=1e-12):
        raise ValueError("the ghost-number anomaly is not integral")
    if n_c - n_b != rounded:
        raise ValueError(
            f"ghost-number selection requires n_c - n_b = {rounded}, got {n_c - n_b}"
        )


def prepare_bc_correlator(
    data: BCGhostCorrelatorData,
    *,
    lambda_weight: float,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> Callable[[Sequence[complex], Sequence[complex]], np.complex128]:
    r"""Prepare all the data necessary to compute the bc correlation function.

    The returned function accepts the insertion points ``(b_points, c_points)`` 
    and reuses cached Abel-Jacobi maps, one-form values, prime forms, and 
    sigma ratios across evaluations.

    Parameters
    ----------
    data : BCGhostCorrelatorData
        Fixed surface and normalization data.
    lambda_weight : float
        Holomorphic conformal weight :math:`\lambda` of the :math:`b` field.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N` for the Riemann theta sums.  If omitted,
        :func:`theta_truncation` selects it.
    tolerance : float, optional
        Tolerance passed to :func:`theta_truncation` when the cutoff is
        selected automatically.

    Returns
    -------
    callable
        Function accepting ``(b_points, c_points)`` and returning the full
        normalized holomorphic :math:`bc` correlator.
    """

    if not isinstance(data, BCGhostCorrelatorData):
        raise TypeError("data must be BCGhostCorrelatorData")
    if isinstance(lambda_weight, bool) or not isinstance(lambda_weight, Real):
        raise TypeError("lambda_weight must be a real number")
    lambda_weight = float(lambda_weight)
    if not math.isfinite(lambda_weight):
        raise ValueError("lambda_weight must be finite")

    surface = data.surface
    weight = 2.0 * lambda_weight - 1.0
    abel_evaluator, form_values_evaluator = _prepare_surface_point_evaluators(
        surface
    )
    theta_evaluator = _prepare_riemann_theta_evaluator(
        surface.Omega,
        characteristic=None,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )
    prime_form_evaluator = _prepare_prime_form_evaluator(
        surface,
        characteristic=None,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
        abel_evaluator=abel_evaluator,
        form_values_evaluator=form_values_evaluator,
    )
    sigma_ratio_evaluator = _prepare_sigma_ratio_evaluator(
        surface,
        reference_point=data.normalization_point,
        divisor_points=data.divisor_points,
        riemann_constant=data.riemann_constant,
        theta_evaluator=theta_evaluator,
        prime_form_evaluator=prime_form_evaluator,
        abel_evaluator=abel_evaluator,
    )
    normalization_prefactor = np.complex128(
        (2.0 * math.pi) ** (16 * (surface.genus - 1))
        / np.sqrt(np.complex128(data.chiral_z1))
    )
    zero = np.zeros(surface.genus, dtype=np.complex128)
    prime_form_cache: dict[tuple[complex, complex], np.complex128] = {}
    sigma_cache: dict[complex, np.complex128] = {}

    def cached_prime_form(
        point: complex,
        reference_point: complex,
    ) -> np.complex128:
        key = (complex(point), complex(reference_point))
        if key not in prime_form_cache:
            prime_form_cache[key] = prime_form_evaluator(*key)
        return prime_form_cache[key]

    def sigma(point: complex) -> np.complex128:
        key = complex(point)
        if key not in sigma_cache:
            sigma_cache[key] = np.complex128(
                data.sigma_normalization * sigma_ratio_evaluator(key)
            )
        return sigma_cache[key]

    def evaluate(
        b_points: Sequence[complex],
        c_points: Sequence[complex],
    ) -> np.complex128:
        b_points = tuple(complex(np.complex128(value)) for value in b_points)
        c_points = tuple(complex(np.complex128(value)) for value in c_points)
        if not all(np.isfinite(value) for value in (*b_points, *c_points)):
            raise ValueError("insertion points must be finite")
        _ghost_number_selection_rule(
            lambda_weight,
            surface.genus,
            n_b=len(b_points),
            n_c=len(c_points),
        )

        b_abel = (
            np.sum(
                np.asarray([abel_evaluator(value) for value in b_points]),
                axis=0,
            )
            if b_points
            else zero
        )
        c_abel = (
            np.sum(
                np.asarray([abel_evaluator(value) for value in c_points]),
                axis=0,
            )
            if c_points
            else zero
        )
        theta, _ = theta_evaluator(
            b_abel - c_abel - weight * data.riemann_constant,
            False,
        )

        b_prime = math.prod(
            (
                cached_prime_form(point, second_point)
                for index, point in enumerate(b_points)
                for second_point in b_points[index + 1 :]
            ),
            start=1.0 + 0.0j,
        )
        c_prime = math.prod(
            (
                cached_prime_form(point, second_point)
                for index, point in enumerate(c_points)
                for second_point in c_points[index + 1 :]
            ),
            start=1.0 + 0.0j,
        )
        mixed_prime = math.prod(
            (
                cached_prime_form(b_point, c_point)
                for b_point in b_points
                for c_point in c_points
            ),
            start=1.0 + 0.0j,
        )
        if b_points and c_points and abs(mixed_prime) <= np.finfo(float).tiny:
            raise ZeroDivisionError("the mixed prime-form product vanished")

        sigma_b = math.prod(
            (sigma(point) ** weight for point in b_points),
            start=1.0 + 0.0j,
        )
        sigma_c = math.prod(
            (sigma(point) ** weight for point in c_points),
            start=1.0 + 0.0j,
        )
        if c_points and abs(sigma_c) <= np.finfo(float).tiny:
            raise ZeroDivisionError("the c-insertion sigma product vanished")
        geometric = np.complex128(
            theta * b_prime * c_prime * sigma_b / (mixed_prime * sigma_c)
        )
        return np.complex128(normalization_prefactor * geometric)

    return evaluate


def bc_correlator(
    b_points: Sequence[complex],
    c_points: Sequence[complex],
    data: BCGhostCorrelatorData,
    *,
    lambda_weight: float,
    lattice_cutoff: int | None = None,
    tolerance: float = 1e-12,
) -> np.complex128:
    r"""Evaluate a holomorphic :math:`bc` correlator for general :math:`b` holomorphic
    weight :math:`\lambda` using the formula of Verlinde and Verlinde (1987):

    .. math::

       \left\langle
       \prod_{i=1}^{n_b} b_{\lambda}(z_i)
       \prod_{j=1}^{n_c} c_{1-\lambda}(w_j)
       \right\rangle
       =
       \frac{(2\pi)^{16(g-1)}}{\sqrt{Z_1}}\,
       \vartheta\!\left(
       \sum_{i=1}^{n_b}\zeta(z_i)
       -
       \sum_{j=1}^{n_c}\zeta(w_j)
       -
       (2\lambda-1)\Delta
       \,\middle|\,\Omega
       \right)
       \frac{
       \displaystyle
       \prod_{i<i'}E(z_i,z_{i'})
       \prod_{j<j'}E(w_j,w_{j'})
       }{
       \displaystyle
       \prod_{i,j}E(z_i,w_j)
       }
       \frac{
       \displaystyle
       \prod_{i=1}^{n_b}\sigma(z_i)^{2\lambda-1}
       }{
       \displaystyle
       \prod_{j=1}^{n_c}\sigma(w_j)^{2\lambda-1}
       }.

    Here :math:`\zeta` is the Abel-Jacobi map, :math:`\Delta` is the
    Riemann constant vector, :math:`E` is the prime form, and
    :math:`Z_1` is the chiral partition function. The number of b and c ghosts must
    satisfy the following equation to saturate the ghost number anomaly:

    .. math::

       n_c-n_b=(1-2\lambda)(g-1).

    Parameters
    ----------
    b_points : sequence of complex
        Coordinates :math:`z_i` of the :math:`b_{\lambda}` insertions.
    c_points : sequence of complex
        Coordinates :math:`w_j` of the :math:`c_{1-\lambda}` insertions.
    data : BCGhostCorrelatorData
        Fixed surface and normalization data.
    lambda_weight : float
        Holomorphic conformal weight :math:`\lambda` of the :math:`b` field.
        The corresponding :math:`c` field has weight :math:`1-\lambda`.
    lattice_cutoff : int or None, optional
        Cutoff :math:`N` for the lattice sums defining the Riemann theta
        functions. If ``None``, :func:`theta_truncation` selects the cutoff.
    tolerance : float, optional
        Tolerance passed to :func:`theta_truncation` when ``lattice_cutoff``
        is not supplied.

    Returns
    -------
    numpy.complex128
        Full normalized holomorphic :math:`bc` ghost correlator.
    """

    evaluator = prepare_bc_correlator(
        data,
        lambda_weight=lambda_weight,
        lattice_cutoff=lattice_cutoff,
        tolerance=tolerance,
    )
    return evaluator(b_points, c_points)


__all__ = (
    "BCGhostCorrelatorData",
    "RiemannSurfaceData",
    "abel_jacobi_map",
    "bc_correlator",
    "characteristic_parity",
    "igusa_cusp_form_genus_two",
    "prime_form",
    "prepare_bc_correlator",
    "riemann_constant_vector",
    "riemann_theta",
    "riemann_theta_gradient",
    "sigma_ratio",
    "holomorphic_one_form_antiderivatives",
    "theta_characteristics",
    "theta_truncation",
)
