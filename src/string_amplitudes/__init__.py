"""Numerical tools for string-amplitude calculations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("string-amplitudes")
except PackageNotFoundError:  # Support direct source-tree imports.
    __version__ = "0+unknown"

from .ribbon_graph_to_period_matrix import (
    DiscHolomorphicOneForm,
    PeriodMapResult,
    compute_period_map,
)
from .free_boson_correlators import (
    FreeBosonOnePointFunctions,
    free_boson_one_point_functions,
    genus_one_disk_frame_one_point_functions,
    genus_one_period_matrix_one_point_functions,
)
from .analytic_string_integrand import (
    genus_three_period_matrix_integrand,
    genus_two_period_matrix_integrand,
)
from .partition_function import (
    compact_boson_partition_function,
    identity_wavefunctional_kernel,
    identity_wavefunctional_kernel_reduced,
    matter_log_determinant,
)
from .ribbon_graph_generator import (
    generate_ribbon_graphs,
    get_boundary_data,
    sample_ribbon_graph,
)
from .riemann_surface_holomorphic_data import (
    BCGhostCorrelatorData,
    RiemannSurfaceData,
    abel_jacobi_map,
    bc_correlator,
    characteristic_parity,
    igusa_cusp_form_genus_three,
    igusa_cusp_form_genus_two,
    prime_form,
    prepare_bc_correlator,
    riemann_constant_vector,
    riemann_theta,
    riemann_theta_gradient,
    sigma_ratio,
    holomorphic_one_form_antiderivatives,
    theta_characteristics,
    theta_truncation,
)
from .string_integrand import (
    bghost_direction_coefficients,
    bghost_measure,
    bghost_measure_from_edge_components,
    critical_bosonic_string_integrand,
    integrated_bghost_edge_components,
)

__all__ = (
    "__version__",
    "compact_boson_partition_function",
    "compute_period_map",
    "DiscHolomorphicOneForm",
    "FreeBosonOnePointFunctions",
    "PeriodMapResult",
    "BCGhostCorrelatorData",
    "RiemannSurfaceData",
    "abel_jacobi_map",
    "bc_correlator",
    "bghost_direction_coefficients",
    "bghost_measure",
    "bghost_measure_from_edge_components",
    "characteristic_parity",
    "critical_bosonic_string_integrand",
    "genus_three_period_matrix_integrand",
    "genus_one_disk_frame_one_point_functions",
    "genus_one_period_matrix_one_point_functions",
    "genus_two_period_matrix_integrand",
    "generate_ribbon_graphs",
    "free_boson_one_point_functions",
    "get_boundary_data",
    "identity_wavefunctional_kernel",
    "identity_wavefunctional_kernel_reduced",
    "igusa_cusp_form_genus_three",
    "igusa_cusp_form_genus_two",
    "integrated_bghost_edge_components",
    "matter_log_determinant",
    "prime_form",
    "prepare_bc_correlator",
    "riemann_constant_vector",
    "riemann_theta",
    "riemann_theta_gradient",
    "sample_ribbon_graph",
    "sigma_ratio",
    "holomorphic_one_form_antiderivatives",
    "theta_characteristics",
    "theta_truncation",
)
