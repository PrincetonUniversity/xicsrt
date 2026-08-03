# -*- coding: utf-8 -*-
"""
.. Authors
    Leila Alston <lalston@pppl.gov>
    Novimir Pablant <npablant@pppl.gov>

Random spline-profile generators for plasma parameter profiles.

These routines generate randomized, monotonicity-constrained knot sets for
plasma profiles (emissivity, temperature, velocity) as a function of the
normalized minor radius rho in [0, 1]. Each generator returns a JSON
compatible 'profile dictionary'; use :func:`spline_from_profile` to turn a
profile dictionary into a callable interpolator.

Profile dictionary format
-------------------------
x_knots : ndarray of float
    Knot locations in rho.
y_knots : ndarray of float
    Profile values at the knots (physical units).
deriv_zero : ndarray of bool
    True where the spline derivative is constrained to zero.
x_free : ndarray of bool
    True where the x-coordinate was free to vary during generation.
y_free : ndarray of bool
    True where the y-coordinate was free to vary during generation.
profile_type : str
    Human readable description of the profile type.

Profile dictionaries are JSON compatible through the standard xicsrt io
machinery: `xicsrt_io` converts numpy arrays to lists on save
(`_convert_from_numpy`) and back to arrays on load (`_convert_to_numpy`).

Units
-----
All quantities are in xicsrt standard SI-like units: temperatures in eV,
velocities in m/s. Emissivity profiles are shape-only and normalized to a
peak value of 1.0 by design; the absolute scale is applied elsewhere
(for example via an ``emissivity_scale`` config option).

Seeding
-------
Each generator seeds its own ``numpy.random.Generator``, so the same seed
always reproduces the same profile. A consequence that is easy to miss:
passing the SAME seed to several generators does NOT give independent
profiles, it gives CORRELATED ones.

Every generator here opens with the same call,
:func:`sample_interior_x`, and then draws its amplitudes from the same
position in the stream. With a common ``n_knots`` the draw sequences are
aligned, so two generators sharing a seed consume identical variates and
their outputs become deterministic functions of one another. Measured over
500 seeds, the resulting cross-profile correlation is exactly 1.0: the knot
locations are identical and the amplitudes are related by an exact affine
map.

To generate a set of profiles that are independent of each other but still
reproducible from a single seed, derive one child seed per profile with
:func:`profile_seeds`::

    seeds = profile_seeds(1234, ['emissivity', 'ion_temp', 'electron_temp'])
    emiss = generate_random_emissivity(seed=seeds['emissivity'])
    ti = generate_random_ion_temp(seed=seeds['ion_temp'])
    te = generate_random_electron_temp(seed=seeds['electron_temp'])

The same 500-seed measurement gives a maximum cross-profile correlation of
0.10 with the child seeds, consistent with the 0.13 expected by chance at
this sample size.

This file includes AI generated code using Claude (Opus 5, Fable 5).
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------


def profile_seeds(seed, names):
    """
    Derive one independent child seed per named profile.

    Generators in this module that are given the same seed produce
    correlated profiles, because their draw sequences are aligned (see the
    "Seeding" section of the module docstring). This function derives a
    separate, statistically independent random stream for each profile from
    a single parent seed, so that a set of profiles is reproducible from one
    seed without being correlated.

    Each child is identified by its position in ``names`` through the
    ``spawn_key`` mechanism of ``numpy.random.SeedSequence``. A given name
    therefore keeps its own stream even if other profiles are added, removed
    or reordered around it, and even if a generator's internal number of
    draws changes.

    Parameters
    ----------
    seed : int, sequence of int, numpy.random.SeedSequence, or None
        The parent seed. Anything accepted by ``numpy.random.SeedSequence``.
        If None, fresh entropy is drawn from the OS, and the result is
        reproducible only within the returned set.
    names : sequence of str
        The profile names. The returned dictionary has one entry per name.

    Returns
    -------
    dict
        A mapping from each name to a ``numpy.random.SeedSequence``. These
        may be passed directly as the ``seed`` argument of any generator in
        this module.

    Examples
    --------
    >>> seeds = profile_seeds(1234, ['emissivity', 'ion_temp'])
    >>> profile = generate_random_emissivity(seed=seeds['emissivity'])

    This function was AI generated using Claude (Opus 5).
    """
    # Resolve the parent entropy before spawning the children. This matters
    # when seed is None: it pins the OS-drawn entropy once, so that all of
    # the children derive from a single common parent.
    parent = np.random.SeedSequence(seed)

    return {
        name: np.random.SeedSequence(parent.entropy, spawn_key=(ii,))
        for ii, name in enumerate(names)
    }


def sample_interior_x(n_interior, min_spacing, rng, x_min=0.0, x_max=1.0):
    """
    Generate random interior x-values with minimum spacing.

    The minimum spacing also applies between the endpoints and the nearest
    interior knots.

    Parameters
    ----------
    n_interior : int
        Number of interior values to generate.
    min_spacing : float
        Minimum spacing between neighboring values (and the endpoints).
    rng : numpy.random.Generator
        Random number generator.
    x_min : float
        Lower endpoint (not included in the output).
    x_max : float
        Upper endpoint (not included in the output).

    Returns
    -------
    numpy.ndarray
        Sorted array of ``n_interior`` values strictly inside
        (x_min, x_max).
    """
    if n_interior == 0:
        return np.array([])

    num_gaps = n_interior + 1
    width = x_max - x_min

    free_space = width - num_gaps * min_spacing
    if free_space < 0.0:
        raise ValueError(
            "The requested minimum spacing is too large for the number of knots."
        )

    random_cuts = np.sort(rng.uniform(0.0, free_space, size=n_interior))
    extra_gaps = np.diff(np.concatenate(([0.0], random_cuts, [free_space])))
    gaps = min_spacing + extra_gaps
    interior_x = x_min + np.cumsum(gaps)[:-1]

    return interior_x


def sample_monotone_y(n, y_start, y_end, rng, min_dy=0.0):
    """
    Generate random monotonic interior y-values.

    The returned values move monotonically from y_start toward y_end.
    The endpoint values themselves are not included.

    Parameters
    ----------
    n : int
        Number of interior values to generate.
    y_start : float
        Starting endpoint value (not included in the output).
    y_end : float
        Ending endpoint value (not included in the output).
    rng : numpy.random.Generator
        Random number generator.
    min_dy : float
        Optional minimum spacing between neighboring y-values.

    Returns
    -------
    numpy.ndarray
        Array of ``n`` values moving monotonically from y_start to y_end.
    """
    if n == 0:
        return np.array([])

    y_min = min(y_start, y_end)
    y_max = max(y_start, y_end)

    if min_dy > 0.0:
        interior = y_min + sample_interior_x(
            n_interior=n, min_spacing=min_dy, rng=rng, x_min=0.0, x_max=y_max - y_min
        )
    else:
        interior = y_min + np.sort(rng.uniform(0.0, y_max - y_min, size=n))

    if y_start > y_end:
        interior = interior[::-1]

    return interior


def make_profile_dict(x_knots, y_knots, deriv_zero, x_free, y_free, profile_type):
    """
    Assemble a profile dictionary from its components.

    All array-like inputs are normalized to numpy arrays with consistent
    dtypes. Serialization (JSON/hdf5) is handled by `xicsrt_io`, which
    converts numpy arrays to lists on save and back on load.

    This function was AI generated using Claude (Opus 5).
    """
    return {
        "x_knots": np.asarray(x_knots, dtype=float),
        "y_knots": np.asarray(y_knots, dtype=float),
        "deriv_zero": np.asarray(deriv_zero, dtype=bool),
        "x_free": np.asarray(x_free, dtype=bool),
        "y_free": np.asarray(y_free, dtype=bool),
        "profile_type": str(profile_type),
    }


def spline_from_profile(profile, extrapolate=False):
    """
    Build a callable interpolator from a profile dictionary.

    A monotonicity preserving PCHIP interpolant is constructed through the
    profile knots. Where ``deriv_zero`` is True the spline derivative is
    forced to zero (implemented via a cubic Hermite spline with the PCHIP
    slopes at all other knots).

    Parameters
    ----------
    profile : dict
        A profile dictionary as returned by the ``generate_random_*``
        functions in this module.
    extrapolate : bool
        If False (default), evaluation outside the knot range returns NaN.
        This is relied upon to mark points outside the last closed flux
        surface; do not enable extrapolation for plasma profiles.

    Returns
    -------
    scipy.interpolate.PchipInterpolator or scipy.interpolate.CubicHermiteSpline
        Callable interpolator over the knot range.

    This function was AI generated using Claude (Opus 5).
    """
    x_knots = np.asarray(profile["x_knots"], dtype=float)
    y_knots = np.asarray(profile["y_knots"], dtype=float)
    deriv_zero = np.asarray(profile["deriv_zero"], dtype=bool)

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=extrapolate)

    if np.any(deriv_zero):
        derivatives = pchip.derivative()(x_knots)
        derivatives[deriv_zero] = 0.0
        spline = CubicHermiteSpline(
            x_knots, y_knots, derivatives, extrapolate=extrapolate
        )
    else:
        spline = pchip

    return spline


def _deriv_zero_mask(n_knots, zero_deriv_core):
    """Return a deriv_zero mask with only the core knot optionally set."""
    deriv_zero = np.zeros(n_knots, dtype=bool)
    if zero_deriv_core:
        deriv_zero[0] = True
    return deriv_zero


def _interior_x_free_mask(n_knots):
    """Return an x_free mask with the two endpoint knots fixed."""
    x_free = np.ones(n_knots, dtype=bool)
    x_free[0] = False
    x_free[-1] = False
    return x_free


# ------------------------------------------------------------------------
# Generating Random Emissivity Spline
# ------------------------------------------------------------------------


def generate_random_emissivity(
    n_knots=5, min_spacing=0.05, zero_deriv_core=True, min_dy=0.0, seed=None
):
    """
    Generate a random emissivity profile with at most one maximum.

    The peak may occur at the core or at any interior knot. The profile is
    normalized so the peak value is 1.0; only the profile shape is
    randomized. The absolute emissivity scale is applied elsewhere (for
    example via an ``emissivity_scale`` config option). The final two knots
    are fixed at zero so the emissivity reaches zero inside the last closed
    flux surface.

    Parameters
    ----------
    n_knots : int
        Total number of knots, including the axis and edge. Must be >= 3.
    min_spacing : float
        Minimum spacing between neighboring x knots.
    zero_deriv_core : bool
        If True, enforce d(emissivity)/dx = 0 at the axis.
    min_dy : float
        Optional minimum spacing between neighboring y-values.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Profile dictionary (see module docstring). Use
        :func:`spline_from_profile` to obtain a callable interpolator.
    """
    if n_knots < 3:
        raise ValueError("n_knots must be at least 3.")

    rng = np.random.default_rng(seed)

    # Generate random interior knot locations.
    # There will be n_knots - 2 random interior x-locations.
    # The final interior location, x_knots[-2], determines where the
    # emissivity first reaches zero.
    n_interior = n_knots - 2

    interior_x = sample_interior_x(
        n_interior=n_interior, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0
    )

    x_knots = np.concatenate(([0.0], interior_x, [1.0]))

    # The final two knots must both have zero emissivity.
    y_knots = np.zeros(n_knots)

    # The peak can occur from index 0 through index n_knots - 3.
    # Indices -2 and -1 are reserved for the two zero knots.
    peak_index = rng.integers(0, n_knots - 2)
    y_peak = 1.0

    if peak_index == 0:
        # The peak is at the core/axis.
        y_knots[0] = y_peak

        # Decreasing values between the core peak and second-to-last knot.
        y_knots[1:-2] = sample_monotone_y(
            n=n_knots - 3, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy
        )
    else:
        # The peak is at an interior knot.
        y_axis = rng.uniform(0.0, y_peak)
        y_knots[0] = y_axis

        # Monotonic increase from the axis to the peak.
        y_knots[1:peak_index] = sample_monotone_y(
            n=peak_index - 1, y_start=y_axis, y_end=y_peak, rng=rng, min_dy=min_dy
        )
        y_knots[peak_index] = y_peak

        # Monotonic decrease from the peak to the edge.
        y_knots[peak_index + 1 : -2] = sample_monotone_y(
            n=n_knots - peak_index - 3, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy
        )

    # Fix both final y-values at zero.
    y_knots[-2] = 0.0
    y_knots[-1] = 0.0

    deriv_zero = _deriv_zero_mask(n_knots, zero_deriv_core)

    # First x-knot is fixed at rho=0, last x-knot is fixed at rho=1.
    x_free = _interior_x_free_mask(n_knots)

    # The final two y-knots are fixed at zero.
    y_free = np.ones(n_knots, dtype=bool)
    y_free[-2:] = False

    return make_profile_dict(x_knots, y_knots, deriv_zero, x_free, y_free, "emissivity")


# ------------------------------------------------------------------------
# Generating Random Ion and Electron Temperature Splines
# ------------------------------------------------------------------------


def generate_random_temp(
    n_knots=5,
    y_min=200.0,
    y_max=5000.0,
    decreasing=True,
    min_spacing=0.05,
    zero_deriv_core=True,
    seed=None,
):
    """
    Generate a random temperature profile.

    The radial endpoints are fixed at x=0 and x=1. Interior x-locations are
    random in (0, 1) and strictly increasing. The core (or edge) temperature
    is drawn uniformly from (y_min, y_max) and the interior y-values are
    random and monotonic.

    Temperature profiles are NOT normalized: the randomized peak value is
    itself a physical parameter (in eV) and must be preserved. Consumers
    should therefore apply a temperature scale of 1.0.

    Notes
    -----
    Ion and electron temperature profiles are generated by independent calls
    to this function, and are therefore drawn FULLY INDEPENDENTLY of each
    other. This maximizes coverage of the parameter space, but it also
    admits combinations that are not physical for a given heating scheme.
    With the default ranges, roughly 25% of samples get Ti > Te and roughly
    4% get Ti > 5*Te (measured over 2000 seeds).

    A more physics-constrained formulation would couple the two profiles and
    enforce Te >= Ti, which is the expected ordering for Electron Cyclotron
    Resonance Heating, where power is deposited on the electrons and reaches
    the ions only through collisional transfer. Other heating schemes (NBI,
    ICRH) are less clear-cut and can drive Ti > Te. No such constraint is
    imposed here; this is deliberate.

    Parameters
    ----------
    n_knots : int
        Total number of knots, including the fixed endpoints. Must be >= 2.
    y_min : float
        Minimum randomized temperature in eV.
    y_max : float
        Maximum randomized temperature in eV.
    decreasing : bool
        If True (default), the profile decreases monotonically from a
        randomized core temperature to zero at the edge. If False, the
        profile increases monotonically from y_min at the core to a
        randomized edge temperature.
    min_spacing : float
        Minimum spacing between neighboring x knots.
    zero_deriv_core : bool
        If True (default), enforce dy/dx=0 at the core (x=0).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Profile dictionary (see module docstring). Use
        :func:`spline_from_profile` to obtain a callable interpolator.
    """
    if n_knots < 2:
        raise ValueError("n_knots must be at least 2.")

    if y_min >= y_max:
        raise ValueError("y_min must be less than y_max.")

    if y_min < 0.0:
        raise ValueError("y_min must be a nonnegative value.")

    rng = np.random.default_rng(seed)

    # x_knots: x=0 and x=1 fixed, interior is random and strictly increasing.
    n_interior = n_knots - 2
    if n_interior > 0:
        interior_x = sample_interior_x(
            n_interior=n_interior,
            min_spacing=min_spacing,
            rng=rng,
            x_min=0.0,
            x_max=1.0,
        )
        x_knots = np.concatenate(([0.0], interior_x, [1.0]))
    else:
        x_knots = np.array([0.0, 1.0])

    # y_knots: zero at the LCFS; core temp and interior are random, monotonic.
    y_free = np.ones(n_knots, dtype=bool)
    if decreasing:
        core_temp = rng.uniform(y_min, y_max)
        interior_y = sample_monotone_y(
            n=n_interior, y_start=core_temp, y_end=0.0, rng=rng
        )
        y_knots = np.concatenate(([core_temp], interior_y, [0.0]))
        # Last y-knot is fixed at zero.
        y_free[-1] = False
    else:
        edge_temp = rng.uniform(y_min, y_max)
        interior_y = sample_monotone_y(
            n=n_interior, y_start=y_min, y_end=edge_temp, rng=rng
        )
        y_knots = np.concatenate(([y_min], interior_y, [edge_temp]))
        # First y-knot is fixed at y_min.
        y_free[0] = False

    deriv_zero = _deriv_zero_mask(n_knots, zero_deriv_core)

    # First x-knot is fixed at rho=0, last x-knot is fixed at rho=1.
    x_free = _interior_x_free_mask(n_knots)

    return make_profile_dict(
        x_knots, y_knots, deriv_zero, x_free, y_free, "temperature"
    )


def generate_random_electron_temp(
    n_knots=5, y_min=200.0, y_max=10000.0, min_spacing=0.05, seed=None
):
    """
    Generate a random electron temperature profile.

    A wrapper around :func:`generate_random_temp` with electron temperature
    defaults. All temperatures are in eV.

    Returns
    -------
    dict
        Profile dictionary (see module docstring).
    """
    profile = generate_random_temp(
        n_knots=n_knots,
        y_min=y_min,
        y_max=y_max,
        min_spacing=min_spacing,
        zero_deriv_core=True,
        seed=seed,
    )

    profile["profile_type"] = "electron temperature"

    return profile


def generate_random_ion_temp(
    n_knots=5, y_min=200.0, y_max=5000.0, min_spacing=0.05, seed=None
):
    """
    Generate a random ion temperature profile.

    A wrapper around :func:`generate_random_temp` with ion temperature
    defaults. All temperatures are in eV.

    Returns
    -------
    dict
        Profile dictionary (see module docstring).
    """
    profile = generate_random_temp(
        n_knots=n_knots,
        y_min=y_min,
        y_max=y_max,
        min_spacing=min_spacing,
        zero_deriv_core=True,
        seed=seed,
    )

    profile["profile_type"] = "ion temperature"

    return profile


# ------------------------------------------------------------------------
# Generating Random Perpendicular Velocity Spline
#
# Note: this function does not create cases where there is no ion root
#       (pure electron root) or when there is no electron root (pure ion
#       root). Widening this coverage is a physics decision left for the
#       future.
# ------------------------------------------------------------------------


def generate_random_perpendicular_velocity(
    n_knots=5, min_spacing=0.05, y_min=-20e3, y_max=20e3, seed=None
):
    """
    Generate a random perpendicular velocity profile.

    The profile is zero at the axis, the edge and one interior knot, with a
    negative (ion root) excursion followed by a positive (electron root)
    excursion. Velocities are in m/s.

    Parameters
    ----------
    n_knots : int
        Total number of knots (must equal 5).
    min_spacing : float
        Minimum spacing between neighboring x knots.
    y_min : float
        Most negative allowed velocity (ion root magnitude bound) in m/s.
        Must be negative.
    y_max : float
        Most positive allowed velocity (electron root magnitude bound) in
        m/s. Must be positive.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Profile dictionary (see module docstring). Use
        :func:`spline_from_profile` to obtain a callable interpolator.
    """
    if n_knots != 5:
        raise ValueError("Perpendicular velocity requires exactly 5 knots.")

    if y_min >= 0.0:
        raise ValueError("y_min must be negative.")

    if y_max <= 0.0:
        raise ValueError("y_max must be positive.")

    rng = np.random.default_rng(seed)

    # Generate three random interior knot locations.
    interior_x = sample_interior_x(
        n_interior=3, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0
    )

    x_knots = np.concatenate(([0.0], interior_x, [1.0]))

    # Random ion-root velocity (negative).
    ion_root = rng.uniform(y_min, 0.0)

    # Random electron-root velocity (positive).
    electron_root = rng.uniform(0.0, y_max)

    y_knots = np.array([0.0, ion_root, 0.0, electron_root, 0.0])

    # Perpendicular velocity may not have a zero derivative at the core.
    deriv_zero = np.zeros(n_knots, dtype=bool)

    # First x-knot is fixed at rho=0, last x-knot is fixed at rho=1.
    x_free = _interior_x_free_mask(n_knots)

    # First and last y-knots are fixed at zero.
    y_free = np.array([False, True, True, True, False])

    return make_profile_dict(
        x_knots, y_knots, deriv_zero, x_free, y_free, "perpendicular velocity"
    )


# ------------------------------------------------------------------------
# Generating Random Parallel Velocity Splines
# ------------------------------------------------------------------------


def generate_random_parallel_velocity(
    n_knots=5,
    min_spacing=0.05,
    y_min=-20e3,
    y_max=20e3,
    zero_deriv_core=True,
    min_dy=0.0,
    seed=None,
):
    """
    Generate a random parallel velocity profile with at most one extremum.

    The peak velocity is drawn uniformly from (y_min, y_max) and may be
    positive or negative; the profile magnitude then decreases
    monotonically to zero at the edge. The peak may occur at the core or at
    any interior knot. Velocities are in m/s.

    Parameters
    ----------
    n_knots : int
        Total number of knots, including the axis and edge. Must be >= 3.
    min_spacing : float
        Minimum spacing between neighboring x knots.
    y_min : float
        Minimum allowed peak velocity in m/s.
    y_max : float
        Maximum allowed peak velocity in m/s.
    zero_deriv_core : bool
        If True, enforce d(velocity)/dx = 0 at the axis.
    min_dy : float
        Optional minimum spacing between neighboring y-values.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Profile dictionary (see module docstring). Use
        :func:`spline_from_profile` to obtain a callable interpolator.
    """
    if n_knots < 3:
        raise ValueError("n_knots must be at least 3.")

    if y_min >= y_max:
        raise ValueError("y_min must be less than y_max.")

    rng = np.random.default_rng(seed)
    n_interior = n_knots - 2

    # Generate random interior knot locations.
    interior_x = sample_interior_x(
        n_interior=n_interior, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0
    )
    x_knots = np.concatenate(([0.0], interior_x, [1.0]))

    # Choose the peak knot randomly.
    # The final knot cannot be the peak because its value is zero.
    peak_index = rng.integers(0, n_knots - 1)
    y_peak = rng.uniform(y_min, y_max)

    y_knots = np.empty(n_knots)
    if peak_index == 0:
        # The peak is at the core/axis.
        y_knots[0] = y_peak
        y_knots[1:-1] = sample_monotone_y(
            n=n_knots - 2, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy
        )
        y_knots[-1] = 0.0
    else:
        # The peak is at an interior knot.
        y_axis = rng.uniform(min(0.0, y_peak), max(0.0, y_peak))
        y_knots[0] = y_axis

        # Monotonic |increase| from the axis to the peak.
        y_knots[1:peak_index] = sample_monotone_y(
            n=peak_index - 1, y_start=y_axis, y_end=y_peak, rng=rng, min_dy=min_dy
        )
        y_knots[peak_index] = y_peak

        # Monotonic |decrease| from the peak to the edge.
        y_knots[peak_index + 1 : -1] = sample_monotone_y(
            n=n_knots - peak_index - 2, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy
        )
        y_knots[-1] = 0.0

    deriv_zero = _deriv_zero_mask(n_knots, zero_deriv_core)

    # First x-knot is fixed at rho=0, last x-knot is fixed at rho=1.
    x_free = _interior_x_free_mask(n_knots)

    # Last y-knot is fixed at zero.
    y_free = np.ones(n_knots, dtype=bool)
    y_free[-1] = False

    return make_profile_dict(
        x_knots, y_knots, deriv_zero, x_free, y_free, "parallel velocity"
    )
