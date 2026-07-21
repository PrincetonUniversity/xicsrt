import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
 
def sample_interior_x(n_interior, min_spacing, rng, x_min=0.0, x_max=1.0):
    """
    Generate random interior x-values with minimum spacing.

    The minimum spacing also applies between the endpoints and
    the nearest interior knots.
    """
    if n_interior == 0:
        return np.array([])

    num_gaps = n_interior + 1
    width = x_max - x_min

    free_space = (width - num_gaps * min_spacing)
    if free_space < 0.0:
        raise ValueError("The requested minimum spacing is too large for the number of knots.")

    random_cuts = np.sort(rng.uniform(0.0, free_space, size=n_interior))
    extra_gaps = np.diff(np.concatenate(([0.0], random_cuts, [free_space])))
    gaps = (min_spacing + extra_gaps)
    interior_x = (x_min + np.cumsum(gaps)[:-1])

    return interior_x


def sample_monotone_y(n, y_start, y_end, rng, min_dy=0.0):
    """
    Generate random monotonic interior y-values.

    The returned values move monotonically from y_start toward y_end.
    The endpoint values themselves are not included.
    """
    if n == 0:
        return np.array([])

    y_min = min(y_start, y_end)
    y_max = max(y_start, y_end)

    if min_dy > 0.0:
        interior = y_min + sample_interior_x(n_interior=n, min_spacing=min_dy, rng=rng, x_min=0.0, x_max=y_max - y_min)
    else:
        interior = y_min + np.sort(rng.uniform(0.0, y_max - y_min, size=n))

    if y_start > y_end:
        interior = interior[::-1]

    return interior


def make_profile_dic(x_knots, y_knots, deriv_zero, x_free, y_free, profile_type):
    return {
        "x_knots": x_knots,
        "y_knots": y_knots,
        "deriv_zero": deriv_zero,
        "x_free": x_free,
        "y_free": y_free,
        "profile_type": profile_type,
    }

# -------------------------------------------------------------------
# Generating Random Emissivity Spline
# -------------------------------------------------------------------

def generate_random_emissivity(n_knots=5, min_spacing=0.05, zero_deriv_core=True, min_dy=0.0, seed=None):
    """
    Generate a random emissivity spline with at most one maximum.

    The peak may occur at the core or at any interior knot.

    Parameters
    ----------
    n_knots : int
        Total number of knots, including the axis and edge.
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
    spline : emissivity interpolation function (PchipInterpolator or CubicHermiteSpline)
    x_knots : knot locations
    y_knots : emissivity values at the knots
    zero_deriv : 
    x_free : boolean array where True means the x-coordinate is free to vary
    y_free : boolean array where True means the y-coordinate is free to vary
    profile_type : profile type
    """
    if n_knots < 3:
        raise ValueError("n_knots must be at least 3.")

    rng = np.random.default_rng(seed)
    
    # Generate random interior knot locations.
    # There will be n_knots - 2 random interior x-locations
    # The final interior location, x_knots[-2], determines where the emissivity first reaches zero
    n_interior = n_knots - 2
    
    interior_x = sample_interior_x(n_interior=n_interior, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0,)

    x_knots = np.concatenate(([0.0], interior_x, [1.0]))

    # The final two knots must both have zero emissivity
    y_knots = np.zeros(n_knots)

    # The peak can occur from index 0 through index n_knots - 3
    # Indices -2 and -1 are reserved for the two zero knots 
    peak_index = rng.integers(0, n_knots - 2)
    y_peak = 1.0

    if peak_index == 0:
        # The peak is at the core/axis.
        y_knots[0] = y_peak

        # generating decreasing values between core peak and second-to-last knot
        y_knots[1:-2] = sample_monotone_y(n=n_knots-3, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy)
    else:
        # The peak is at an interior knot.
        y_axis = rng.uniform(0.0, y_peak)
        y_knots[0] = y_axis

        # Monotonic increase from the axis to the peak.
        y_knots[1:peak_index] = sample_monotone_y(n=peak_index - 1, y_start=y_axis, y_end=y_peak, rng=rng, min_dy=min_dy)
        y_knots[peak_index] = y_peak

        # Monotonic decrease from the peak to the edge.
        y_knots[peak_index + 1:-2] = sample_monotone_y(n=n_knots-peak_index-3, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy)

    # Then fix both final y-values at zero
    y_knots[-2] = 0.0
    y_knots[-1] = 0.0

    pchip = PchipInterpolator(x_knots, y_knots)

    if zero_deriv_core:
        derivatives = pchip.derivative()(x_knots)
        derivatives[0] = 0.0
        spline = CubicHermiteSpline(x_knots, y_knots, derivatives)
        deriv_zero = np.array([True, False, False, False, False])
    else:
        spline = pchip
        deriv_zero = np.array([False, False, False, False, False])

    # first x-knot is fixed at rho=0
    # last x-knot is fixed at rho=1
    x_free = np.array([False, True, True, True, False])

    # second-to-last y-knot is fixed at zero
    # last y-knot is fixed at zero
    y_free = np.array([True, True, True, False, False])
    
    return make_profile_dic(x_knots, y_knots, deriv_zero, x_free, y_free, "emissivity")

# ---------------------------------------------------------------
# Generating Random Ion and Electron Temperature Splines
# ---------------------------------------------------------------

def generate_random_temp(
    n_knots=6, 
    y_min=0.2,       # minimum y-value is 0.2 keV for electron and ion temp
    y_max=5.0,       # maximum y-value is 10.0 keV for electron and 5.0 keV for ion temp
    decreasing=True, 
    min_spacing=0.05, 
    zero_deriv_core=True,  
    seed=None
): 
    """
    Generates a random PCHIP spline for ion and electron temp profiles. 

    The radial endpoints are fixed at x=0 and x=1.
    Interior x-locations are random in (0, 1) and strictly increasing.
    Interior y-locations are random in (y_min, y_max) and monotonic.

    Parameters
    ----------
    n_knots : total number of knots, including the fixed endpoints
    y_min : minimum temp in keV
    y_max : maximum allowed temp in keV
    decreasing : if true (default), then y goes monotonically from 1 -> 0
    zero_deriv_core : if true (default), then enforce dy/dx=0 at core (x=0)
    seed : random seed for reproducibility

    Returns
    -------
    spline : callable interpolator (PchipInterpolator or CubicHermiteSpline)
    x_knots :  
    y_knots : 
    zero_deriv : 
    x_free : boolean array where True means the x-coordinate is free to vary
    y_free : boolean array where True means the y-coordinate is free to vary
    profile_type :
    """

    if n_knots < 2: 
        raise ValueError("n_knots must be at least 2.")

    if y_min >= y_max: 
        raise ValueError("y_min must be less than y_max.")

    if y_min < 0.0: 
        raise ValueError("y_min must be nonnegative value.")

    rng = np.random.default_rng(seed)

    # x_knots : x=0 and x=1 fixed, interior is random and strictly increasing
    n_interior = n_knots - 2
    if n_interior > 0:
        interior_x = sample_interior_x(n_interior=n_interior, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0)
        x_knots = np.concatenate(([0.0], interior_x, [1.0]))
    else:
        x_knots = np.array([0.0, 1.0])

    # y_knots : y=0 at LCFS; core temp and interior is random, and monotonic
    if decreasing: 
        core_temp = rng.uniform(y_min, y_max)
        interior_y = sample_monotone_y(n=n_interior, y_start=core_temp, y_end=0.0, rng=rng)
        y_knots = np.concatenate(([core_temp], interior_y, [0.0]))
    else: 
        edge_temp = rng.uniform(y_min, y_max)
        interior_y = sample_monotone_y(n=n_interior, y_start=y_min, y_end=edge_temp, rng=rng)
        y_knots = np.concatenate(([y_min], interior_y, [edge_temp]))

    # Building the spline
    pchip = PchipInterpolator(x_knots, y_knots)
    if zero_deriv_core:
        # take PCHIP's monotone-preserving slopes, then zero out the one at x=0
        dydx = pchip.derivative()(x_knots)
        dydx[0] = 0.0
        spline = CubicHermiteSpline(x_knots, y_knots, dydx)
        deriv_zero = np.array([True, False, False, False, False])
    else:
        spline = pchip
        deriv_zero = np.array([False, False, False, False, False])

    # first x-knot fixed at rho=0
    # last x-knot fixed at rho=1
    x_free = np.array([False, True, True, True, False])

    # last y-knot fixed at zero
    y_free = np.array([True, True, True, True, False])

    return make_profile_dic(x_knots, y_knots, deriv_zero, x_free, y_free, "temperature")


# creating a wrapper to create electron temp spline
def generate_random_electron_temp(n_knots=5, min_spacing=0.05, seed=None):
    """
    Generate a random electron temperature profile.
    """

    profile = generate_random_temp(
        n_knots=n_knots,
        y_min=0.2,
        y_max=10.0,
        min_spacing=min_spacing,
        zero_deriv_core=True,
        seed=seed,
    )

    profile["profile_type"] = "electron temperature"

    return profile


# creating a wrapper to create ion temp spline
def generate_random_ion_temp(n_knots=5, min_spacing=0.05, seed=None):
    """
    Generate a random ion temperature profile.
    """
    
    profile = generate_random_temp(
        n_knots=n_knots,
        y_min=0.2,
        y_max=5.0,
        min_spacing=min_spacing,
        zero_deriv_core=True,
        seed=seed,
    )

    profile["profile_type"] = "ion temperature"

    return profile

# -------------------------------------------------------------------
# Generating Random Perpendicular Velocity Spline
# -------------------------------------------------------------------

def generate_random_perpendicular_velocity(n_knots=5, min_spacing=0.05, seed=None):
    """
    Generate a random perpendicular velocity profile. 

    Parameters
    ----------
    n_knots : total number of knots (must equal 5)
    min_spacing : minimum spacing between neighboring knots
    seed : random seed for reproducibility

    Returns
    -------
    spline : perpendicular velocity interpolation function (PchipInterpolator)
    x_knots : knot locations
    y_knots : perpendicular velocity values at the knots
    zero_deriv : 
    x_free : boolean array where True means the x-coordinate is free to vary
    y_free : boolean array where True means the y-coordinate is free to vary
    """
    if n_knots != 5: 
        raise ValueError("Perpendicular velocity requires exactly 5 knots.")

    rng = np.random.default_rng(seed)

    # Generate three random interior knot locations
    interior_x = sample_interior_x(n_interior=3, min_spacing=min_spacing, rng=rng, x_min=0.0, x_max=1.0)

    x_knots = np.concatenate(([0.0], interior_x, [1.0]))

    # Random ion-root magnitude
    ion_root = -rng.uniform(0.0, 20.0)
    
    # Random electron-root magnitude
    electron_root = rng.uniform(0.0, 20.0)

    y_knots = np.array([0.0, ion_root, 0.0, electron_root, 0.0])

    spline = PchipInterpolator(x_knots, y_knots)

    # Perpendicular velocity may not have a zero derivative at the core so we set all to 'False'
    deriv_zero = [False, False, False, False, False]

    # first x-knot is fixed at rho=0
    # last x-knot is fixed at rho=1
    x_free = np.array([False, True, True, True, False])
    
    # first y-knot is fixed at zero
    # last y-knot is fixed at zero
    y_free = np.array([False, True, True, True, False])

    return make_profile_dic(x_knots, y_knots, deriv_zero, x_free, y_free, "perpendicular velocity")

# ---------------------------------------------------------------
# Generating Random Parallel Velocity Splines
# ---------------------------------------------------------------

def generate_random_parallel_velocity(n_knots=5, min_spacing=0.05, zero_deriv_core=True, min_dy=0.0, seed=None): 
    """ 
    Generate a random parallel velocity spline with at most one maximum. The peak may occur at the core or at any interior knot. 
    
    Parameters 
    ---------- 
    n_knots : int Total number of knots, including the axis and edge. 
    min_spacing : float Minimum spacing between neighboring x knots. 
    zero_deriv_core : bool If True, enforce d(emissivity)/dx = 0 at the axis. 
    min_dy : float Optional minimum spacing between neighboring y-values. 
    seed : int or None Random seed for reproducibility. 
    
    Returns 
    ------- 
    x_knots : knot locations 
    y_knots : emissivity values at the knots 
    zero_deriv : 
    x_free : boolean array where True means the x-coordinate is free to vary 
    y_free : boolean array where True means the y-coordinate is free to vary 
    profile_type : 
    """ 
    
    if n_knots < 3: 
        raise ValueError("n_knots must be at least 3.") 
        
    rng = np.random.default_rng(seed) 
    n_interior = n_knots - 2 
    
    # Generate random interior knot locations. 
    interior_x = sample_interior_x(n_interior=n_interior, min_spacing=min_spacing, rng=rng, ) 
    x_knots = np.concatenate(([0.0], interior_x, [1.0]))
    
    # Choose the peak knot randomly. 
    # The final knot cannot be the peak because its value is zero. 
    peak_index = rng.integers(0, n_knots - 1) 
    y_peak = rng.uniform(0.0, 20.0)
    
    if peak_index == 0: 
        # The peak is at the core/axis. 
        y_knots = np.empty(n_knots) 
        y_knots[0] = y_peak 
        y_knots[1:-1] = sample_monotone_y(n=n_knots-2, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy) 
        y_knots[-1] = 0.0 
    else: 
        # The peak is at an interior knot. 
        y_axis = rng.uniform(0.0, y_peak) 
        y_knots = np.empty(n_knots) 
        y_knots[0] = y_axis 
        
        # Monotonic increase from the axis to the peak. 
        y_knots[1:peak_index] = sample_monotone_y(n=peak_index-1, y_start=y_axis, y_end=y_peak, rng=rng, min_dy=min_dy) 
        y_knots[peak_index] = y_peak 
    
        # Monotonic decrease from the peak to the edge. 
        y_knots[peak_index + 1:-1] = sample_monotone_y(n=n_knots-peak_index-2, y_start=y_peak, y_end=0.0, rng=rng, min_dy=min_dy) 
        y_knots[-1] = 0.0 
    
    pchip = PchipInterpolator(x_knots, y_knots) 
    
    if zero_deriv_core: 
        derivatives = pchip.derivative()(x_knots) 
        derivatives[0] = 0.0 
        spline = CubicHermiteSpline(x_knots, y_knots, derivatives) 
        deriv_zero = np.array([True, False, False, False, False]) 
    else: 
        spline = pchip 
        deriv_zero = np.array([False, False, False, False, False]) 

    # first x-knot is fixed at rho=0
    # last x-knot is fixed at rho=1
    x_free = np.array([False, True, True, True, False]) 
    
    # last y-knot is fixed at zero
    y_free = np.array([True, True, True, True, False])   
    
    return make_profile_dic(x_knots, y_knots, deriv_zero, x_free, y_free, "parallel velocity")