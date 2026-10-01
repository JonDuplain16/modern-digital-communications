"""commlib.satellite -- orbital geometry and link-budget tools for satellite links.

Used by Chapter 23 ("Satellite Communications") figures and by the satellite lab.
Import explicitly:  ``from commlib import satellite as sat``.

Conventions
-----------
* SI units everywhere (metres, seconds, hertz, kelvin) unless a name says otherwise
  (``_km``, ``_deg``, ``_db``, ``_dbw``, ``_dbhz`` ...).
* Angles in function arguments are in degrees unless the name ends in ``_rad``.
* The Earth is a sphere of radius ``R_E`` (equatorial) for geometry; Earth
  oblateness (J2) is ignored except where stated.
* Everything is vectorised with NumPy where it makes sense.

The rain-attenuation routines implement the *structure* of the ITU-R P.618 slant-path
method with a coarse table of ITU-R P.838-3 coefficients. They are good for teaching and
first-cut designs; use the full Recommendations (and P.837/P.839 maps) for real work.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ constants
C = 299_792_458.0            # speed of light, m/s
MU_E = 3.986004418e14        # Earth's gravitational parameter GM, m^3/s^2
R_E = 6_378_137.0            # Earth equatorial radius, m (WGS-84)
OMEGA_E = 7.2921159e-5       # Earth rotation rate, rad/s (sidereal)
SIDEREAL_DAY = 2 * np.pi / OMEGA_E   # ~86164.1 s
K_BOLTZ = 1.380649e-23       # J/K
K_BOLTZ_DB = 10 * np.log10(K_BOLTZ)  # -228.6 dBW/K/Hz
AU = 1.495978707e11          # astronomical unit, m
T0 = 290.0                   # reference temperature, K


# ================================================================== orbits
def orbital_period(a):
    """Period (s) of an orbit with semi-major axis ``a`` (m): T = 2 pi sqrt(a^3/mu)."""
    return 2 * np.pi * np.sqrt(np.asarray(a, float) ** 3 / MU_E)


def semi_major_axis(period):
    """Semi-major axis (m) giving orbital period ``period`` (s) (Kepler's third law)."""
    return (MU_E * (np.asarray(period, float) / (2 * np.pi)) ** 2) ** (1 / 3)


def circular_speed(h):
    """Speed (m/s) of a circular orbit at altitude ``h`` (m): v = sqrt(mu/r)."""
    return np.sqrt(MU_E / (R_E + np.asarray(h, float)))


def geo_radius():
    """Radius (m) of the geostationary orbit (period = one sidereal day), ~42 164 km."""
    return float(semi_major_axis(SIDEREAL_DAY))


GEO_ALT = geo_radius() - R_E  # ~35 786 km


def kepler_E(M, e, tol=1e-12, maxiter=50):
    """Solve Kepler's equation M = E - e sin E for the eccentric anomaly E (rad), Newton's method."""
    M = np.asarray(M, float)
    E = np.where(e > 0.8, np.pi * np.ones_like(M), M)
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def orbit_eci(a, e, inc_deg, raan_deg, argp_deg, M0_deg, t):
    """ECI position (3, N) in m of a Keplerian orbit at times ``t`` (s) (two-body, no J2).

    a: semi-major axis (m); e: eccentricity; inc/raan/argp: inclination, right ascension of
    the ascending node, argument of perigee (deg); M0: mean anomaly at t=0 (deg).
    """
    t = np.asarray(t, float)
    n = np.sqrt(MU_E / a ** 3)
    M = np.radians(M0_deg) + n * t
    E = kepler_E(np.mod(M, 2 * np.pi), e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    xp, yp = r * np.cos(nu), r * np.sin(nu)
    i, O, w = np.radians([inc_deg, raan_deg, argp_deg])
    cO, sO, ci, si, cw, sw = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    x = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
    y = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.vstack([x, y, z])


def ground_track(a, e, inc_deg, raan_deg, argp_deg, M0_deg, t, lon0_deg=0.0):
    """Sub-satellite latitude and longitude (deg) versus time, accounting for Earth rotation.

    Returns (lat_deg, lon_deg) with longitude wrapped to [-180, 180). ``lon0_deg`` rotates
    the whole track (the Greenwich angle at t=0).
    """
    x, y, z = orbit_eci(a, e, inc_deg, raan_deg, argp_deg, M0_deg, t)
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x) - OMEGA_E * np.asarray(t, float)) + lon0_deg
    lon = (lon + 180) % 360 - 180
    return lat, lon


# ================================================================== geometry
def slant_range(h, elev_deg, re=R_E):
    """Slant range (m) from a ground station to a satellite at altitude ``h`` seen at elevation.

    From the Earth-centre / station / satellite triangle (law of cosines):
    d = sqrt((re+h)^2 - re^2 cos^2(el)) - re sin(el).
    """
    el = np.radians(elev_deg)
    r = re + np.asarray(h, float)
    return np.sqrt(r ** 2 - (re * np.cos(el)) ** 2) - re * np.sin(el)


def central_angle(h, elev_deg, re=R_E):
    """Earth central angle (rad) between the station and the sub-satellite point.

    gamma = arccos(re cos(el) / (re+h)) - el.
    """
    el = np.radians(elev_deg)
    return np.arccos(re * np.cos(el) / (re + np.asarray(h, float))) - el


def nadir_angle(h, elev_deg, re=R_E):
    """Off-nadir angle (deg) at the satellite: sin(eta) = re cos(el) / (re+h)."""
    el = np.radians(elev_deg)
    return np.degrees(np.arcsin(re * np.cos(el) / (re + np.asarray(h, float))))


def elevation_from_central_angle(h, gamma_rad, re=R_E):
    """Elevation (deg) of a satellite at altitude ``h`` whose sub-point is ``gamma`` rad away.

    tan(el) = (cos(gamma) - re/r) / sin(gamma). Negative values mean below the horizon.
    """
    g = np.asarray(gamma_rad, float)
    r = re + np.asarray(h, float)
    return np.degrees(np.arctan2(np.cos(g) - re / r, np.sin(g)))


def coverage_radius(h, elev_min_deg, re=R_E):
    """Ground radius (m, along the surface) of the area that sees the satellite above ``elev_min``."""
    return re * central_angle(h, elev_min_deg, re)


def coverage_fraction(h, elev_min_deg, re=R_E):
    """Fraction of the Earth's surface inside one satellite's coverage cap: (1 - cos gamma)/2."""
    return (1 - np.cos(central_angle(h, elev_min_deg, re))) / 2


def geo_look_angles(lat_deg, lon_deg, sat_lon_deg, re=R_E):
    """Azimuth (deg from true north), elevation (deg) and slant range (m) to a GEO satellite.

    Spherical-Earth formulas: cos(gamma) = cos(lat) cos(dlon); el from the central angle;
    azimuth from the spherical triangle, with the usual quadrant rules.
    """
    lat = np.radians(np.asarray(lat_deg, float))
    dl = np.radians(np.asarray(sat_lon_deg, float) - np.asarray(lon_deg, float))
    dl = (dl + np.pi) % (2 * np.pi) - np.pi
    r = geo_radius()
    cg = np.cos(lat) * np.cos(dl)
    g = np.arccos(np.clip(cg, -1, 1))
    d = np.sqrt(r ** 2 + re ** 2 - 2 * r * re * cg)
    el = np.degrees(np.arctan2(cg - re / r, np.sqrt(np.maximum(1 - cg ** 2, 1e-30))))
    # azimuth: bearing from station to the sub-satellite point (on the equator)
    az = np.degrees(np.arctan2(np.sin(dl), -np.sin(lat) * np.cos(dl)))
    az = np.where(np.isclose(g, 0), 0.0, az) % 360
    return az, el, d


def propagation_delay(d):
    """One-way free-space propagation delay (s) over distance ``d`` (m)."""
    return np.asarray(d, float) / C


def max_pass_duration(h, elev_min_deg, re=R_E):
    """Duration (s) of an overhead pass above ``elev_min``, ignoring Earth rotation.

    The satellite sweeps a central angle of 2*gamma at angular rate 2*pi/T.
    """
    return 2 * central_angle(h, elev_min_deg, re) / (2 * np.pi) * orbital_period(re + np.asarray(h, float))


def leo_pass(h, f_hz, max_elev_deg=90.0, elev_min_deg=0.0, dt=1.0, re=R_E):
    """Simulate one pass of a circular-orbit satellite over a fixed ground station.

    The station sits off the orbital plane by the central angle that gives the requested
    maximum elevation; Earth rotation is neglected (it changes LEO Doppler by a few %).
    Returns a dict of arrays: t (s, zero at closest approach), elev (deg), range (m),
    range_rate (m/s, positive = receding), doppler (Hz, = -f * range_rate / c),
    doppler_rate (Hz/s), delay (s).
    """
    r = re + h
    w = np.sqrt(MU_E / r ** 3)
    beta = central_angle(h, max_elev_deg, re) if max_elev_deg < 90 else 0.0
    obs = re * np.array([np.cos(beta), 0.0, np.sin(beta)])
    # time span: until the satellite drops below elev_min
    gmax = central_angle(h, elev_min_deg, re)
    # angle along track where total central angle = gmax: cos gmax = cos(beta) cos(phi)
    phimax = np.arccos(np.clip(np.cos(gmax) / np.cos(beta), -1, 1))
    tmax = phimax / w
    t = np.arange(-tmax, tmax + dt / 2, dt)
    ph = w * t
    s = r * np.vstack([np.cos(ph), np.sin(ph), np.zeros_like(ph)])
    v = r * w * np.vstack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)])
    rel = s - obs[:, None]
    rng_ = np.linalg.norm(rel, axis=0)
    rr = np.sum(rel * v, axis=0) / rng_
    up = obs / re
    elev = np.degrees(np.arcsin(np.clip(np.sum(rel * up[:, None], axis=0) / rng_, -1, 1)))
    dop = -f_hz * rr / C
    # analytic range acceleration: (|v|^2 + rel.a - rr^2)/range, a = -w^2 s
    acc = -w ** 2 * s
    rdd = (np.sum(v * v, axis=0) + np.sum(rel * acc, axis=0) - rr ** 2) / rng_
    return dict(t=t, elev=elev, range=rng_, range_rate=rr, doppler=dop,
                doppler_rate=-f_hz * rdd / C, delay=rng_ / C)


def max_doppler(h, f_hz, re=R_E):
    """Peak Doppler shift (Hz) for an overhead pass, seen at the horizon: f v re / (c r)."""
    r = re + np.asarray(h, float)
    return f_hz * circular_speed(h) * re / (C * r)


def max_doppler_rate(h, f_hz, re=R_E):
    """Peak Doppler rate (Hz/s, magnitude) at zenith of an overhead pass: f v^2 re / (c r h)."""
    h = np.asarray(h, float)
    r = re + h
    return f_hz * circular_speed(h) ** 2 * re / (C * r * h)


def eclipse_fraction(h, beta_deg=0.0, re=R_E):
    """Fraction of a circular orbit spent in the Earth's (cylindrical) shadow.

    ``beta`` is the angle between the orbit plane and the Sun direction.
    f = arccos( sqrt(h^2 + 2 re h) / (r cos beta) ) / pi, and 0 when the argument exceeds 1.
    """
    h = np.asarray(h, float)
    r = re + h
    arg = np.sqrt(h ** 2 + 2 * re * h) / (r * np.cos(np.radians(beta_deg)))
    return np.where(arg < 1, np.arccos(np.clip(arg, -1, 1)) / np.pi, 0.0)


# ================================================================== antennas
def wavelength(f_hz):
    return C / np.asarray(f_hz, float)


def dish_gain_dbi(diam_m, f_hz, eff=0.65):
    """Gain (dBi) of a circular aperture: G = eff * (pi D / lambda)^2."""
    return 10 * np.log10(eff * (np.pi * np.asarray(diam_m, float) / wavelength(f_hz)) ** 2)


def dish_beamwidth_deg(diam_m, f_hz, k=70.0):
    """Approximate 3 dB beamwidth (deg) of a reflector: ~ k lambda / D with k ~ 65-70."""
    return k * wavelength(f_hz) / np.asarray(diam_m, float)


def dish_pattern_db(theta_deg, diam_m, f_hz, taper=1):
    """Normalised far-field pattern (dB) of a circular aperture with a (1-rho^2)^taper illumination.

    taper=0 is uniform (first sidelobe -17.6 dB); taper=1 gives about -24.6 dB sidelobes and
    a wider beam; taper=2 about -30.6 dB.
    """
    from scipy.special import jv, factorial
    u = np.pi * np.asarray(diam_m, float) / wavelength(f_hz) * np.sin(np.radians(theta_deg))
    u = np.where(np.abs(u) < 1e-9, 1e-9, u)
    n = taper + 1
    pat = factorial(n) * 2 ** n * jv(n, u) / u ** n
    return 20 * np.log10(np.abs(pat) + 1e-12)


def itu_s580_mask_dbi(theta_deg):
    """Earth-station sidelobe design objective 29 - 25 log10(theta) dBi (ITU-R S.580), 1<=theta<=20 deg."""
    return 29 - 25 * np.log10(np.asarray(theta_deg, float))


def pointing_loss_db(err_deg, beamwidth_deg):
    """Gaussian-beam approximation of pointing loss: 12 (err/theta_3dB)^2 dB."""
    return 12 * (np.asarray(err_deg, float) / beamwidth_deg) ** 2


def array_scan_loss_db(scan_deg, exponent=1.3):
    """Planar phased-array gain loss when scanned off broadside: -10 log10(cos^x(theta)).

    x=1 is the projected-aperture loss alone; element patterns typically give x ~ 1.2-1.5.
    """
    return -10 * exponent * np.log10(np.cos(np.radians(scan_deg)))


# ================================================================== noise & link
def fspl_db(d, f_hz):
    """Free-space path loss (dB): 20 log10(4 pi d f / c)."""
    return 20 * np.log10(4 * np.pi * np.asarray(d, float) * np.asarray(f_hz, float) / C)


def db(x):
    return 10 * np.log10(np.asarray(x, float))


def undb(x_db):
    return 10 ** (np.asarray(x_db, float) / 10)


def system_noise_temp(t_ant, t_rx, loss_db=0.0, t_phys=T0):
    """System noise temperature (K) referred to the LNA input.

    Antenna temperature ``t_ant`` seen through a lossy feed of ``loss_db`` at physical
    temperature ``t_phys``, followed by a receiver of noise temperature ``t_rx``:
    Tsys = t_ant/L + t_phys (1 - 1/L) + t_rx.
    """
    L = undb(loss_db)
    return np.asarray(t_ant, float) / L + t_phys * (1 - 1 / L) + t_rx


def nf_to_temp(nf_db):
    """Noise figure (dB) -> equivalent noise temperature (K)."""
    return T0 * (undb(nf_db) - 1)


def g_over_t_dbk(g_dbi, tsys_k):
    """Receive figure of merit G/T in dB/K."""
    return np.asarray(g_dbi, float) - db(tsys_k)


def cn0_dbhz(eirp_dbw, path_loss_db, g_over_t, other_loss_db=0.0):
    """Carrier-to-noise-density ratio C/N0 (dB-Hz) = EIRP - L + G/T - k."""
    return eirp_dbw - path_loss_db - other_loss_db + g_over_t - K_BOLTZ_DB


def combine_cn_db(*cn_db):
    """Combine independent noise/interference contributions: 1/(C/N)tot = sum 1/(C/N)_i (dB in, dB out)."""
    terms = np.broadcast_arrays(*[undb(-np.asarray(x, float)) for x in cn_db])
    return -db(np.sum(terms, axis=0))


def rain_noise_temp(atten_db, t_medium=275.0):
    """Sky-noise temperature added by an absorbing medium of loss ``atten_db``: Tm (1 - 10^(-A/10))."""
    return t_medium * (1 - undb(-np.asarray(atten_db, float)))


class LinkBudget:
    """An itemised link budget in dB.

    >>> lb = LinkBudget("Ku DTH")
    >>> lb.add("Satellite EIRP", 52.0, "dBW")
    >>> lb.add("Free-space loss", -205.5)
    >>> lb.total()
    Items are (name, value_db, unit) and are summed in order; ``table()`` prints them.
    """

    def __init__(self, title=""):
        self.title, self.items = title, []

    def add(self, name, value_db, unit="dB"):
        self.items.append((name, float(value_db), unit))
        return self

    def total(self):
        return sum(v for _, v, _ in self.items)

    def running(self):
        return np.cumsum([v for _, v, _ in self.items])

    def table(self):
        lines = [self.title] if self.title else []
        for (n, v, u), s in zip(self.items, self.running()):
            lines.append(f"  {n:<34s}{v:+9.2f} {u:<6s} (running {s:8.2f})")
        return "\n".join(lines)

    def __repr__(self):
        return self.table()


# ================================================================== rain (ITU-R P.838 / P.618)
# (f GHz, kH, alphaH, kV, alphaV) -- selected rows after ITU-R P.838-3; interpolate log-log.
_P838 = np.array([
    [1, 0.0000259, 0.9691, 0.0000308, 0.8592],
    [2, 0.0000847, 1.0664, 0.0000998, 0.9490],
    [4, 0.0001071, 1.6009, 0.0002461, 1.2476],
    [6, 0.0007056, 1.5900, 0.0004878, 1.5728],
    [7, 0.001915, 1.4810, 0.001425, 1.4745],
    [8, 0.004115, 1.3905, 0.003450, 1.3797],
    [10, 0.01217, 1.2571, 0.01129, 1.2156],
    [12, 0.02386, 1.1825, 0.02455, 1.1216],
    [15, 0.04481, 1.1233, 0.05008, 1.0440],
    [20, 0.09164, 1.0568, 0.09611, 0.9847],
    [25, 0.1571, 0.9991, 0.1533, 0.9491],
    [30, 0.2403, 0.9485, 0.2291, 0.9129],
    [35, 0.3374, 0.9047, 0.3224, 0.8761],
    [40, 0.4431, 0.8673, 0.4274, 0.8421],
    [50, 0.6600, 0.8084, 0.6472, 0.7871],
])


def rain_coeffs(f_ghz, pol="circular", elev_deg=0.0, tilt_deg=45.0):
    """Power-law coefficients (k, alpha) for specific rain attenuation gamma = k R^alpha (dB/km).

    Log-log interpolation of a coarse table after ITU-R P.838-3 (1-50 GHz). ``pol`` is 'h',
    'v' or 'circular' (use tilt 45 deg); the P.838 combination rule handles elevation/tilt.
    """
    f = np.log10(np.asarray(f_ghz, float))
    lf = np.log10(_P838[:, 0])
    kH = 10 ** np.interp(f, lf, np.log10(_P838[:, 1]))
    aH = np.interp(f, lf, _P838[:, 2])
    kV = 10 ** np.interp(f, lf, np.log10(_P838[:, 3]))
    aV = np.interp(f, lf, _P838[:, 4])
    if pol == "h":
        return kH, aH
    if pol == "v":
        return kV, aV
    th, tau = np.radians(elev_deg), np.radians(tilt_deg)
    c = np.cos(th) ** 2 * np.cos(2 * tau)
    k = (kH + kV + (kH - kV) * c) / 2
    a = (kH * aH + kV * aV + (kH * aH - kV * aV) * c) / (2 * k)
    return k, a


def rain_specific_atten(f_ghz, rain_rate_mmh, pol="circular", elev_deg=0.0):
    """Specific rain attenuation (dB/km) = k R^alpha."""
    k, a = rain_coeffs(f_ghz, pol, elev_deg)
    return k * np.asarray(rain_rate_mmh, float) ** a


def rain_atten_p618(f_ghz, elev_deg, p_percent, r001_mmh, lat_deg=45.0, h_rain_km=None,
                    h_station_km=0.0, pol="circular"):
    """Rain attenuation (dB) exceeded ``p_percent`` of an average year on an Earth-space path.

    Follows the step-by-step method of ITU-R P.618 (horizontal and vertical path-reduction
    factors, then scaling from 0.01% to p). Inputs: frequency (GHz), elevation (deg, >= 5),
    percentage p (0.001-5), the point rain rate exceeded 0.01% of the time R0.01 (mm/h, from
    ITU-R P.837 maps), station latitude (deg). ``h_rain_km`` defaults to 0.36 km above a
    0 C isotherm of 4.0 km (a mid-latitude value; take the true one from ITU-R P.839).
    """
    f, th = float(f_ghz), np.radians(elev_deg)
    p = np.asarray(p_percent, float)
    hR = (4.0 + 0.36) if h_rain_km is None else h_rain_km
    if hR <= h_station_km:
        return np.zeros_like(p)
    Ls = (hR - h_station_km) / np.sin(th)                    # slant path below rain height
    LG = Ls * np.cos(th)                                      # horizontal projection
    k, a = rain_coeffs(f, pol, elev_deg)
    gR = k * r001_mmh ** a
    r001 = 1 / (1 + 0.78 * np.sqrt(LG * gR / f) - 0.38 * (1 - np.exp(-2 * LG)))
    zeta = np.degrees(np.arctan2(hR - h_station_km, LG * r001))
    LR = LG * r001 / np.cos(th) if zeta > elev_deg else (hR - h_station_km) / np.sin(th)
    chi = 36 - abs(lat_deg) if abs(lat_deg) < 36 else 0.0
    v001 = 1 / (1 + np.sqrt(np.sin(th)) * (31 * (1 - np.exp(-elev_deg / (1 + chi)))
                                           * np.sqrt(LR * gR) / f ** 2 - 0.45))
    A001 = gR * LR * v001
    if abs(lat_deg) >= 36:
        beta = 0.0
    elif elev_deg >= 25:
        beta = -0.005 * (abs(lat_deg) - 36)
    else:
        beta = -0.005 * (abs(lat_deg) - 36) + 1.8 - 4.25 * np.sin(th)
    beta = np.where(p >= 1, 0.0, beta)
    expo = -(0.655 + 0.033 * np.log(p) - 0.045 * np.log(A001) - beta * (1 - p) * np.sin(th))
    return A001 * (p / 0.01) ** expo


# ================================================================== payload / nonlinearity
SALEH_TWTA = dict(aa=2.1587, ba=1.1517, ap=4.0033, bp=9.1040)  # Saleh (1981) fitted values


def saleh_am_am_pm(r, aa=2.1587, ba=1.1517, ap=4.0033, bp=9.1040):
    """Saleh TWTA model: output amplitude A(r) = aa r/(1+ba r^2), phase shift P(r) = ap r^2/(1+bp r^2) rad."""
    r = np.asarray(r, float)
    return aa * r / (1 + ba * r ** 2), ap * r ** 2 / (1 + bp * r ** 2)


def saleh_twta(x, ibo_db=0.0, am_pm=True, **params):
    """Pass complex baseband ``x`` through a Saleh TWTA driven at input back-off ``ibo_db``.

    The input is scaled so that its *mean* power is ``ibo_db`` below the single-carrier
    saturation input power; the output is normalised so saturated output power is 1.
    Returns y. Output back-off (OBO) can be measured as -10 log10(mean |y|^2).
    """
    p = dict(SALEH_TWTA, **params)
    r_sat = 1 / np.sqrt(p["ba"])                          # input amplitude at saturation
    a_sat = p["aa"] * r_sat / (1 + p["ba"] * r_sat ** 2)  # saturated output amplitude
    x = np.asarray(x, complex)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2)) * r_sat * 10 ** (-ibo_db / 20)
    A, P = saleh_am_am_pm(np.abs(x), **p)
    ph = np.angle(x) + (P if am_pm else 0)
    return A / a_sat * np.exp(1j * ph)


# ================================================================== DVB-S2
# Ideal Es/N0 (dB) for quasi-error-free operation (PER 1e-7, AWGN, normal 64800-bit
# FECFRAME) and spectral efficiency (b/s/Hz per unit symbol rate, no pilots),
# after ETSI EN 302 307-1, Table 13.
DVBS2_MODCODS = [
    ("QPSK 1/4", 0.490, -2.35), ("QPSK 1/3", 0.656, -1.24), ("QPSK 2/5", 0.789, -0.30),
    ("QPSK 1/2", 0.989, 1.00), ("QPSK 3/5", 1.188, 2.23), ("QPSK 2/3", 1.322, 3.10),
    ("QPSK 3/4", 1.487, 4.03), ("QPSK 4/5", 1.587, 4.68), ("QPSK 5/6", 1.655, 5.18),
    ("QPSK 8/9", 1.767, 6.20), ("QPSK 9/10", 1.789, 6.42),
    ("8PSK 3/5", 1.779, 5.50), ("8PSK 2/3", 1.980, 6.62), ("8PSK 3/4", 2.228, 7.91),
    ("8PSK 5/6", 2.479, 9.35), ("8PSK 8/9", 2.646, 10.69), ("8PSK 9/10", 2.679, 10.98),
    ("16APSK 2/3", 2.637, 8.97), ("16APSK 3/4", 2.967, 10.21), ("16APSK 4/5", 3.166, 11.03),
    ("16APSK 5/6", 3.300, 11.61), ("16APSK 8/9", 3.523, 12.89), ("16APSK 9/10", 3.567, 13.13),
    ("32APSK 3/4", 3.703, 12.73), ("32APSK 4/5", 3.952, 13.64), ("32APSK 5/6", 4.120, 14.28),
    ("32APSK 8/9", 4.398, 15.69), ("32APSK 9/10", 4.453, 16.05),
]


def acm_select(esn0_db, margin_db=0.5, modcods=None):
    """Pick the most efficient DVB-S2 MODCOD whose threshold + margin <= esn0_db.

    Returns (name, spectral_efficiency, threshold_db), or None if even the most robust
    MODCOD does not close. Works on a scalar Es/N0.
    """
    best = None
    for name, eta, thr in (modcods or DVBS2_MODCODS):
        if thr + margin_db <= esn0_db and (best is None or eta > best[1]):
            best = (name, eta, thr)
    return best


def apsk(rings, radii, phases=None):
    """Generic APSK constellation: ``rings`` points per ring, ring ``radii``, first-point phases (rad).

    Returned points are normalised to unit average energy.
    """
    pts = []
    for i, (n, r) in enumerate(zip(rings, radii)):
        ph0 = (np.pi / n) if phases is None else phases[i]
        pts.append(r * np.exp(1j * (ph0 + 2 * np.pi * np.arange(n) / n)))
    pts = np.concatenate(pts)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


# ring-ratio parameters gamma from ETSI EN 302 307-1 (16APSK: R2/R1; 32APSK: R2/R1, R3/R1)
DVBS2_16APSK_GAMMA = {"2/3": 3.15, "3/4": 2.85, "4/5": 2.75, "5/6": 2.70, "8/9": 2.60, "9/10": 2.57}
DVBS2_32APSK_GAMMA = {"3/4": (2.84, 5.27), "4/5": (2.72, 4.87), "5/6": (2.64, 4.64),
                      "8/9": (2.54, 4.33), "9/10": (2.53, 4.30)}


def dvbs2_16apsk(rate="3/4"):
    """DVB-S2 4+12 16APSK constellation (unit energy) for the given code rate's ring ratio."""
    g = DVBS2_16APSK_GAMMA[rate]
    return apsk([4, 12], [1.0, g], [np.pi / 4, np.pi / 12])


def dvbs2_32apsk(rate="3/4"):
    """DVB-S2 4+12+16 32APSK constellation (unit energy) for the given code rate's ring ratios."""
    g1, g2 = DVBS2_32APSK_GAMMA[rate]
    return apsk([4, 12, 16], [1.0, g1, g2], [np.pi / 4, np.pi / 12, 0.0])


# ================================================================== protocols
def tcp_window_limited_rate(window_bytes, rtt_s):
    """Maximum TCP throughput (b/s) with a fixed window: 8 W / RTT."""
    return 8 * np.asarray(window_bytes, float) / np.asarray(rtt_s, float)


def bent_pipe_rtt(h, elev_deg=90.0):
    """Round-trip time (s) user->satellite->gateway and back, both ends at elevation ``elev``.

    Four slant-range traversals (request up/down, response up/down); processing ignored.
    """
    return 4 * slant_range(h, elev_deg) / C


if __name__ == "__main__":  # quick self-check
    print(f"GEO altitude {GEO_ALT/1e3:.1f} km, period {orbital_period(geo_radius())/3600:.3f} h")
    print(f"Starlink 550 km: T={orbital_period(R_E+550e3)/60:.1f} min v={circular_speed(550e3)/1e3:.2f} km/s")
    print(f"  max Doppler @12 GHz {max_doppler(550e3, 12e9)/1e3:.1f} kHz, rate {max_doppler_rate(550e3, 12e9):.0f} Hz/s")
    ps = leo_pass(550e3, 12e9, 90, 0, 0.5)
    print(f"  sim: max |f_D| {np.max(np.abs(ps['doppler']))/1e3:.1f} kHz, max |rate| {np.max(np.abs(ps['doppler_rate'])):.0f} Hz/s")
    print("GEO look angles from 45N, 10W to 0E:", [np.round(v, 2) for v in geo_look_angles(45, -10, 0)])
    print("rain 12 GHz 35 deg R=42:", rain_atten_p618(12, 35, [1, 0.1, 0.01], 42))
    print("ACM @ 9 dB:", acm_select(9.0))
