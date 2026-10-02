"""commlib.cellsim -- system-level simulation helpers for Chapter 20 and Lab 26.

Import explicitly:  ``from commlib import cellsim as cs``.  Builds on ``commlib.cellular``
(Erlang, hexagonal geometry, ALOHA/CSMA, Bianchi, PPP, schedulers), which it never modifies.
The functions here are the ones the book's figure script (book/figscripts/ch20_figs.py) uses
inline, collected so that the lab reproduces the chapter's numbers:

* reuse: sector antenna pattern, SIR samples over a cell with shadowing, SIR maps of a hex grid
* traffic: Erlang distribution (truncated Poisson) of busy channels
* random access: ALOHA event timelines, slotted-ALOHA backlog dynamics and drift, 802.11 timing,
  RACH contention with access-class barring
* mobility: Gudmundson-correlated shadowing and an event-A3 handover simulator with ping-pong count
* coverage: Rayleigh SINR on a hexagonal grid at unit site density
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from . import cellular as cel


# ================================================================== reuse and SIR

def sector_gain_db(theta_deg, bw=65.0, am=20.0):
    """3GPP horizontal sector pattern -min(12 (theta/bw)^2, am) dB."""
    th = (np.asarray(theta_deg) + 180) % 360 - 180
    return -np.minimum(12 * (th / bw) ** 2, am)


def interferer_sites(i, j, tiers=2):
    """Co-channel interferer positions (complex, cell radius 1) for reuse (i, j); reuse 1
    (i, j) = (1, 0) returns every neighbouring site out to 2*tiers rings."""
    if (i, j) == (1, 0):
        q, r = cel.hex_axial_grid(2 * tiers)
        x, y = cel.axial_to_xy(q, r)
        return (x + 1j * y)[1:]
    return cel.cochannel_centres(i, j, tiers=tiers)


def sir_samples(i, j, n=4.0, sigma=8.0, sectors=1, users=20000, rng=None, tiers=2):
    """Downlink SIR (dB) of users dropped uniformly in the central cell (in the sector facing
    0 degrees when sectors = 3), independent log-normal shadowing on every link."""
    rng = np.random.default_rng(7) if rng is None else rng
    u = cel.drop_in_hex(users, rng=rng, rmin=0.05)
    if sectors == 3:
        u = u[np.abs(np.angle(u, deg=True)) < 60]
    bs = interferer_sites(i, j, tiers)
    sh = lambda shape: 10 ** (sigma * rng.standard_normal(shape) / 10)
    S = np.abs(u) ** -n * sh(len(u))
    d = u[:, None] - bs[None, :]
    g = np.abs(d) ** -n * sh((len(u), len(bs)))
    if sectors == 3:
        S = S * 10 ** (sector_gain_db(np.angle(u, deg=True)) / 10)
        g = g * 10 ** (sector_gain_db(np.angle(d, deg=True)) / 10)
    return 10 * np.log10(S / g.sum(axis=1))


def hex_outlines(centres, R=1.0):
    """x, y polyline (NaN-separated) tracing pointy-top hexagons of circumradius R."""
    a = np.deg2rad(np.arange(30, 391, 60))
    c = np.asarray(centres)
    x = c.real[:, None] + R * np.cos(a)[None, :]
    y = c.imag[:, None] + R * np.sin(a)[None, :]
    x = np.c_[x, np.full(len(c), np.nan)].ravel()
    y = np.c_[y, np.full(len(c), np.nan)].ravel()
    return x, y


def sir_map(xs, ys, sites, labels, n=4.0, sectors=1, dmin=0.04):
    """Geometric downlink SIR (dB) on a grid for sites (complex) with channel-set labels.

    sectors = 1: omni cells. sectors = 3: three 120-degree sectors per site (boresights 0, 120,
    240 degrees, 3GPP pattern); a sector reuses only the channels of the same label and the
    same orientation. Every location is served by its strongest cell/sector."""
    X, Y = np.meshgrid(xs, ys)
    P = X + 1j * Y
    D = P[..., None] - sites[None, None, :]
    d = np.maximum(np.abs(D), dmin)
    pl = d ** -n
    if sectors == 1:
        best = np.argmax(pl, axis=-1)
        S = np.take_along_axis(pl, best[..., None], -1)[..., 0]
        same = labels[None, None, :] == labels[best][..., None]
        I = np.sum(pl * same, axis=-1) - S
        return 10 * np.log10(S / np.maximum(I, 1e-30))
    ang = np.angle(D, deg=True)
    gs = np.empty(d.shape + (3,), dtype=np.float32)               # (ny, nx, sites, 3)
    for s_, b in enumerate((0.0, 120.0, 240.0)):
        th = (ang - b + 180) % 360 - 180
        gs[..., s_] = pl * 10 ** (-np.minimum(12 * (th / 65.0) ** 2, 20.0) / 10)
    flat = gs.reshape(gs.shape[0], gs.shape[1], -1)
    best = np.argmax(flat, axis=-1)
    S = np.take_along_axis(flat, best[..., None], -1)[..., 0]
    site_b, sec_b = np.divmod(best, 3)
    same = (labels[None, None, :] == labels[site_b][..., None])          # (ny, nx, sites)
    gsec = np.take_along_axis(gs, np.broadcast_to(sec_b[..., None, None], d.shape + (1,)),
                              axis=-1)[..., 0]
    I = np.sum(gsec * same, axis=-1) - S
    return 10 * np.log10(S / np.maximum(I, 1e-30))


# ================================================================== traffic

def erlang_occupancy(A, C):
    """Probability of k busy channels (k = 0..C) in an M/M/C/C loss system: the truncated
    Poisson ('Erlang') distribution. P[C] is Erlang B."""
    k = np.arange(C + 1)
    logp = k * np.log(max(A, 1e-300)) - np.cumsum(np.r_[0.0, np.log(np.arange(1, C + 1))])
    p = np.exp(logp - logp.max())
    return p / p.sum()


# ================================================================== random access

def aloha_events(G, T, slotted=False, rng=None):
    """Poisson packet starts at rate G per packet time over [0, T); unit-length packets.
    Returns (start times, success flags). Slotted: starts at integer slot boundaries."""
    rng = np.random.default_rng(rng)
    n = rng.poisson(G * T)
    if slotted:
        t = np.sort(rng.integers(0, int(T), n)).astype(float)
        cnt = np.bincount(t.astype(int), minlength=int(T))
        ok = cnt[t.astype(int)] == 1
        return t, ok
    t = np.sort(rng.random(n) * T)
    gap = np.diff(t)
    ok = np.r_[True, gap > 1] & np.r_[gap > 1, True]
    return t, ok


def aloha_backlog_step(state, lam, q_fixed, rng, mode="fixed"):
    """One slot of slotted ALOHA with Poisson arrivals and backlogged retransmissions.

    state: dict(n=backlog, nhat=estimate). mode 'fixed' retransmits with probability q_fixed;
    'pb' uses Rivest's pseudo-Bayesian q = 1/nhat. Returns 1 if the slot carried a success."""
    new = rng.poisson(lam)
    n, nhat = state["n"], state["nhat"]
    q = q_fixed if mode == "fixed" else min(1.0, 1.0 / max(nhat, 1.0))
    ret = rng.binomial(n, q) if n > 0 else 0
    tot = new + ret
    succ = 0
    if tot == 1:
        succ = 1
        if ret == 1:
            n -= 1
    elif tot > 1:
        n += new
    nhat = max(lam, nhat + lam - 1) if tot <= 1 else nhat + lam + 1 / (np.e - 2)
    state["n"], state["nhat"] = n, nhat
    return succ


def aloha_drift(m, qa, qr):
    """Finite-population slotted ALOHA (m users, new-packet probability qa, retransmission qr):
    returns (n, departure rate P_succ(n), arrival rate (m - n) qa)."""
    nb = np.arange(0, m + 1)
    ps = ((m - nb) * qa * (1 - qa) ** np.maximum(m - nb - 1, 0) * (1 - qr) ** nb +
          nb * qr * (1 - qr) ** np.maximum(nb - 1, 0) * (1 - qa) ** (m - nb))
    return nb, ps, (m - nb) * qa


def ofdm_ppdu_us(nbytes, rate_mbps):
    """802.11a/g OFDM PPDU duration (us): 20 us preamble+SIGNAL, 16 service + 6 tail bits."""
    ndbps = rate_mbps * 4
    return 20 + 4 * np.ceil((16 + 8 * nbytes + 6) / ndbps)


def dcf_times(payload=1500, rate=54, basic=24):
    """802.11a timing for Bianchi's model: (slot, payload time, basic dict, RTS/CTS dict)."""
    slot, sifs = 9e-6, 16e-6
    difs = sifs + 2 * slot
    data = ofdm_ppdu_us(payload + 34, rate) * 1e-6
    ack = ofdm_ppdu_us(14, basic) * 1e-6
    rts = ofdm_ppdu_us(20, basic) * 1e-6
    cts = ofdm_ppdu_us(14, basic) * 1e-6
    pl = payload * 8 / (rate * 1e6)
    basic_ = dict(Ts=data + sifs + ack + difs, Tc=data + difs + sifs + ack)
    rtscts = dict(Ts=rts + sifs + cts + sifs + data + sifs + ack + difs, Tc=rts + difs + sifs + cts)
    return slot, pl, basic_, rtscts


def rach_sim(ndev, spread_s, beta=True, M=54, period=5e-3, maxtx=10, backoff=20e-3,
             acb=1.0, t_barring=0.5, rng=None):
    """Massive access through LTE/NR-style RACH opportunities every ``period`` seconds.

    Devices activate over ``spread_s`` seconds (Beta(3,4) burst or uniform). In each
    opportunity every active device first passes access-class barring with probability
    ``acb`` (else it waits (0.7 + 0.6 U) * t_barring), then picks one of M preambles; a
    preamble chosen by exactly one device succeeds; collided devices back off uniformly up to
    ``backoff`` and give up after ``maxtx`` attempts.
    Returns dict(success, mean_delay, attempts, per_ro=(success counts, collided counts))."""
    rng = np.random.default_rng(5) if rng is None else rng
    t = (rng.beta(3, 4, ndev) if beta else rng.random(ndev)) * spread_s
    nxt = np.ceil(t / period).astype(int)
    start = nxt.copy()
    tries = np.zeros(ndev, int)
    done = np.zeros(ndev, bool)
    fail = np.zeros(ndev, bool)
    finish = np.full(ndev, -1)
    horizon = int(spread_s / period) + 1200
    succ_ro = np.zeros(horizon, int)
    coll_ro = np.zeros(horizon, int)
    for ro in range(horizon):
        act = np.flatnonzero((nxt == ro) & ~done & ~fail)
        if len(act) == 0:
            if ro > nxt[~done & ~fail].max(initial=-1):
                break
            continue
        if acb < 1.0:
            passed = rng.random(len(act)) < acb
            barred = act[~passed]
            nxt[barred] = ro + 1 + np.floor((0.7 + 0.6 * rng.random(len(barred))) * t_barring
                                           / period).astype(int)
            act = act[passed]
            if len(act) == 0:
                continue
        pre = rng.integers(0, M, len(act))
        cnt = np.bincount(pre, minlength=M)
        ok = cnt[pre] == 1
        done[act[ok]] = True
        finish[act[ok]] = ro
        succ_ro[ro] = ok.sum()
        coll_ro[ro] = (~ok).sum()
        bad = act[~ok]
        tries[act] += 1
        fail[bad[tries[bad] >= maxtx]] = True
        rest = bad[tries[bad] < maxtx]
        nxt[rest] = ro + 1 + rng.integers(0, int(backoff / period) + 1, len(rest)) + 1
    delay = np.mean((finish[done] - start[done]) * period) if done.any() else float("nan")
    used = max(int(np.max(np.flatnonzero(succ_ro + coll_ro), initial=0)) + 1, 1)
    return dict(success=float(done.mean()), mean_delay=float(delay),
                attempts=float(np.mean(tries[tries > 0])) if (tries > 0).any() else 0.0,
                per_ro=(succ_ro[:used], coll_ro[:used]))


# ================================================================== mobility

def correlated_shadow(n, dx, dcorr, sigma, rng):
    """Gudmundson log-normal shadowing along a route: AR(1) with correlation exp(-dx/dcorr)."""
    a = np.exp(-dx / dcorr)
    w = rng.standard_normal(n)
    s = np.empty(n)
    s[0] = w[0]
    if n > 1:
        s[1:] = lfilter([np.sqrt(1 - a * a)], [1, -a], w[1:], zi=[a * w[0]])[0]
    return sigma * s


def handover_drive(hyst, ttt_m, rng, L=2000.0, dx=1.0, sigma=8.0, dcorr=50.0, n=3.5, kfilt=0.5,
                   rho=0.5, pingpong_m=None):
    """A user drives from site 1 to site 2 (L metres apart, as in the book's Figure 20.x).

    Shadowing: sigma dB, Gudmundson decorrelation distance dcorr, inter-site correlation rho;
    measurements add 1.5 dB of residual error per metre and pass a layer-3 filter (factor kfilt).
    Event-A3 trigger: the other cell exceeds the serving one by ``hyst`` dB for ``ttt_m`` metres.
    A ping-pong is a handover back to the previous cell within ``pingpong_m`` metres
    (default 75 m, i.e. 5 s at 15 m/s).
    Returns dict(x, f1, f2, lm1, lm2, serving, ho_positions, count, pingpong, worse)."""
    pingpong_m = 75.0 if pingpong_m is None else pingpong_m
    x = np.arange(0, L, dx)
    d1 = np.hypot(x + 50, 30)
    d2 = np.hypot(L + 50 - x, 30)
    c = correlated_shadow(len(x), dx, dcorr, sigma, rng)
    s1 = np.sqrt(rho) * c + np.sqrt(1 - rho) * correlated_shadow(len(x), dx, dcorr, sigma, rng)
    s2 = np.sqrt(rho) * c + np.sqrt(1 - rho) * correlated_shadow(len(x), dx, dcorr, sigma, rng)
    lm1 = -10 * n * np.log10(d1) + s1
    lm2 = -10 * n * np.log10(d2) + s2
    m1 = lm1 + rng.normal(0, 1.5, len(x))
    m2 = lm2 + rng.normal(0, 1.5, len(x))
    f1 = lfilter([kfilt], [1, -(1 - kfilt)], m1, zi=[(1 - kfilt) * m1[0]])[0]
    f2 = lfilter([kfilt], [1, -(1 - kfilt)], m2, zi=[(1 - kfilt) * m2[0]])[0]
    diff = f2 - f1
    serv = 0
    trig = 0.0
    hos = []
    srv = np.empty(len(x), int)
    for k in range(len(x)):
        lead = diff[k] if serv == 0 else -diff[k]
        if lead > hyst:
            trig += dx
            if trig > ttt_m:
                serv = 1 - serv
                hos.append(x[k])
                trig = 0.0
        else:
            trig = 0.0
        srv[k] = serv
    hos = np.array(hos)
    pp = int(np.sum(np.diff(hos) < pingpong_m)) if len(hos) > 1 else 0
    LM = np.vstack([lm1, lm2])
    worse = float(np.mean(LM[srv, np.arange(len(x))] < LM.max(axis=0) - 3))
    return dict(x=x, f1=f1, f2=f2, lm1=lm1, lm2=lm2, serving=srv, ho_positions=hos,
                count=len(hos), pingpong=pp, worse=worse)


# ================================================================== coverage

def hex_rayleigh_sinr(alpha=4.0, users=40000, rng=None):
    """Interference-limited SINR (linear) of users in the central cell of a hexagonal grid
    scaled to unit site density, Rayleigh fading on every link (the book's hexagonal curve)."""
    rng = np.random.default_rng(2) if rng is None else rng
    q, rr = cel.hex_axial_grid(6)
    x, y = cel.axial_to_xy(q, rr)
    R = np.sqrt(2 / (3 * np.sqrt(3)))
    sites = (x + 1j * y) * R
    u = cel.drop_in_hex(users, R=R, rng=rng)
    d = np.abs(u[:, None] - sites[None, :])
    p = rng.exponential(size=d.shape) * d ** -alpha
    return p[:, 0] / (p.sum(axis=1) - p[:, 0])
