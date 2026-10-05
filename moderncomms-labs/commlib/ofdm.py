"""CP-OFDM and DFT-s-OFDM building blocks plus 5G NR numerology helpers."""
from __future__ import annotations

import numpy as np

__all__ = ["OFDMConfig", "ofdm_modulate", "ofdm_demodulate", "comb_pilot_mask",
           "ls_channel_estimate", "papr_db", "ccdf", "nr_numerology",
           "dft_s_ofdm_modulate", "dft_s_ofdm_demodulate"]


class OFDMConfig:
    """Simple OFDM layout: nfft bins, n_used active subcarriers centred on DC
    (DC bin left empty), cyclic prefix length ncp samples."""

    def __init__(self, nfft=64, n_used=48, ncp=16):
        self.nfft, self.n_used, self.ncp = nfft, n_used, ncp
        half = n_used // 2
        k = np.concatenate([np.arange(-half, 0), np.arange(1, n_used - half + 1)])
        self.active = np.mod(k, nfft)  # FFT bin indices of active carriers
        self.k = k                     # signed subcarrier indices

    @property
    def sym_len(self):
        return self.nfft + self.ncp


def ofdm_modulate(grid, cfg: OFDMConfig):
    """grid: (n_sym, n_used) frequency-domain symbols -> time samples with CP.
    Scaled so that time-domain average power equals mean |grid|^2 * n_used/nfft."""
    grid = np.atleast_2d(grid)
    X = np.zeros((grid.shape[0], cfg.nfft), dtype=complex)
    X[:, cfg.active] = grid
    x = np.fft.ifft(X, axis=1) * np.sqrt(cfg.nfft)
    x = np.concatenate([x[:, -cfg.ncp:], x], axis=1) if cfg.ncp else x
    return x.reshape(-1)


def ofdm_demodulate(y, cfg: OFDMConfig, n_sym=None, timing_offset=0):
    """Remove CP (optionally sampling `timing_offset` samples early) and FFT."""
    L = cfg.sym_len
    n_sym = n_sym or len(y) // L
    y = y[:n_sym * L].reshape(n_sym, L)
    start = cfg.ncp - timing_offset
    Y = np.fft.fft(y[:, start:start + cfg.nfft], axis=1) / np.sqrt(cfg.nfft)
    return Y[:, cfg.active]


def comb_pilot_mask(n_sym, n_used, f_spacing=6, t_spacing=1, offset=0):
    """Boolean pilot mask on a (n_sym, n_used) grid."""
    m = np.zeros((n_sym, n_used), dtype=bool)
    for s in range(0, n_sym, t_spacing):
        m[s, (offset + (s // t_spacing) * 0) % f_spacing::f_spacing] = True
    return m


def ls_channel_estimate(Y, pilots, mask, method="linear"):
    """LS estimates at pilots, then interpolation across frequency and time.

    method: 'linear' (per-symbol linear interpolation in frequency, then in time)
    or 'nearest'. Returns H_hat with shape of Y.
    """
    n_sym, n_used = Y.shape
    H = np.full(Y.shape, np.nan + 0j)
    H[mask] = Y[mask] / pilots[mask]
    k = np.arange(n_used)
    # frequency interpolation on symbols that carry pilots
    rows = np.where(mask.any(axis=1))[0]
    for r in rows:
        kp = np.where(mask[r])[0]
        if method == "nearest":
            idx = kp[np.argmin(np.abs(k[:, None] - kp[None, :]), axis=1)]
            H[r] = H[r, idx]
        else:
            H[r] = np.interp(k, kp, H[r, kp].real) + 1j * np.interp(k, kp, H[r, kp].imag)
    # time interpolation
    for c in range(n_used):
        H[:, c] = (np.interp(np.arange(n_sym), rows, H[rows, c].real)
                   + 1j * np.interp(np.arange(n_sym), rows, H[rows, c].imag))
    return H


def papr_db(x, sym_len):
    x = x[:len(x) // sym_len * sym_len].reshape(-1, sym_len)
    p = np.abs(x) ** 2
    return 10 * np.log10(p.max(axis=1) / p.mean(axis=1))


def ccdf(values, grid=None):
    grid = np.linspace(np.min(values), np.max(values), 200) if grid is None else grid
    return grid, np.array([(values > g).mean() for g in grid])


def _dfts_bins(cfg: OFDMConfig):
    """Contiguous (localized) bins for an n_used-point DFT-spread block, centred on DC."""
    M = cfg.n_used
    return (np.arange(M) - M // 2) % cfg.nfft


def dft_s_ofdm_modulate(sym, cfg: OFDMConfig, contiguous=False):
    """DFT-spread OFDM (SC-FDMA): an n_used-point DFT precoding onto n_used subcarriers.

    contiguous=False (legacy default, used by the book's Chapter 17 PAPR figure) maps the DFT
    outputs onto ``cfg.active``, i.e. around DC with the DC bin skipped: the hole in the middle
    breaks the single-carrier structure and raises the PAPR at 10^-3 by about 0.3 dB.
    contiguous=True is the true localized mapping of LTE/NR (TS 36.211 / 38.211): the
    fftshift-ed DFT outputs occupy n_used adjacent bins, DC included, so the time signal is the
    symbol stream itself, periodically sinc-interpolated (nfft / n_used samples per symbol)."""
    sym = np.asarray(sym).reshape(-1, cfg.n_used)
    S = np.fft.fft(sym, axis=1) / np.sqrt(cfg.n_used)
    if not contiguous:
        return ofdm_modulate(S, cfg)
    X = np.zeros((len(sym), cfg.nfft), dtype=complex)
    X[:, _dfts_bins(cfg)] = np.fft.fftshift(S, axes=1)
    x = np.fft.ifft(X, axis=1) * np.sqrt(cfg.nfft)
    x = np.concatenate([x[:, -cfg.ncp:], x], axis=1) if cfg.ncp else x
    return x.reshape(-1)


def dft_s_ofdm_demodulate(y, cfg: OFDMConfig, H=None, n0=0.0, contiguous=False):
    """Inverse of ``dft_s_ofdm_modulate`` (same ``contiguous`` flag); optional per-subcarrier
    MMSE equalisation with H given on the occupied subcarriers in the transmit order."""
    if contiguous:
        L = cfg.sym_len
        n_sym = len(y) // L
        yy = np.asarray(y)[:n_sym * L].reshape(n_sym, L)
        Yf = np.fft.fft(yy[:, cfg.ncp:cfg.ncp + cfg.nfft], axis=1) / np.sqrt(cfg.nfft)
        Y = np.fft.ifftshift(Yf[:, _dfts_bins(cfg)], axes=1)
    else:
        Y = ofdm_demodulate(y, cfg)
    if H is not None:  # per-subcarrier MMSE equalization before de-spreading
        Y = Y * np.conj(H) / (np.abs(H) ** 2 + n0)
    return np.fft.ifft(Y, axis=1) * np.sqrt(cfg.n_used)


def nr_numerology(mu):
    """5G NR numerology per 3GPP TS 38.211 (normal cyclic prefix).

    Returns a dict with subcarrier spacing, slot duration, symbol and CP lengths.
    """
    scs = 15e3 * 2 ** mu
    Tc = 1 / (480e3 * 4096)          # NR basic time unit
    kappa = 64
    Nu = 2048 * kappa * 2 ** -mu * Tc
    Ncp = 144 * kappa * 2 ** -mu * Tc
    return {
        "mu": mu,
        "scs_khz": scs / 1e3,
        "slots_per_subframe": 2 ** mu,
        "slot_ms": 1.0 / 2 ** mu,
        "symbols_per_slot": 14,
        "useful_symbol_us": Nu * 1e6,
        "cp_us": Ncp * 1e6,
        "rb_bandwidth_khz": 12 * scs / 1e3,
    }
