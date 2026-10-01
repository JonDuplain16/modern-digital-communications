"""Optional audio playback: ``play(x, fs)`` sends a mono or stereo signal to the speakers.

Backends, tried in order: ``sounddevice`` (if installed), Windows ``winsound`` (built in),
``afplay`` on macOS, ``paplay``/``aplay`` on Linux. Nothing is required: when no backend
works, ``play`` returns (False, reason) and the lab carries on silently. Set the
environment variable ``STUDIO_MUTE=1`` to disable audio (the self-test does this).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

_TMP = os.path.join(tempfile.gettempdir(), "studio_audio_{}.wav")
_counter = [0]


def _to_int16(x):
    x = np.asarray(x, float)
    peak = np.max(np.abs(x)) if x.size else 0
    if peak > 0:
        x = 0.8 * x / peak
    # 10 ms fade in/out to avoid clicks at the ends
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def _fade(x, fs):
    n = min(len(x) // 4, int(0.01 * fs))
    if n > 1:
        w = np.linspace(0, 1, n)
        x = x.copy()
        x[:n] *= w[:, None] if x.ndim == 2 else w
        x[-n:] *= w[::-1, None] if x.ndim == 2 else w[::-1]
    return x


def _resample_to(x, fs, target=48000):
    if abs(fs - target) < 1:
        return x, fs
    from scipy.signal import resample_poly
    from fractions import Fraction
    fr = Fraction(int(round(target)), int(round(fs))).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0), target


def available():
    """True if some playback backend is likely to work."""
    if os.environ.get("STUDIO_MUTE"):
        return False
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:
        pass
    if sys.platform.startswith("win"):
        return True
    return any(shutil.which(c) for c in ("afplay", "paplay", "aplay"))


def play(x, fs):
    """Play ``x`` (N samples, or N×2 for stereo) sampled at ``fs`` Hz. Non-blocking.
    Returns (ok, message)."""
    if os.environ.get("STUDIO_MUTE"):
        return False, "Audio is muted (STUDIO_MUTE)."
    x = np.asarray(x, float)
    x, fs = _resample_to(x, fs)
    x = _fade(x, fs)
    try:
        import sounddevice as sd
        sd.stop()
        sd.play(0.8 * x / (np.max(np.abs(x)) + 1e-12), int(fs))
        return True, "playing"
    except Exception:
        pass
    _counter[0] = (_counter[0] + 1) % 4
    fn = _TMP.format(_counter[0])
    try:
        pcm = _to_int16(x)
        with wave.open(fn, "wb") as w:
            w.setnchannels(2 if pcm.ndim == 2 else 1)
            w.setsampwidth(2)
            w.setframerate(int(fs))
            w.writeframes(pcm.tobytes())
    except OSError as e:
        return False, f"Could not write audio file: {e}"
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(fn, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True, "playing"
        except Exception as e:  # pragma: no cover
            return False, f"Audio failed: {e}"
    for cmd in (["afplay", fn], ["paplay", fn], ["aplay", "-q", fn]):
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "playing"
    return False, "No audio backend found (pip install sounddevice to enable sound)."
