"""gf: finite fields GF(2^m), Reed-Solomon and binary BCH codes, CRC helpers.

Written to be read alongside Chapter 14 (Classical Error-Control Codes). Everything is
table-driven and deliberately plain: a field element is a Python/NumPy integer whose bits
are the coefficients of a polynomial in alpha (bit i <-> alpha^i in the polynomial basis).

Import explicitly (it is not re-exported by ``commlib``):

    from commlib import gf
    F = gf.GF(8)                       # GF(2^8) with the default primitive polynomial 0x11D
    rs = gf.ReedSolomon(255, 223, F)   # t = 16
    c = rs.encode(msg); msg_hat, nerr = rs.decode(r)

Polynomial conventions
----------------------
* Field polynomials used inside the algorithms (syndrome polynomial S(x), error locator
  Lambda(x), evaluator Omega(x)) are stored LOWEST degree first: p[i] is the coefficient of x^i.
* Codewords are stored in TRANSMISSION order: c[0] is the coefficient of x^(n-1) (sent first),
  c[n-1] the coefficient of x^0. Systematic encoders put the message first, parity last.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PRIMITIVE_POLYS", "GF", "ReedSolomon", "BCH", "berlekamp_massey_gf2",
           "crc_bits", "CRC_CATALOG", "cyclotomic_cosets"]

# Primitive polynomials, bit i = coefficient of x^i (the conventional choices; 0x11D is the
# one used by DVB, ATSC, QR codes, RAID-6 and CD CIRC; CCSDS uses 0x187).
PRIMITIVE_POLYS = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011,
                   7: 0b10001001, 8: 0x11D, 9: 0x211, 10: 0x409, 11: 0x805, 12: 0x1053,
                   13: 0x201B, 14: 0x402B, 15: 0x8003, 16: 0x1002D}


def poly_str(p, var="x"):
    """Pretty-print a GF(2) polynomial given as an int (bit i <-> x^i)."""
    terms = []
    for i in range(int(p).bit_length() - 1, -1, -1):
        if (p >> i) & 1:
            terms.append("1" if i == 0 else (var if i == 1 else f"{var}^{i}"))
    return " + ".join(terms) if terms else "0"


class GF:
    """The finite field GF(2^m) built from a primitive polynomial."""

    def __init__(self, m, prim=None):
        self.m = m
        self.q = 1 << m
        self.prim = PRIMITIVE_POLYS[m] if prim is None else prim
        n = self.q - 1
        self.exp = np.zeros(2 * n, dtype=np.int64)     # doubled to avoid a modulo in mul
        self.log = np.full(self.q, -1, dtype=np.int64)  # log[0] undefined (-1)
        x = 1
        for i in range(n):
            self.exp[i] = x
            if self.log[x] != -1:
                raise ValueError(f"polynomial {self.prim:#x} is not primitive for m={m}")
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= self.prim
        self.exp[n:] = self.exp[:n]

    # --------------------------------------------------------------- scalar arithmetic
    @property
    def n(self):
        return self.q - 1

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] - self.log[b]) % self.n])

    def inv(self, a):
        return self.div(1, a)

    def pow(self, a, e):
        if a == 0:
            return 0 if e else 1
        return int(self.exp[(self.log[a] * e) % self.n])

    def alpha(self, i):
        """alpha^i for any integer i."""
        return int(self.exp[i % self.n])

    # vectorised multiply (arrays of elements)
    def vmul(self, a, b):
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        out = self.exp[(self.log[a] + self.log[b]) % self.n]
        return np.where((a == 0) | (b == 0), 0, out)

    # --------------------------------------------------------------- polynomials (low-first)
    def poly_eval(self, p, x):
        """Evaluate p (lowest degree first) at x by Horner's rule."""
        y = 0
        for c in reversed(list(p)):
            y = self.mul(y, x) ^ int(c)
        return y

    def poly_mul(self, a, b):
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    out[i + j] ^= self.mul(int(ai), int(bj))
        return out

    def poly_scale(self, p, s):
        return [self.mul(int(c), s) for c in p]

    def poly_add(self, a, b):
        n = max(len(a), len(b))
        a = list(a) + [0] * (n - len(a)); b = list(b) + [0] * (n - len(b))
        return [x ^ y for x, y in zip(a, b)]

    def poly_deriv(self, p):
        """Formal derivative in characteristic 2: only odd powers survive."""
        return [p[i] if i % 2 == 1 else 0 for i in range(1, len(p))] or [0]

    def elem_str(self, a, style="power"):
        """Render an element as 'alpha^i' (power), '0b..' (vector) or polynomial in alpha."""
        a = int(a)
        if style == "power":
            return "0" if a == 0 else ("1" if a == 1 else f"a^{self.log[a]}")
        if style == "vector":
            return format(a, f"0{self.m}b")
        return poly_str(a, "a")

    def table(self):
        """Rows (i, alpha^i as polynomial, as m-bit vector, as integer) for i = 0..q-2."""
        return [(i, poly_str(int(self.exp[i]), "a"), format(int(self.exp[i]), f"0{self.m}b"),
                 int(self.exp[i])) for i in range(self.n)]

    def minimal_poly(self, i):
        """Minimal polynomial of alpha^i over GF(2), returned as an int (bit k <-> x^k)."""
        coset = sorted({(i * (1 << k)) % self.n for k in range(self.m)})
        p = [1]
        for j in coset:
            p = self.poly_mul(p, [self.alpha(j), 1])       # (x + alpha^j), low-first
        assert all(c in (0, 1) for c in p)
        return sum(int(c) << k for k, c in enumerate(p))


def cyclotomic_cosets(m):
    n = (1 << m) - 1
    seen, out = set(), []
    for i in range(n):
        if i not in seen:
            c = sorted({(i * (1 << k)) % n for k in range(m)})
            seen.update(c); out.append(c)
    return out


# ------------------------------------------------------------------- GF(2) polynomial helpers
def gf2_polymul(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1; b >>= 1
    return out


def gf2_polymod(a, b):
    db = b.bit_length() - 1
    while a and a.bit_length() - 1 >= db:
        a ^= b << (a.bit_length() - 1 - db)
    return a


# ------------------------------------------------------------------- Berlekamp-Massey core
def _berlekamp_massey(F, S):
    """Shortest LFSR (connection polynomial Lambda, low-first, Lambda[0] = 1) generating S."""
    L = 0
    Lam = [1]; B = [1]
    b = 1
    mshift = 1
    for r in range(len(S)):
        d = S[r]
        for i in range(1, L + 1):
            if i < len(Lam):
                d ^= F.mul(Lam[i], S[r - i])
        if d == 0:
            mshift += 1
            continue
        coef = F.div(d, b)
        T = list(Lam)
        shifted = [0] * mshift + F.poly_scale(B, coef)
        Lam = F.poly_add(Lam, shifted)
        if 2 * L <= r:
            L = r + 1 - L
            B = T; b = d; mshift = 1
        else:
            mshift += 1
    while len(Lam) > 1 and Lam[-1] == 0:
        Lam.pop()
    return Lam, L


class _GF2:
    """GF(2) viewed as a field object for the shared Berlekamp-Massey routine."""
    def mul(self, a, b): return a & b
    def div(self, a, b): return a
    def poly_scale(self, p, s): return [c & s for c in p]
    def poly_add(self, a, b):
        n = max(len(a), len(b))
        a = list(a) + [0] * (n - len(a)); b = list(b) + [0] * (n - len(b))
        return [x ^ y for x, y in zip(a, b)]


def berlekamp_massey_gf2(bits):
    """Linear complexity and connection polynomial (low-first list) of a binary sequence."""
    lam, L = _berlekamp_massey(_GF2(), [int(b) for b in bits])
    return lam, L


# ------------------------------------------------------------------- Reed-Solomon
class ReedSolomon:
    """Systematic (n, k) Reed-Solomon code over GF(2^m), possibly shortened (n < 2^m - 1).

    g(x) = prod_{j=0}^{2t-1} (x - alpha^(fcr + j)). Decoding: syndromes, Berlekamp-Massey,
    Chien search, Forney's formula. ``decode`` returns (message, number of corrected symbols)
    or (received message part, -1) on detected failure. Optional erasure positions (indices
    in transmission order) are supported: 2*errors + erasures <= n - k.
    """

    def __init__(self, n, k, field=None, fcr=0):
        self.F = field if field is not None else GF(8)
        if n > self.F.n:
            raise ValueError("n too large for the field")
        self.n, self.k, self.fcr = n, k, fcr
        self.nk = n - k
        self.t = self.nk // 2
        g = [1]
        for j in range(self.nk):
            g = self.F.poly_mul(g, [self.F.alpha(fcr + j), 1])
        self.g = g                     # low-first, monic, degree n-k

    def encode(self, msg):
        F = self.F
        msg = [int(v) for v in msg]
        assert len(msg) == self.k
        # long division of m(x) x^(n-k) by g(x), processed high degree first (LFSR form)
        reg = [0] * self.nk                          # reg[i] <-> x^i of the remainder
        for mi in msg:
            fb = mi ^ reg[-1]
            for i in range(self.nk - 1, 0, -1):
                reg[i] = reg[i - 1] ^ F.mul(fb, self.g[i])
            reg[0] = F.mul(fb, self.g[0])
        parity = reg[::-1]                           # high degree first
        return np.array(msg + parity, dtype=np.int64)

    def syndromes(self, r):
        F = self.F
        # r in transmission order -> r(x) with r[0] the x^(n-1) coefficient
        low = [int(v) for v in r[::-1]]
        return [F.poly_eval(low, F.alpha(self.fcr + j)) for j in range(self.nk)]

    def decode(self, r, erasures=(), return_detail=False):
        F = self.F
        r = np.array(r, dtype=np.int64).copy()
        S = self.syndromes(r)
        detail = {"S": S}
        if not any(S):
            out = (r[:self.k], 0)
            return (*out, detail) if return_detail else out
        # erasure locator Gamma(x) = prod (1 - X_j x), X_j = alpha^(position power)
        gam = [1]
        for idx in erasures:
            gam = F.poly_mul(gam, [1, F.alpha(self.n - 1 - idx)])
        # Forney syndromes: T(x) = S(x) Gamma(x) mod x^(n-k); run BM on the tail
        T = F.poly_mul(S, gam)[:self.nk]
        e = len(erasures)
        lam_err, L = _berlekamp_massey(F, T[e:])
        lam = F.poly_mul(lam_err, gam)
        detail.update(Lambda=lam, L=L)
        # Chien search over the positions that exist in the (possibly shortened) code
        pos = []
        for j in range(self.n):
            p = self.n - 1 - j                        # power of x at index j
            if F.poly_eval(lam, F.alpha(-p)) == 0:
                pos.append(j)
        nroots = len(pos)
        if nroots != len(lam) - 1 or 2 * L + e > self.nk:
            out = (r[:self.k], -1)
            return (*out, detail) if return_detail else out
        omega = F.poly_mul(S, lam)[:self.nk]
        dlam = F.poly_deriv(lam)
        vals = []
        for j in pos:
            p = self.n - 1 - j
            Xinv = F.alpha(-p)
            num = F.mul(F.poly_eval(omega, Xinv), F.alpha(p * (1 - self.fcr)))
            den = F.poly_eval(dlam, Xinv)
            if den == 0:
                out = (r[:self.k], -1)
                return (*out, detail) if return_detail else out
            v = F.div(num, den)
            vals.append(v)
            r[j] ^= v
        detail.update(Omega=omega, positions=pos, values=vals)
        if any(self.syndromes(r)):
            out = (r[:self.k], -1)
        else:
            out = (r[:self.k], sum(1 for v in vals if v))
        return (*out, detail) if return_detail else out


# ------------------------------------------------------------------- binary BCH
class BCH:
    """Narrow-sense primitive binary BCH code of length n = 2^m - 1 (optionally shortened)
    and designed error-correcting capability t. Decoded with Berlekamp-Massey + Chien."""

    def __init__(self, m, t, field=None, shorten=0):
        self.F = field if field is not None else GF(m)
        F = self.F
        self.t = t
        N = F.n
        g = 1
        used = set()
        for i in range(1, 2 * t + 1, 1):
            c = min((i * (1 << k)) % N for k in range(m))
            if c not in used:
                used.add(c)
                g = gf2_polymul(g, F.minimal_poly(i))
        self.g = g
        self.nk = g.bit_length() - 1
        self.n = N - shorten
        self.k = self.n - self.nk

    def encode(self, msg):
        msg = [int(b) for b in msg]
        m = 0
        for b in msg:
            m = (m << 1) | b
        rem = gf2_polymod(m << self.nk, self.g)
        par = [(rem >> i) & 1 for i in range(self.nk - 1, -1, -1)]
        return np.array(msg + par, dtype=np.int64)

    def decode(self, r):
        F = self.F
        r = np.array(r, dtype=np.int64).copy()
        low = [int(v) for v in r[::-1]]
        S = [F.poly_eval(low, F.alpha(j)) for j in range(1, 2 * self.t + 1)]
        if not any(S):
            return r[:self.k], 0
        lam, L = _berlekamp_massey(F, S)
        pos = [j for j in range(self.n) if F.poly_eval(lam, F.alpha(-(self.n - 1 - j))) == 0]
        if len(pos) != len(lam) - 1 or L > self.t:
            return r[:self.k], -1
        r[pos] ^= 1
        return r[:self.k], len(pos)


# ------------------------------------------------------------------- CRC helpers
# name: (width, poly without the x^width term, init, refin, refout, xorout, check("123456789"))
CRC_CATALOG = {
    "CRC-8/ATM (HEC)": (8, 0x07, 0x00, False, False, 0x00, 0xF4),
    "CRC-16/CCITT-FALSE": (16, 0x1021, 0xFFFF, False, False, 0x0000, 0x29B1),
    "CRC-16/KERMIT": (16, 0x1021, 0x0000, True, True, 0x0000, 0x2189),
    "CRC-16/ARC": (16, 0x8005, 0x0000, True, True, 0x0000, 0xBB3D),
    "CRC-24/LTE-A": (24, 0x864CFB, 0x000000, False, False, 0x000000, 0xCDE703),
    "CRC-32 (IEEE 802.3)": (32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xCBF43926),
    "CRC-32C (Castagnoli)": (32, 0x1EDC6F41, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xE3069283),
}


def _reflect(v, w):
    return int(format(v, f"0{w}b")[::-1], 2)


def crc_bits(data: bytes, width, poly, init=0, refin=False, refout=False, xorout=0):
    """Bit-at-a-time CRC in the 'Rocksoft model' (Williams, 1993) parameterisation."""
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    reg = init
    for byte in data:
        if refin:
            byte = _reflect(byte, 8)
        for i in range(7, -1, -1):
            bit = (byte >> i) & 1
            fb = ((reg & top) != 0) ^ bit
            reg = (reg << 1) & mask
            if fb:
                reg ^= poly
    if refout:
        reg = _reflect(reg, width)
    return reg ^ xorout


def _selftest():
    rng = np.random.default_rng(1)
    for name, (w, p, i, ri, ro, xo, chk) in CRC_CATALOG.items():
        assert crc_bits(b"123456789", w, p, i, ri, ro, xo) == chk, name
    F = GF(8)
    rs = ReedSolomon(255, 223, F)
    for trial in range(20):
        msg = rng.integers(0, 256, 223)
        c = rs.encode(msg)
        assert not any(rs.syndromes(c))
        r = c.copy()
        ne = trial % 17
        pos = rng.choice(255, ne, replace=False)
        r[pos] ^= rng.integers(1, 256, ne)
        mh, nc = rs.decode(r)
        assert np.array_equal(mh, msg) and nc == ne, (trial, ne, nc)
    # erasures + errors
    r = c.copy(); er = list(rng.choice(255, 10, replace=False)); r[er] ^= 7
    rest = [p for p in range(255) if p not in er][:11]; r[rest] ^= 3
    mh, nc = rs.decode(r, erasures=er)
    assert np.array_equal(mh, msg), "erasure decoding"
    rs2 = ReedSolomon(204, 188, F)                  # shortened DVB code
    msg = rng.integers(0, 256, 188); c = rs2.encode(msg); r = c.copy(); r[[0, 50, 100, 150, 180, 203, 7, 9]] ^= 1
    assert np.array_equal(rs2.decode(r)[0], msg)
    b = BCH(6, 3)
    assert (b.n, b.k) == (63, 45)
    msg = rng.integers(0, 2, b.k); c = b.encode(msg); r = c.copy(); r[[3, 30, 60]] ^= 1
    assert np.array_equal(b.decode(r)[0], msg)
    lam, L = berlekamp_massey_gf2([1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    return True


if __name__ == "__main__":
    print("gf self-test:", _selftest())
