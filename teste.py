"""Busca soluções de soma de cubos usando exclusivamente Numba CPU prange."""
import math, time
import numpy as np

try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

# =================== CONFIGURAÇÃO ===================
#K = 33
#KNOWN_X =   8_866_128_975_287_528
#KNOWN_Y =  -8_778_405_442_862_239
#KNOWN_Z =  -2_736_111_468_807_040

# testando para k=26 (114844365, 110902301, -142254840)
K = 26
KNOWN_X =   114_844_365
KNOWN_Y =  110_902_301
KNOWN_Z =  -142_254_840


KNOWN_D = KNOWN_X + KNOWN_Y          # = 87_723_532_425_289  (x+y)

Z_RADIUS = 500          # ±500 → 1001 valores de z
D_RADIUS = 5_900_000     # ±900k → 1800.001 valores de d por z

Z_LO, Z_HI = KNOWN_Z - Z_RADIUS, KNOWN_Z + Z_RADIUS + 1
D_LO, D_HI = KNOWN_D - D_RADIUS, KNOWN_D + D_RADIUS + 1

print(f"Faixa z ∈ [{Z_LO:,}, {Z_HI:,})  →  {Z_HI-Z_LO} valores")
print(f"Faixa d ∈ [{D_LO:,}, {D_HI:,})  →  {D_HI-D_LO:,} valores por z")
print(f"Total de pares (z,d): {(Z_HI-Z_LO)*(D_HI-D_LO):,}")
print(f"Numba CPU prange: {NUMBA_OK}")

# =================== VERIFICAÇÃO EXATA ===================
def verify(z, d):
    """Dado (z,d), resolve x,y por Bhaskara e confirma x³+y³+z³=K."""
    N = K - z**3
    if N % d != 0:
        return None
    num = d**3 - N               # = 3·d·xy   (identidade d³-N)
    if num % (3*d) != 0:
        return None
    xy = num // (3*d)
    delta = d*d - 4*xy           # = (x-y)²
    if delta < 0:
        return None
    s = math.isqrt(delta)
    if s*s != delta or (d + s) % 2 != 0:
        return None
    x = (d + s) // 2
    y = (d - s) // 2
    return (x, y, z) if x**3 + y**3 + z**3 == K else None

# =================== MOTOR CPU NUMBA PRANGE ===================
if NUMBA_OK:
    NLIMBS = 11  # 11*16 = 176 bits > log2(2e46) ≈ 153 bits  ✓

    def to_limbs(N):
        """N (>=0) → array uint16 little-endian."""
        arr = np.zeros(NLIMBS, dtype=np.uint16)
        for i in range(NLIMBS):
            arr[i] = N & 0xFFFF
            N >>= 16
        return arr

    @njit(parallel=True)
    def scan_parallel(limbs, d_lo, d_hi, mask):
        n = d_hi - d_lo
        for i in prange(n):
            d = d_lo + i
            r = 0
            for j in range(limbs.shape[0]-1, -1, -1):
                r = (r * 65536 + int(limbs[j])) % d
            if r == 0:
                mask[i] = 1

    def run_numba_cpu():
        label = "CPU (Numba prange)"
        print("\n" + "="*62)
        print(f"BUSCA — {label}")
        print("="*62)

        # Warm-up compila o kernel antes da medição.
        warm = np.zeros(8, dtype=np.uint16); warm[0] = 1
        dummy = np.zeros(8, dtype=np.uint8)
        scan_parallel(warm, 1, 9, dummy)

        t0 = time.time()
        hits = []
        n_d = D_HI - D_LO
        for z in range(Z_LO, Z_HI):
            N = K - z**3
            limbs = to_limbs(N)              # N > 0 sempre nesta faixa
            mask = np.zeros(n_d, dtype=np.uint8)

            scan_parallel(limbs, D_LO, D_HI, mask)

            for d in np.arange(D_LO, D_HI, dtype=np.int64)[mask == 1]:
                r = verify(z, int(d))
                if r:
                    hits.append(r)

        dt = time.time() - t0
        print(f"Tempo {label}: {dt:.3f} s")
        print(f"Soluções: {hits}")
        return dt, hits

# =================== MAIN ===================
if __name__ == "__main__":
    if NUMBA_OK:
        elapsed, hits = run_numba_cpu()
    else:
        raise SystemExit("Numba não está instalado. Instale com: pip install numba")

    expected = (KNOWN_X, KNOWN_Y, KNOWN_Z)
    print(f"\nSolução de Booker (teste): {expected}")
    print(f"Encontrada no Numba CPU:   {expected in hits}")