"""
Demo: achar (x,y,z) com x³+y³+z³=33 em faixa pequena ao redor da solução
conhecida. Compara:
  - CPU: multiprocessing + big-int puro
  - GPU: CUDA (se disponível) ou Numba-prange (fallback em i5 sem GPU)
"""
import math, time, multiprocessing
import numpy as np

try:
    from numba import njit, prange, cuda
    NUMBA_OK = True
    CUDA_OK  = cuda.is_available()
except ImportError:
    NUMBA_OK = False
    CUDA_OK  = False

# =================== CONFIGURAÇÃO ===================
K = 42
KNOWN_X =   8_866_128_975_287_528
KNOWN_Y =  -8_778_405_442_862_239
KNOWN_Z =  -2_736_111_468_807_040
KNOWN_D = KNOWN_X + KNOWN_Y          # = 87_723_532_425_289  (x+y)

Z_RADIUS = 500          # ±500 → 1001 valores de z
D_RADIUS = 900_000     # ±900k → 1800.001 valores de d por z

Z_LO, Z_HI = KNOWN_Z - Z_RADIUS, KNOWN_Z + Z_RADIUS + 1
D_LO, D_HI = KNOWN_D - D_RADIUS, KNOWN_D + D_RADIUS + 1

print(f"Faixa z ∈ [{Z_LO:,}, {Z_HI:,})  →  {Z_HI-Z_LO} valores")
print(f"Faixa d ∈ [{D_LO:,}, {D_HI:,})  →  {D_HI-D_LO:,} valores por z")
print(f"Total de pares (z,d): {(Z_HI-Z_LO)*(D_HI-D_LO):,}")
print(f"Numba: {NUMBA_OK}   CUDA: {CUDA_OK}")

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

# =================== TESTE 1: CPU (multiprocessing) ===================
def cpu_worker(args):
    z_list, d_lo, d_hi = args
    hits = []
    for z in z_list:
        N = K - z**3
        for d in range(d_lo, d_hi):
            if N % d != 0:
                continue
            r = verify(z, d)
            if r:
                hits.append(r)
    return hits

def run_cpu():
    print("\n" + "="*62)
    print("TESTE 1 — CPU  (multiprocessing, big-int puro)")
    print("="*62)
    cores = multiprocessing.cpu_count()
    print(f"Núcleos lógicos: {cores}")
    z_list = list(range(Z_LO, Z_HI))
    n_chunks = min(cores, len(z_list))
    chunks = [(z_list[i::n_chunks], D_LO, D_HI) for i in range(n_chunks)]

    t0 = time.time()
    with multiprocessing.Pool(n_chunks) as pool:
        results = pool.map(cpu_worker, chunks)
    dt = time.time() - t0
    hits = [h for sub in results for h in sub]
    print(f"Tempo CPU: {dt:.3f} s")
    print(f"Soluções: {hits}")
    return dt, hits

# =================== TESTE 2: GPU (CUDA) ou Numba-CPU ===================
if NUMBA_OK:
    NLIMBS = 11  # 11*16 = 176 bits > log2(2e46) ≈ 153 bits  ✓

    def to_limbs(N):
        """N (>=0) → array uint16 little-endian."""
        arr = np.zeros(NLIMBS, dtype=np.uint16)
        for i in range(NLIMBS):
            arr[i] = N & 0xFFFF
            N >>= 16
        return arr

    # Fallback CPU-paralelo (prange) — mesmo algoritmo vetorizado da GPU
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

    # Kernel CUDA (se houver GPU NVIDIA)
    if CUDA_OK:
        @cuda.jit
        def scan_cuda(limbs, d_lo, n_d, mask):
            i = cuda.grid(1)
            if i >= n_d:
                return
            d = d_lo + i
            r = 0
            for j in range(limbs.shape[0]-1, -1, -1):
                r = (r * 65536 + int(limbs[j])) % d
            if r == 0:
                mask[i] = 1

    def run_parallel(use_cuda):
        label = "GPU (CUDA)" if use_cuda else "CPU (Numba prange — estilo GPU)"
        print("\n" + "="*62)
        print(f"TESTE 2 — {label}")
        print("="*62)

        # warm-up (compilação JIT)
        warm = np.zeros(8, dtype=np.uint16); warm[0] = 1
        dummy = np.zeros(8, dtype=np.uint8)
        if use_cuda:
            scan_cuda[1, 8](warm, 1, 8, dummy); cuda.synchronize()
        else:
            scan_parallel(warm, 1, 9, dummy)

        t0 = time.time()
        hits = []
        n_d = D_HI - D_LO
        for z in range(Z_LO, Z_HI):
            N = K - z**3
            limbs = to_limbs(N)              # N > 0 sempre nesta faixa
            mask = np.zeros(n_d, dtype=np.uint8)

            if use_cuda:
                threads = 256
                blocks = (n_d + threads - 1) // threads
                scan_cuda[blocks, threads](limbs, D_LO, n_d, mask)
                cuda.synchronize()
            else:
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
    cpu_time, cpu_hits = run_cpu()

    gpu_time = gpu_hits = None
    if NUMBA_OK:
        gpu_time, gpu_hits = run_parallel(use_cuda=CUDA_OK)

        print("\n" + "="*62)
        print("COMPARAÇÃO")
        print("="*62)
        print(f"CPU multiprocessing      : {cpu_time:8.3f} s   hits={len(cpu_hits)}")
        backend = "GPU CUDA" if CUDA_OK else "Numba CPU prange"
        print(f"{backend:25}: {gpu_time:8.3f} s   hits={len(gpu_hits)}")
        if gpu_time > 0:
            print(f"Speedup                  : {cpu_time/gpu_time:6.1f}×")
    else:
        print("\nNumba não instalado — só teste CPU executado.")
        print("Instale com:  pip install numba")

    expected = (KNOWN_X, KNOWN_Y, KNOWN_Z)
    print(f"\nSolução de Booker (2019): {expected}")
    print(f"Encontrada na CPU:        {expected in cpu_hits}")
    if gpu_hits is not None:
        print(f"Encontrada no 2º teste:   {expected in gpu_hits}")