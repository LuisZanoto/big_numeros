from timeit import default_timer as timer
from numba import njit
import numpy as np



# ================= MODO 2: TODAS (sem permutação) =================
@njit(cache=True)
def busca_todas(max_x, obj, cap=10000):
    """
    Retorna (results[cap,3], n, conta) — sem permutações (y ≤ z).
    """
    results = np.empty((cap, 3), dtype=np.int64)
    n = 0
    conta = 0
    for xi in range(1, max_x + 1):
        x = -xi
        x3 = x * x * x
        for y in range(1, max_x + 1):
            y3 = y * y * y
            s = x3 + y3
            if s > obj:
                break
            alvo = (obj - s) % 9
            if alvo != 0 and alvo != 1 and alvo != 8:
                continue
            z = xi - y + 1
            for _ in range(max_x):
                z3 = z * z * z
                resp = s + z3
                if resp > obj:
                    break
                conta += 1
                if resp == obj and y <= z:    # evita (y,z) e (z,y)
                    if n < cap:
                        results[n, 0] = x
                        results[n, 1] = y
                        results[n, 2] = z
                        n += 1
                z += 1
    return results, n, conta


# ================= DEMO =================
if __name__ == "__main__":
    max_x = 14000 # 10k=42 segundos em pc com i5
    obj = 26


    print("\n[Modo 2] Todas as soluções (sem permutação, y ≤ z):")
    t0 = timer()
    results, n, conta2 = busca_todas(max_x, obj)
    dt2 = timer() - t0
    for i in range(n):
        print(f"  ({results[i,0]}, {results[i,1]}, {results[i,2]})")
    print(f"  Contador: {conta2}   Hits: {n}")
    print(f"  Tempo: {dt2:.6f} s")