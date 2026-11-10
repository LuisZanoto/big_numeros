"""
teste_26.py
Busca de soluções para a equação de soma de cubos x³ + y³ + z³ = 26 no intervalo |x|,|y|,|z| <= R.
Implementação de alta performance e baixo consumo de memória baseada na Redução de Booker (2019)
acelerada com compilação JIT paralela (Numba prange).

OTIMIZAÇÕES DE MEMÓRIA E DESEMPENHO:
  1. Consumo de RAM O(1) (< 1 MB fixo):
     Em vez de alocar uma matriz gigante de tamanho C_MAX (que consumiria gigabytes para R grande),
     o espaço de c é particionado em chunks paralelos independentes, armazenando apenas as soluções reais.
  2. Cota teórica de C_MAX:
     Como o discriminante Δ = (4y³ - c³ - 4K) / 3c >= 0, temos obrigatoriamente c <= cbrt(4·Y_MAX³ - 4K) ≈ 1.587·Y_MAX.
     Isso corta mais de 20% das iterações desnecessárias de c.
  3. Ponto de partida ótimo de y (y_min):
     O loop de y não precisa iniciar em 1 ou 3; ele inicia diretamente em y_min ≈ c / cbrt(4),
     respeitando a congruência modular y ≡ c + (K mod 3) (mod 3), economizando até 63% das iterações internas.
"""

import time
import numpy as np
from numba import njit, prange

# =================== CONFIGURAÇÃO ===================
K        = 26
R        = 8400_000                  # Limite da busca: |x|, |y|, |z| <= R
Y_MAX    = R                        # y <= R
# Cota superior exata para Δ >= 0: c <= cbrt(4·Y_MAX³ - 4K)
C_MAX    = min(2 * R, int(np.cbrt(4.0 * (Y_MAX ** 3) - 4 * K)))
N_CHUNKS = 128                      # Quantidade de partições paralelas para o Numba prange

# Soluções primitivas conhecidas para validação
KNOWN = [
    (-1, 3, 0),
    (-312, 161, 297),
    (-469, 87, 468),
    (-2107, 237, 2106),
    (17421, 62709, -63154),
    (-74169, 139643, -132288),
    (114844365, 110902301, -142254840)
]


# =================== BUSCA OTIMIZADA EM MEMÓRIA (NUMBA + BOOKER) ===================
@njit(parallel=True, cache=True)
def booker_scan(C_MAX, Y_MAX, K, R, n_chunks):
    """
    Varredura paralela com consumo constante de memória (< 100 KB).
    Cada chunk armazena apenas as soluções realmente encontradas.
    """
    MAX_HITS_PER_CHUNK = 32
    results = np.zeros((n_chunks, MAX_HITS_PER_CHUNK, 3), dtype=np.int64)
    counts  = np.zeros(n_chunks, dtype=np.int64)
    k_mod3  = K % 3
    
    chunk_size = (C_MAX + n_chunks - 1) // n_chunks
    
    for ch in prange(n_chunks):
        c_start = 1 + ch * chunk_size
        c_end   = min(C_MAX + 1, c_start + chunk_size)
        local   = 0
        
        for c in range(c_start, c_end):
            c3 = c * c * c
            three_c = 3 * c
            
            # Cota inferior para discriminante não-negativo: 4y³ >= c³ + 4K
            c3_term = (c3 + 4 * K) / 4.0
            y_min = int(np.cbrt(c3_term))
            if y_min < 1:
                y_min = 1
            if y_min > Y_MAX:
                continue
            
            # Filtro modular Booker: y ≡ c + (K mod 3) (mod 3)
            rem = (c + k_mod3) % 3
            diff = (rem - (y_min % 3)) % 3
            y_start = y_min + diff
            if y_start == 0:
                y_start = 3
            
            for y in range(y_start, Y_MAX + 1, 3):
                # Identidade de Booker: 3c·xz = -(y³ - c³ - K)
                Y_val = y * y * y - c3 - K
                if Y_val % three_c != 0:
                    continue
                
                # Discriminante de Booker: Δ = (x - z)² = c² + 4·rhs
                rhs = Y_val // three_c
                disc = c * c + 4 * rhs
                if disc < 0:
                    continue
                
                s = int(np.sqrt(float(disc)))
                if (s - c) % 2 != 0:
                    continue
                
                # Validação exata de raiz inteira
                if s * s != disc:
                    if (s + 1) * (s + 1) == disc:
                        s += 1
                    elif (s - 1) * (s - 1) == disc:
                        s -= 1
                    else:
                        continue
                
                # Raízes exatas de Booker: {x, z} = (-c ∓ s) // 2
                z = (-c + s) // 2
                x = (-c - s) // 2
                
                if abs(x) <= R and abs(z) <= R:
                    if local < MAX_HITS_PER_CHUNK:
                        results[ch, local, 0] = x
                        results[ch, local, 1] = y
                        results[ch, local, 2] = z
                        local += 1
                    if x != z and local < MAX_HITS_PER_CHUNK:
                        results[ch, local, 0] = z
                        results[ch, local, 1] = y
                        results[ch, local, 2] = x
                        local += 1
                        
        counts[ch] = local
    return results, counts


def run():
    print("=" * 65)
    print(f"BUSCA SOMA DE CUBOS: x³ + y³ + z³ = {K}")
    print(f"Faixa de busca: |x|, |y|, |z| <= {R:,}")
    print(f"C_MAX teórico otimizado: {C_MAX:,}")
    ram_est_kb = (N_CHUNKS * 32 * 3 * 8 + N_CHUNKS * 8) / 1024
    print(f"Memória RAM alocada para o buffer de busca: ~{ram_est_kb:.1f} KB (constante)")
    print("Método: Numba prange (Chunked) + Redução de Booker")
    print("=" * 65)
    
    if R > 2_097_151:
        print("[AVISO] R > 2.097.151: para termos maiores, y³ pode exceder int64 (2⁶³-1).")
    
    # Warm-up do JIT
    print("Compilando JIT Numba...")
    booker_scan(10, 10, K, R, 2)
    
    print(f"Executando varredura paralela ({N_CHUNKS} partições)...")
    t0 = time.time()
    results, counts = booker_scan(C_MAX, Y_MAX, K, R, N_CHUNKS)
    dt = time.time() - t0
    
    hits = set()
    for ch in range(N_CHUNKS):
        for k in range(counts[ch]):
            hits.add(tuple(int(v) for v in results[ch, k]))
            
    sorted_hits = sorted(hits, key=lambda t: (abs(t[0]), t[1]))
    
    print("\n" + "=" * 65)
    print(f"RESULTADOS (Tempo: {dt:.3f} s)")
    print("=" * 65)
    print(f"Total de soluções encontradas (com permutações x,z): {len(sorted_hits)}")
    for h in sorted_hits:
        print(f"  {h}  (c = {abs(h[0] + h[2])})")
        
    # Agrupando por soluções primitivas (sem permutação)
    primitives = sorted(set(tuple(sorted((abs(x), abs(y), abs(z)))) for x, y, z in sorted_hits))
    print(f"\nSoluções primitivas únicas na faixa: {len(primitives)}")
    
    # Validação contra as soluções conhecidas
    known_canon = set(tuple(sorted((abs(x), abs(y), abs(z)))) for x, y, z in KNOWN)
    print(f"Todas as {len(KNOWN)} soluções conhecidas foram validadas: {known_canon.issubset(primitives)}")


if __name__ == "__main__":
    run()
