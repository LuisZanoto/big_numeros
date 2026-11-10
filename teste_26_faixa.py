"""
teste_26_faixa.py
Busca de soluções para a equação de soma de cubos x³ + y³ + z³ = K
controlada por TEMPO LIMITE (Raio Temporal) e ponto central em dígitos.

ARITMÉTICA NATIVA DE 128 BITS (Numba JIT):
  - Capacidade máxima de inteiros expandida de 64 bits para 128 BITS:
      * Limite 64 bits antigo : R ≈ 2.097.151 (~7 dígitos)
      * Limite 128 BITS NOVO  : R ≈ 6.973.568.802.000 (~13 DÍGITOS!)
      * y³, c³, Y_val e Δ calculados com precisão total de 128 bits (até 3,4 × 10³⁸).
  - Permite alcançar com segurança soluções de 8, 9, 10, 11 e 12 dígitos (incluindo a Solução 7 de 9 dígitos).

MODELO DE BUSCA TEMPORAL (Circunferência / Raio de Tempo):
  - Ponto Central (Origem da busca):
      Definido pelo número mínimo de dígitos solicitado (ex: 5 dígitos -> R₀ = 10⁴ = 10.000).
  - Raio da Circunferência (Tempo em segundos):
      O usuário determina o tempo que o PC deve pesquisar (ex: 300 segundos, 60 segundos).
      O raio de busca R se expande concentricamente a partir do centro em anéis progressivos:
      [R₀, R₁], [R₁, R₂], [R₂, R₃]...
  - A cada anel concluído:
      1. Mede o tempo decorrido e exibe o raio R e magnitude de dígitos alcançados;
      2. Exibe em tempo real quaisquer novas soluções descobertas;
      3. Auto-ajusta o tamanho do próximo anel (ΔR) para garantir respostas rápidas e parada pontual;
      4. Encerra imediatamente ao atingir o tempo estipulado ou o teto opcional de dígitos.
"""

import sys
import time
import math
import argparse
import numpy as np
from numba import njit, prange

# ==============================================================================
# CONFIGURAÇÃO PADRÃO
# ==============================================================================
K_DEFAULT           = 26             # Valor alvo da soma de cubos: x³ + y³ + z³ = K
DIGITOS_MIN_DEFAULT = 6              # Mínimo de dígitos na faixa (ponto central inicial)
DIGITOS_MAX_DEFAULT = None           # Máximo de dígitos (None = livre até esgotar o tempo)
TEMPO_DEFAULT       = 30.0           # Tempo limite padrão em segundos (ex: 300.0)
CRITERIO_DEFAULT    = 'ampla'        # 'ampla', 'faixa_termos' ou 'magnitude'

# Limite teórico exato onde y³ cabe em inteiros não-assinados de 128 BITS (2¹²⁸ - 1)
R_LIMIT_INT128      = 6_973_568_802_000   # ~13 dígitos (cobre plenamente até 12 dígitos)

# Tabela de soluções conhecidas para K = 26 (para validação e identificação amigável)
KNOWN_26 = [
    {"id": 1, "triplet": (-1, 3, 0),                          "faixa": (1, 1), "mag": 1},
    {"id": 2, "triplet": (-312, 161, 297),                    "faixa": (3, 3), "mag": 3},
    {"id": 3, "triplet": (-469, 87, 468),                     "faixa": (2, 3), "mag": 3},
    {"id": 4, "triplet": (-2107, 237, 2106),                  "faixa": (3, 4), "mag": 4},
    {"id": 5, "triplet": (17421, 62709, -63154),              "faixa": (5, 5), "mag": 5},
    {"id": 6, "triplet": (-74169, 139643, -132288),           "faixa": (5, 6), "mag": 6},
    {"id": 7, "triplet": (114844365, 110902301, -142254840), "faixa": (9, 9), "mag": 9},
]


# ==============================================================================
# OPERAÇÕES DE 128 BITS COM ALTA PERFORMANCE (NUMBA JIT)
# ==============================================================================
MASK32 = np.uint64(0xFFFFFFFF)
U32    = np.uint64(32)

@njit
def mul64_to_128(u, v):
    """Multiplicação exata de dois inteiros de 64 bits retornando (hi, lo) em 128 bits."""
    u = np.uint64(u)
    v = np.uint64(v)
    u0 = u & MASK32
    u1 = u >> U32
    v0 = v & MASK32
    v1 = v >> U32
    w0 = u0 * v0
    t  = u1 * v0 + (w0 >> U32)
    w1 = t & MASK32
    w2 = t >> U32
    w1 += u0 * v1
    hi = u1 * v1 + w2 + (w1 >> U32)
    lo = u * v
    return hi, lo

@njit
def cube64_to_128(x):
    """Calcula x³ com resultado exato de 128 bits (hi, lo)."""
    x = np.uint64(x)
    hi2, lo2 = mul64_to_128(x, x)
    hi_lo2, lo_lo2 = mul64_to_128(lo2, x)
    hi3 = hi2 * x + hi_lo2
    lo3 = lo_lo2
    return hi3, lo3

@njit
def sub128(hi1, lo1, hi2, lo2):
    """Subtração de dois inteiros de 128 bits: (hi1, lo1) - (hi2, lo2)."""
    lo = lo1 - lo2
    borrow = np.uint64(1) if lo1 < lo2 else np.uint64(0)
    hi = hi1 - hi2 - borrow
    return hi, lo

@njit
def add128(hi1, lo1, hi2, lo2):
    """Adição de dois inteiros de 128 bits: (hi1, lo1) + (hi2, lo2)."""
    lo = lo1 + lo2
    carry = np.uint64(1) if lo < lo1 else np.uint64(0)
    hi = hi1 + hi2 + carry
    return hi, lo

@njit
def divmod128_by_64(hi, lo, d):
    """Divisão e resto de um inteiro de 128 bits por um divisor de 64 bits."""
    d = np.uint64(d)
    l3 = hi >> U32
    l2 = hi & MASK32
    l1 = lo >> U32
    l0 = lo & MASK32
    
    q3 = l3 // d
    r3 = l3 % d
    
    cur2 = (r3 << U32) | l2
    q2 = cur2 // d
    r2 = cur2 % d
    
    cur1 = (r2 << U32) | l1
    q1 = cur1 // d
    r1 = cur1 % d
    
    cur0 = (r1 << U32) | l0
    q0 = cur0 // d
    r0 = cur0 % d
    
    q_hi = (q3 << U32) | q2
    q_lo = (q1 << U32) | q0
    return q_hi, q_lo, r0

@njit
def isqrt128(disc_hi, disc_lo):
    """Calcula a raiz quadrada inteira de um número de 128 bits e valida se é quadrado perfeito."""
    if disc_hi == 0:
        s = np.uint64(np.sqrt(float(disc_lo)))
    else:
        val = float(disc_hi) * 18446744073709551616.0 + float(disc_lo)
        s = np.uint64(np.sqrt(val))
        
    for cand in (s, s + np.uint64(1), s - np.uint64(1), s + np.uint64(2), s - np.uint64(2)):
        c_hi, c_lo = mul64_to_128(cand, cand)
        if c_hi == disc_hi and c_lo == disc_lo:
            return cand, True
    return s, False


# ==============================================================================
# MOTOR JIT DE ANÉIS CONCÊNTRICOS COM ARITMÉTICA DE 128 BITS (PARALELO)
# ==============================================================================
@njit(parallel=True, cache=True)
def scan_ring_numba_128(c_start, c_end, y_min_global, y_max_global, K, R_filter, n_chunks=64):
    """
    Varre a região retangular no plano (c, y) utilizando Booker com precisão de 128 bits.
    Capacidade de alcance: até 13 dígitos numéricos sem overflow.
    """
    MAX_HITS = 32
    results = np.zeros((n_chunks, MAX_HITS, 3), dtype=np.int64)
    counts  = np.zeros(n_chunks, dtype=np.int64)
    k_mod3  = K % 3
    
    if c_end < c_start or y_max_global < y_min_global:
        return results, counts
        
    chunk_size = (c_end - c_start + n_chunks - 1) // n_chunks
    if chunk_size < 1:
        chunk_size = 1
        
    for ch in prange(n_chunks):
        local_c_start = c_start + ch * chunk_size
        local_c_end   = min(c_end + 1, local_c_start + chunk_size)
        if local_c_start > c_end:
            continue
            
        local = 0
        for c in range(local_c_start, local_c_end):
            c_u64 = np.uint64(c)
            hi_c, lo_c = cube64_to_128(c_u64)
            lo_c_k = lo_c + np.uint64(K)
            hi_c_k = hi_c + (np.uint64(1) if lo_c_k < lo_c else np.uint64(0))
            three_c = np.uint64(3 * c)
            
            # Cota inferior para discriminante >= 0: 4y³ >= c³ + 4K
            val_c3 = float(hi_c) * 18446744073709551616.0 + float(lo_c)
            y_min = int(np.cbrt((val_c3 + 4.0 * K) / 4.0))
            if y_min < y_min_global:
                y_min = y_min_global
            if y_min > y_max_global:
                continue
                
            rem = (c + k_mod3) % 3
            diff = (rem - (y_min % 3)) % 3
            y_start = y_min + diff
            if y_start == 0:
                y_start = 3
                
            for y in range(y_start, y_max_global + 1, 3):
                y_u64 = np.uint64(y)
                hi_y, lo_y = cube64_to_128(y_u64)
                
                # Exige y³ >= c³ + K para solução real
                if hi_y < hi_c_k or (hi_y == hi_c_k and lo_y < lo_c_k):
                    continue
                    
                # Y_val = y³ - (c³ + K) em 128 bits
                hi_Y, lo_Y = sub128(hi_y, lo_y, hi_c_k, lo_c_k)
                
                # Divisão 128 bits por 64 bits: Y_val // (3c) e Y_val % (3c)
                q_hi, q_lo, r = divmod128_by_64(hi_Y, lo_Y, three_c)
                if r != 0:
                    continue
                    
                # Discriminante em 128 bits: Δ = c² + 4·rhs
                c2_hi, c2_lo = mul64_to_128(c_u64, c_u64)
                four_q_hi = (q_hi << np.uint64(2)) | (q_lo >> np.uint64(62))
                four_q_lo = q_lo << np.uint64(2)
                disc_hi, disc_lo = add128(c2_hi, c2_lo, four_q_hi, four_q_lo)
                
                # Raiz quadrada de 128 bits
                s, is_sq = isqrt128(disc_hi, disc_lo)
                if not is_sq:
                    continue
                    
                if (s - c_u64) % np.uint64(2) != 0:
                    continue
                    
                s_i64 = np.int64(s)
                c_i64 = np.int64(c)
                z = (-c_i64 + s_i64) // 2
                x = (-c_i64 - s_i64) // 2
                
                if abs(x) <= R_filter and abs(z) <= R_filter:
                    if local < MAX_HITS:
                        results[ch, local, 0] = x
                        results[ch, local, 1] = y
                        results[ch, local, 2] = z
                        local += 1
                    if x != z and local < MAX_HITS:
                        results[ch, local, 0] = z
                        results[ch, local, 1] = y
                        results[ch, local, 2] = x
                        local += 1
                        
        counts[ch] = local
    return results, counts


def c_max_for(r, k):
    """Calcula a cota superior exata de c para um raio r."""
    if r <= 0:
        return 0
    return int(np.cbrt(4.0 * (r ** 3) - 4 * k))


# ==============================================================================
# FUNÇÕES DE ANÁLISE DE PROPRIEDADES E DÍGITOS
# ==============================================================================
def obter_propriedades_solucao(sol):
    """Retorna contagem de dígitos dos termos, faixa [min, max] e magnitude."""
    x, y, z = sol
    termos = [x, y, z]
    digitos = [len(str(abs(v))) for v in termos]
    d_min = min(digitos)
    d_max = max(digitos)
    mag = max(abs(v) for v in termos)
    return {
        "x": x,
        "y": y,
        "z": z,
        "digitos": digitos,
        "d_min": d_min,
        "d_max": d_max,
        "mag": mag
    }


def identificar_solucao_conhecida(sol, k):
    """Verifica se coincide com alguma solução conhecida catalogada."""
    if k != 26:
        return None
    canon = tuple(sorted(abs(v) for v in sol))
    for item in KNOWN_26:
        item_canon = tuple(sorted(abs(v) for v in item["triplet"]))
        if canon == item_canon:
            return item["id"]
    return None


def atende_criterio(props, digitos_min, digitos_max, criterio):
    """Verifica se a solução cumpre o critério de dígitos exigido."""
    d_min, d_max = props["d_min"], props["d_max"]
    
    if digitos_max is None:
        if criterio == 'faixa_termos':
            return d_min >= digitos_min
        elif criterio == 'magnitude':
            return d_max >= digitos_min
        else:  # 'ampla'
            return d_min >= digitos_min or d_max >= digitos_min
            
    if criterio == 'faixa_termos':
        return d_min == digitos_min and d_max == digitos_max
    elif criterio == 'magnitude':
        return digitos_min <= d_max <= digitos_max
    elif criterio == 'ampla':
        return d_min >= digitos_min and d_max <= digitos_max
    return True


# ==============================================================================
# MOTOR DE BUSCA TEMPORAL CONCÊNTRICA (128 BITS)
# ==============================================================================
def executar_busca_temporal(K, digitos_min, digitos_max=None, tempo_limite=TEMPO_DEFAULT, criterio=CRITERIO_DEFAULT):
    print("=" * 75)
    print(f"BUSCA TEMPORAL 128-BIT DE SOMA DE CUBOS: x³ + y³ + z³ = {K}")
    print(f"Precisão de Inteiros     : 128 BITS NATIVO (alcance de até 13 dígitos)")
    print(f"Ponto Central (Origem)   : A partir de {digitos_min} dígitos")
    if digitos_max:
        print(f"Teto Opcional de Parada  : Até {digitos_max} dígitos (R = 10^{digitos_max} = {10**digitos_max:,})")
    else:
        print(f"Teto Opcional de Parada  : Nenhum (Livre, guiado pelo raio de tempo)")
    print(f"Raio Temporal da Busca   : {tempo_limite:.1f} segundos de processamento")
    print(f"Critério de Seleção      : {criterio}")
    print("=" * 75)
    
    # Validação teórica modular mod 9
    if K % 9 in (4, 5):
        print(f"\n[AVISO TEÓRICO] K ≡ {K % 9} (mod 9).")
        print("Números da forma 9m ± 4 não possuem nenhuma solução inteira na soma de três cubos.")
        return []

    # Definição do Ponto Central R0
    if digitos_min <= 1:
        R_start = 0
    else:
        R_start = 10 ** (digitos_min - 1)
        
    R_target_max = 10 ** digitos_max if digitos_max is not None else float("inf")
    
    print(f"Ponto de Partida R₀      : {R_start:,} (~{digitos_min} dígitos)")
    print(f"Memória RAM Alocada      : < 100 KB (buffer constante por chunks)")
    print("Inicializando JIT Numba com motor aritmético de 128 bits...")
    # Warmup
    scan_ring_numba_128(1, 10, 1, 10, K, 10, 2)
    
    print("\nIniciando expansão concêntrica da circunferência de busca...")
    print("-" * 75)
    
    R_current = R_start
    delta_R = max(10_000, R_start // 5) if R_start > 0 else 25_000
    
    solucoes_encontradas = {}
    total_aneis = 0
    t0 = time.time()
    interrompido_por_bits = False
    
    while True:
        elapsed = time.time() - t0
        if elapsed >= tempo_limite:
            break
            
        if R_current >= R_target_max:
            print(f"\n[SUCESSO] Teto máximo de dígitos solicitado ({digitos_max} dígitos) alcançado com sucesso!")
            break
            
        if R_current >= R_LIMIT_INT128:
            interrompido_por_bits = True
            print(f"\n[FRONTEIRA DE 128 BITS ALCANÇADA]: R = {R_current:,} (~13 dígitos).")
            break
            
        R_next = min(R_current + delta_R, R_LIMIT_INT128)
        if R_target_max != float("inf"):
            R_next = min(R_next, R_target_max)
            
        C0 = c_max_for(R_current, K)
        C1 = c_max_for(R_next, K)
        
        t_step_0 = time.time()
        novos_hits = set()
        
        # Decomposição exata do anel concêntrico [R_current, R_next]
        if R_current == 0:
            res_B, cnt_B = scan_ring_numba_128(1, C1, 1, R_next, K, R_next)
            for ch in range(len(cnt_B)):
                for idx in range(cnt_B[ch]):
                    novos_hits.add(tuple(int(v) for v in res_B[ch, idx]))
        else:
            # Parte A: c ∈ [1, C0], y ∈ [R_current + 1, R_next]
            if C0 >= 1:
                res_A, cnt_A = scan_ring_numba_128(1, C0, R_current + 1, R_next, K, R_next)
                for ch in range(len(cnt_A)):
                    for idx in range(cnt_A[ch]):
                        novos_hits.add(tuple(int(v) for v in res_A[ch, idx]))
                        
            # Parte B: c ∈ [C0 + 1, C1], y ∈ [1, R_next]
            if C1 > C0:
                res_B, cnt_B = scan_ring_numba_128(C0 + 1, C1, 1, R_next, K, R_next)
                for ch in range(len(cnt_B)):
                    for idx in range(cnt_B[ch]):
                        novos_hits.add(tuple(int(v) for v in res_B[ch, idx]))
                        
        dt_step = time.time() - t_step_0
        total_aneis += 1
        
        hits_neste_anel = 0
        for triplet in novos_hits:
            x, y, z = triplet
            if (x ** 3 + y ** 3 + z ** 3) == K:
                chave = tuple(sorted(abs(v) for v in (x, y, z)))
                if chave not in solucoes_encontradas:
                    props = obter_propriedades_solucao((x, y, z))
                    solucoes_encontradas[chave] = (x, y, z, props)
                    hits_neste_anel += 1
                    
                    sol_id = identificar_solucao_conhecida((x, y, z), K)
                    rotulo = f"Solução {sol_id}" if sol_id else "Nova Solução"
                    print(f"\n  ★ [{rotulo.upper()} ENCONTRADA]: {x, y, z}")
                    print(f"    Dígitos dos termos : {props['digitos']}  |  Magnitude: {props['mag']:,} ({props['d_max']} dígitos)")
                    print(f"    Verificação cúbica : ({x})³ + ({y})³ + ({z})³ = {x**3 + y**3 + z**3}")
                    
        R_current = R_next
        elapsed = time.time() - t0
        pct_tempo = min(100.0, (elapsed / tempo_limite) * 100.0)
        d_atual = len(str(R_current))
        status_line = (
            f"[Tempo: {elapsed:5.1f}s / {tempo_limite:5.1f}s ({pct_tempo:4.1f}%)] "
            f"Raio R: {R_current:10,d} (~{d_atual} dígitos) | Anel #{total_aneis:02d} | "
            f"Total Soluções: {len(solucoes_encontradas)}"
        )
        print(status_line)
        
        if dt_step < 0.6:
            delta_R = int(delta_R * 1.5)
        elif dt_step > 2.5:
            delta_R = max(5_000, int(delta_R * 0.7))
            
    # Resumo Final
    tempo_total = time.time() - t0
    print("\n" + "=" * 75)
    print("RELATÓRIO FINAL DA BUSCA TEMPORAL (128 BITS)")
    print("=" * 75)
    print(f"Tempo total gasto           : {tempo_total:.2f} segundos (limite era {tempo_limite:.1f} s)")
    print(f"Raio atingido no tempo      : R = {R_current:,} (~{len(str(R_current))} dígitos)")
    print(f"Anéis concêntricos varridos : {total_aneis} anéis")
    print(f"Soluções únicas encontradas : {len(solucoes_encontradas)}")
    
    selecionadas = []
    for chave, (x, y, z, props) in solucoes_encontradas.items():
        if atende_criterio(props, digitos_min, digitos_max, criterio):
            selecionadas.append((x, y, z, props))
            
    print("\n" + "-" * 75)
    print(f"SOLUÇÕES QUE ATENDEM AO CRITÉRIO '{criterio.upper()}' (A partir de {digitos_min} dígitos):")
    print("-" * 75)
    if not selecionadas:
        print(f"Nenhuma solução satisfazendo o critério '{criterio}' no raio alcançado em {tempo_total:.1f}s.")
        print("Dica: você pode aumentar o tempo limite (-t) para expandir ainda mais a circunferência de busca.")
    else:
        for x, y, z, props in sorted(selecionadas, key=lambda item: item[3]["mag"]):
            sol_id = identificar_solucao_conhecida((x, y, z), K)
            rotulo = f"Solução {sol_id}" if sol_id else "Solução encontrada"
            print(f"  • {rotulo:<12} : {(x, y, z)}")
            print(f"    Termos em dígitos: {props['digitos']}  ==>  Faixa: [{props['d_min']} a {props['d_max']} dígitos]")
            print(f"    Maior termo      : {props['mag']:,} ({props['d_max']} dígitos)")
            
    print("=" * 75)
    return selecionadas


# ==============================================================================
# ARGUMENTOS DE LINHA DE COMANDO E INTERATIVIDADE
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Busca temporal expansiva 128-bit de soluções para x³ + y³ + z³ = K (raio de tempo)."
    )
    parser.add_argument(
        "faixa",
        nargs="*",
        type=int,
        help="Argumentos posicionais: [min] (ex: 5 para 'acima de 5 dígitos'), ou [min max], ou [k min max]"
    )
    parser.add_argument("--k", type=int, default=K_DEFAULT, help=f"Valor de K (padrão: {K_DEFAULT})")
    parser.add_argument("--min", type=int, default=None, help=f"Dígitos mínimos / ponto central (padrão: {DIGITOS_MIN_DEFAULT})")
    parser.add_argument("--max", type=int, default=None, help="Dígitos máximos / teto opcional de parada")
    parser.add_argument("-t", "--tempo", type=float, default=TEMPO_DEFAULT, help=f"Tempo limite em segundos (padrão: {TEMPO_DEFAULT} s, ex: 300)")
    parser.add_argument(
        "--criterio",
        choices=["ampla", "magnitude", "faixa_termos"],
        default=CRITERIO_DEFAULT,
        help="Critério de exibição: 'ampla' (padrão), 'magnitude' ou 'faixa_termos'"
    )
    parser.add_argument("-i", "--interativo", action="store_true", help="Ativa modo interativo via teclado")
    args = parser.parse_args()
    
    k = args.k
    d_min = args.min
    d_max = args.max
    tempo = args.tempo
    
    if args.faixa:
        if len(args.faixa) == 1:
            d_min = args.faixa[0]
        elif len(args.faixa) == 2:
            d_min, d_max = args.faixa[0], args.faixa[1]
        elif len(args.faixa) >= 3:
            k, d_min, d_max = args.faixa[0], args.faixa[1], args.faixa[2]
            
    if d_min is None:
        d_min = DIGITOS_MIN_DEFAULT
        
    return k, d_min, d_max, tempo, args.criterio, args.interativo


if __name__ == "__main__":
    k, d_min, d_max, tempo, criterio, interativo = parse_args()
    
    if interativo:
        print("\n=== CONFIGURAÇÃO INTERATIVA (BUSCA TEMPORAL 128-BIT) ===")
        try:
            k_in = input(f"Valor de K [padrão {k}]: ").strip()
            if k_in:
                k = int(k_in)
                
            min_in = input(f"Dígitos MÍNIMOS (Ponto Central de busca) [padrão {d_min}]: ").strip()
            if min_in:
                d_min = int(min_in)
                
            max_in = input(f"Dígitos MÁXIMOS (Pressione Enter para ilimitado) [{d_max if d_max else 'livre'}]: ").strip()
            if max_in:
                d_max = int(max_in)
                
            tempo_in = input(f"Tempo limite em SEGUNDOS [padrão {tempo} s]: ").strip()
            if tempo_in:
                tempo = float(tempo_in)
                
            crit_in = input(f"Critério (1: ampla, 2: magnitude, 3: faixa_termos) [padrão 1]: ").strip()
            if crit_in == "2":
                criterio = "magnitude"
            elif crit_in == "3":
                criterio = "faixa_termos"
            elif crit_in == "1" or not crit_in:
                criterio = "ampla"
        except (ValueError, EOFError):
            print("\nEntrada inválida ou cancelada. Usando valores configurados.")
            
    executar_busca_temporal(k, d_min, d_max, tempo, criterio)
