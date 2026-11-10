import matplotlib.pyplot as plt


def gerar_trajetoria(n):
  """Gera a lista com todos os valores da sequência de Collatz até chegar a 1."""
  trajetoria = [n]
  atual = n

  while atual > 1:
    if atual % 2 == 0:
      atual //= 2
    else:
      atual = 3 * atual + 1
    trajetoria.append(atual)

  return trajetoria


# --- ESCOLHA O NÚMERO A SER TESTADO AQUI ---
numero_escolhido = 27

# Calculando a trajetória
historico = gerar_trajetoria(numero_escolhido)
total_passos = len(historico) - 1
pico_maximo = max(historico)

print(f"Número inicial: {numero_escolhido}")
print(f"Total de etapas: {total_passos}")
print(f"Maior valor atingido (Pico): {pico_maximo}")

# --- GERANDO O GRÁFICO DA CURVA ---
plt.figure(figsize=(12, 6))
plt.plot(
    historico,
    marker="o",
    markersize=3,
    linestyle="-",
    color="purple",
    linewidth=1.5,
)

# Destacando o ponto mais alto (pico) no gráfico
indice_pico = historico.index(pico_maximo)
plt.annotate(
    f"Pico: {pico_maximo}",
    xy=(indice_pico, pico_maximo),
    xytext=(indice_pico + 2, pico_maximo),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    fontsize=10,
    weight="bold",
)

plt.title(
    f"Trajetória de Collatz para o número {numero_escolhido} ({total_passos}"
    " passos)",
    fontsize=14,
)
plt.xlabel("Número da Etapa (Passo)", fontsize=12)
plt.ylabel("Valor Atual", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()