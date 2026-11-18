"""Exemplo do MOFA com interações (mofa_interacao.py) em dados simulados.

Simula o mesmo cenário do exemplo_mofa_aditivo.py (respostas lineares, côncavas, sigmoides e em U) e acrescenta
interações verdadeiras: em parte das features, o efeito de um fator depende do nível de outro (termo c z_k z_l).
    rna: par F1xF2 em 8% das features e par F1xF4 em 5%
    met: par F1xF2 em 8% das features
    cnv: nenhuma interação
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import mofa_aditivo as MA, mofa_interacao as MI
from exemplo_mofa_aditivo import simula

VERDADE = {"rna": [((0, 1), 0.08), ((0, 3), 0.05)], "met": [((0, 1), 0.08)], "cnv": []}


def simula_interacoes(N=300, semente=1, forca=(0.8, 1.5)):
    V, Z, T = simula(N=N, semente=semente, frac_nl=0.5)
    rng = np.random.default_rng(semente + 100)
    inter = {}
    for m, lista in VERDADE.items():
        D = V[m].shape[1]
        inter[m] = {}
        for (k, l), frac in lista:
            idx = rng.choice(D, int(frac * D), replace=False)
            c = rng.choice([-1, 1], len(idx)) * rng.uniform(*forca, len(idx))
            V[m][:, idx] += (Z[:, k] * Z[:, l])[:, None] * c[None]
            inter[m][(k, l)] = idx
    return V, Z, T, inter


if __name__ == "__main__":
    V, Z, T, inter = simula_interacoes()
    V = {m: v - v.mean(0) for m, v in V.items()}
    res = MI.mofa_interacao(V, K=10)

    # fatores estimados -> verdadeiros (podem vir em outra ordem e sinal)
    C = np.abs(np.corrcoef(Z.T, res["Z"].T)[:4, 4:])
    est_para_vdd = {int(C[:, j].argmax()): j for j in range(res["Z"].shape[1]) if C[:, j].max() > 0.7}
    print("correlação |r| fator verdadeiro x estimado:\n", np.round(C, 2))
    print("pares que sobraram:", [MI.nome_par(res, p) for p in range(len(res["pares"]))])
    for m in V:
        print(m, "R² interações por par:", dict(zip([MI.nome_par(res, p) for p in range(len(res["pares"]))],
                                                    np.round(res["r2_interacao"][m], 3))))
        print(m, f"R² do modelo: {res['r2_modelo'][m]:.3f} (só os efeitos principais: {res['r2_modelo_sem_interacao'][m]:.3f})")
    print("fatores no aditivo (v1):", res["aditivo"]["Z"].shape[1], "| na v2:", res["Z"].shape[1])

    # recuperação: features verdadeiras com q(interação) > 0,5 e falsos positivos
    for m in V:
        for (k, l), idx in inter[m].items():
            if k not in est_para_vdd or l not in est_para_vdd:
                print(m, (k, l), "fatores não recuperados"); continue
            par = tuple(sorted((est_para_vdd[k], est_para_vdd[l])))
            if par not in res["pares"]:
                print(m, f"F{k+1}xF{l+1} verdadeiro: par foi podado"); continue
            pv = res["p_interacao"][m][:, res["pares"].index(par)]
            vdd = np.zeros(len(pv), bool); vdd[idx] = True
            print(m, f"F{k+1}xF{l+1} verdadeiro: sensibilidade {np.mean(pv[vdd] > .5):.2f}, "
                     f"falsos positivos {np.sum(pv[~vdd] > .5)} de {np.sum(~vdd)}")
        outros = [p for p in range(len(res["pares"]))]
        print(m, "features com interação por par:", {MI.nome_par(res, p): int((res["p_interacao"][m][:, p] > .5).sum()) for p in outros})

    tab = MI.tabela_interacoes(res, V)
    print(tab.groupby(["omica", "par", "leitura"]).size())
    MI.figura_resumo_interacoes(res, "interacao_resumo.png")
    if len(tab):
        lin = tab.iloc[0]
        MI.figura_interacao(res, V, lin.omica, lin.feature, par=tuple(int(x[1:]) - 1 for x in lin.par.split("x")),
                            arquivo="interacao_exemplo.png")
    plt.close("all")

    # comparação fora da amostra (demora ~3x um ajuste por fold)
    cv = MI.validacao_cruzada(V, K=10, folds=5)
    print(cv.groupby(["vista", "modelo"]).r2.mean().unstack().round(3))
