"""Exemplo do MOFA não linear aditivo (mofa_aditivo.py) em dados simulados com respostas lineares, côncavas, sigmoides e em U."""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import mofa_aditivo as MA
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def simula(N=300, semente=1, frac_nl=0.5):
    rng = np.random.default_rng(semente)
    D = {"rna": 400, "met": 300, "cnv": 200}
    ativo = {"rna": [1, 1, 0, 1], "met": [1, 1, 0, 0], "cnv": [1, 0, 1, 0]}
    ruido = {"rna": 1.0, "met": 1.2, "cnv": 0.7}
    Z = rng.standard_normal((N, 4)); V = {}; T = {}
    formas = ["concava", "sigmoide", "U"]
    for m, d in D.items():
        X = np.zeros((N, d)); tipo = np.full((d, 4), "", dtype=object)
        for k in range(4):
            if not ativo[m][k]: continue
            idx = rng.choice(d, int(0.2 * d), replace=False)
            for j in idx:
                a = rng.choice([-1, 1]) * rng.uniform(1.0, 2.0); z = Z[:, k]
                if rng.random() < frac_nl:
                    f = rng.choice(formas)
                    if f == "concava": x = -softplus(-1.5 * z) / 1.5 * a * 1.3
                    elif f == "sigmoide": x = 1.5 * np.tanh(1.5 * z) * a
                    else: x = 0.8 * (z ** 2 - 1) * abs(a)
                    tipo[j, k] = f
                else:
                    x = a * z; tipo[j, k] = "linear"
                X[:, j] += x
        V[m] = X + ruido[m] * rng.standard_normal((N, d)); T[m] = tipo
    return V, Z, T


if __name__ == "__main__":
    visoes, Z, tipos = simula(N=300, frac_nl=0.5)
    res = MA.mofa_aditivo(visoes, K=10)            # 1) MOFA linear  2) remove fatores derivados  3) liga as curvas

    # quanto cada fator explica, separado em parte linear e parte não linear
    for m in visoes:
        print(m, "R2 linear", np.round(res["r2_linear"][m], 3), "R2 não linear", np.round(res["r2_nao_linear"][m], 3))

    # tabela por feature: associação, P(não linear), coeficientes e forma da curva
    tab = MA.tabela_formas(res, visoes)
    print(tab.groupby(["omica", "fator", "forma"]).size())

    # conferência com a verdade simulada (os fatores podem vir em outra ordem e sinal)
    C = np.abs(np.corrcoef(Z.T, res["Z"].T)[:4, 4:])
    ordem = C.argmax(axis=1)
    for m in visoes:
        for k_true, k_est in enumerate(ordem):
            if k_est >= res["Z"].shape[1]:
                continue
            verd = pd.Series(tipos[m][:, k_true])
            pnl = res["p_nao_linear"][m][:, k_est]
            for f in ["linear", "concava", "sigmoide", "U"]:
                if (verd == f).sum():
                    print(m, f"F{k_true + 1}", f, "fração marcada não linear:", round(float((pnl[verd == f] > 0.5).mean()), 2))

    MA.figura_resumo(res)
    MA.figura_curvas(res, visoes, "rna", 0, n=8)
    plt.show()
