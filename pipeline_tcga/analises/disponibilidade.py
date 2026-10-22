"""Quantos pacientes têm cada ômica e cada combinação de ômicas (UpSet)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from .. import config
from ..graficos import AZUL, CINZA, VERMELHO, salvar


def desenhar_upset(tab: pd.DataFrame, titulo: str, max_comb: int = 25):
    """tab: pacientes x ômicas (bool). Desenha um UpSet com interseções exclusivas."""
    cols = [c for c in tab.columns if tab[c].any()]
    t = tab[cols]
    combos = t.apply(lambda r: tuple(c for c in cols if r[c]), axis=1).value_counts()
    combos = combos[[len(k) > 0 for k in combos.index]]
    total = len(t)
    combos = combos.iloc[:max_comb]
    tamanhos = t.sum().sort_values(ascending=True)
    ordem = list(tamanhos.index)  # de baixo para cima
    rot = [config.ROTULO.get(c, c) for c in ordem]

    n = len(combos)
    larg = max(3.0, 0.42 * n)  # larguras em polegadas: barras, rótulos, matriz
    fig = plt.figure(figsize=(1.6 + 2.1 + larg, 2.6 + 0.34 * len(ordem) + 3.2))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 2.1, larg],
                           height_ratios=[3.2, 0.34 * len(ordem) + 0.3], wspace=0.0, hspace=0.04)
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_mat = fig.add_subplot(gs[1, 2], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)
    ax_lab = fig.add_subplot(gs[1, 1], sharey=ax_mat)

    x = np.arange(n)
    completo = [len(k) == len(cols) for k in combos.index]
    cores = [VERMELHO if c else AZUL for c in completo]
    ax_bar.bar(x, combos.values, color=cores, width=0.65)
    for xi, v in zip(x, combos.values):
        ax_bar.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("pacientes")
    ax_bar.set_title(titulo, loc="left")
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.grid(axis="x", visible=False)

    for i in range(len(ordem)):
        if i % 2 == 0:
            ax_mat.axhspan(i - 0.5, i + 0.5, color="#f2f4f8", zorder=0)
    for xi, k in enumerate(combos.index):
        ys = [ordem.index(c) for c in k]
        ax_mat.scatter([xi] * len(ordem), range(len(ordem)), s=55, color="#dfe3ea", zorder=2)
        ax_mat.scatter([xi] * len(ys), ys, s=55, color="#1d2433", zorder=3)
        if len(ys) > 1:
            ax_mat.plot([xi, xi], [min(ys), max(ys)], color="#1d2433", lw=2, zorder=2)
    ax_mat.set_yticks(range(len(ordem)))
    ax_mat.set_yticklabels([])
    ax_mat.tick_params(axis="y", left=False)
    ax_mat.set_xticks([])
    ax_mat.grid(False)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    ax_mat.set_ylim(-0.5, len(ordem) - 0.5)

    ax_set.barh(range(len(ordem)), tamanhos.values, color=CINZA, height=0.55)
    ax_set.invert_xaxis()
    for i, v in enumerate(tamanhos.values):
        ax_set.text(v, i, f"{v} ", ha="right", va="center", fontsize=8)
    ax_set.tick_params(axis="y", left=False, labelleft=False)
    ax_set.set_xlabel("pacientes por ômica")
    ax_set.grid(axis="y", visible=False)
    ax_set.spines["left"].set_visible(False)
    for i, r in enumerate(rot):
        if i % 2 == 0:
            ax_lab.axhspan(i - 0.5, i + 0.5, color="#f2f4f8", zorder=0)
        ax_lab.text(0.96, i, r, ha="right", va="center", fontsize=9)
    ax_lab.set_xlim(0, 1)
    ax_lab.axis("off")
    fig.text(0.01, 0.01, f"{total} pacientes · barras vermelhas = paciente com todas as ômicas disponíveis",
             fontsize=8, color="#5b6475")
    return fig, combos


def analisar(ctx, col) -> dict:
    pasta = ctx.pasta_omica("00_disponibilidade")
    tab = col.tabela_disponibilidade()
    tab.to_csv(os.path.join(pasta, "disponibilidade_por_paciente.csv"))

    # Tabela de quantidade por ômica
    linhas = []
    for o in config.OMICAS:
        d = col.disp.get(o["chave"], pd.DataFrame(columns=["paciente", "tumor", "normal"]))
        if o["chave"] == "clinico":
            n_tum, n_norm, par = len(d["paciente"].unique()), 0, 0
            pacs = len(d["paciente"].unique())
        else:
            n_tum, n_norm = int(d["tumor"].sum()), int(d["normal"].sum())
            pacs = d.loc[d["tumor"], "paciente"].nunique()
            par = len(set(d.loc[d["tumor"], "paciente"]) & set(d.loc[d["normal"], "paciente"]))
        linhas.append({"ômica": o["rotulo"], "chave": o["chave"], "pacientes (tumor)": pacs,
                       "% da coorte": round(100 * pacs / max(1, len(tab)), 1), "amostras tumorais": n_tum,
                       "amostras normais": n_norm, "pacientes com par tumor-normal": par,
                       "fonte": col.fonte_disp.get(o["chave"], "")})
    qt = pd.DataFrame(linhas)
    qt.to_csv(os.path.join(pasta, "quantidade_por_omica.csv"), index=False)

    fig, combos = desenhar_upset(tab, f"{ctx.cancer}: pacientes por combinação de ômicas")
    f_upset = salvar(fig, pasta, "upset_combinacoes")
    todas = tab.apply(lambda r: "+".join(config.ROTULO[c] for c in tab.columns if r[c]) or "nenhuma", axis=1)
    comb = todas.value_counts().rename_axis("combinação").reset_index(name="pacientes")
    comb["nº de ômicas"] = comb["combinação"].map(lambda s: 0 if s == "nenhuma" else s.count("+") + 1)
    comb.to_csv(os.path.join(pasta, "combinacoes.csv"), index=False)

    # Funil: interseção acumulada das ômicas, da mais comum para a menos comum
    ordem = [c for c in tab.sum().sort_values(ascending=False).index if tab[c].any()]
    acum, cur = [], pd.Series(True, index=tab.index)
    for c in ordem:
        cur = cur & tab[c]
        acum.append({"ômicas acumuladas": " + ".join(config.ROTULO[x] for x in ordem[:len(acum) + 1]),
                     "última adicionada": config.ROTULO[c], "pacientes": int(cur.sum())})
    funil = pd.DataFrame(acum)
    funil.to_csv(os.path.join(pasta, "funil_intersecao.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(funil) + 1.2))
    ax.barh(range(len(funil))[::-1], funil["pacientes"], color=AZUL, height=0.6)
    ax.set_yticks(range(len(funil))[::-1])
    ax.set_yticklabels(["+ " + x for x in funil["última adicionada"]])
    for i, v in zip(range(len(funil))[::-1], funil["pacientes"]):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("pacientes com todas as ômicas até esta linha")
    ax.set_title("Quantos pacientes sobram ao exigir cada ômica a mais")
    ax.grid(axis="y", visible=False)
    f_funil = salvar(fig, pasta, "funil_intersecao")

    nomics = tab.sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(nomics.index, nomics.values, color=AZUL)
    ax.set_xlabel("número de ômicas por paciente")
    ax.set_ylabel("pacientes")
    ax.set_xticks(nomics.index)
    f_n = salvar(fig, pasta, "omicas_por_paciente")

    res = {
        "pacientes": len(tab),
        "por_omica": {r["chave"]: int(r["pacientes (tumor)"]) for r in linhas},
        "combinacoes": {k: int(v) for k, v in todas.value_counts().items()},
        "todas_disponiveis": int(tab[ordem].all(axis=1).sum()) if ordem else 0,
        "omicas_disponiveis": ordem,
        "funil": funil.to_dict(orient="records"),
    }
    return {"res": res, "qt": qt, "comb": comb, "funil": funil, "figs": (f_upset, f_funil, f_n), "tab": tab}
