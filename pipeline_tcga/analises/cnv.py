"""Alterações de número de cópias (SNP6): frequência ao longo do genoma, fração alterada e genes recorrentes."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..graficos import AZUL, CINZA, VERMELHO, salvar, tabela_html
from ..relatorio import Relatorio, fmt, pct

ORDEM_CHR = [str(i) for i in range(1, 23)] + ["X", "Y"]


def _ordenar_genes(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy()
    p["ordem_chr"] = p["chrom"].map({c: i for i, c in enumerate(ORDEM_CHR)})
    return p.sort_values(["ordem_chr", "meio"])


def analisar(ctx, om) -> dict:
    pasta = ctx.pasta_omica("cnv")
    rel = Relatorio(f"{ctx.cancer} · Número de cópias (CNV)", "Affymetrix SNP6, segmentos whitelisted do PanCanAtlas")
    rel.fonte(om.fonte)
    M = om.matriz
    seg = om.extra["segmentos"]
    pos = _ordenar_genes(om.extra["posicoes"].loc[om.extra["posicoes"].index.intersection(M.index)])
    M = M.loc[pos.index]
    anot = ctx.anotar(M.columns)
    tum = anot.index[anot["tipo"] == "Tumor"]
    nor = anot.index[anot["tipo"] == "Normal"]
    T = M[tum]
    nt = len(tum)

    ganho = (T > config.CNV_GANHO).mean(axis=1)
    perda = (T < config.CNV_PERDA).mean(axis=1)
    fga = ((T > config.CNV_GANHO) | (T < config.CNV_PERDA)).mean(axis=0)
    ctx.metricas.setdefault("fracao_genoma_alterada", {}).update(fga.to_dict())
    nseg = seg[seg["sampleID"].isin(tum)].groupby("sampleID").size().reindex(tum).fillna(0)
    pd.DataFrame({"fracao_genes_alterados": fga, "segmentos": nseg}).to_csv(
        os.path.join(pasta, "fracao_alterada_por_amostra.csv"))
    tab_genes = pd.DataFrame({"chrom": pos["chrom"], "% ganho": (ganho * 100).round(1), "% perda": (perda * 100).round(1),
                              "média log2": T.mean(axis=1).round(3)})
    tab_genes.to_csv(os.path.join(pasta, "frequencia_por_gene.csv"))

    # ---------------- frequência ao longo do genoma
    x = np.arange(len(pos))
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.fill_between(x, 0, ganho.values * 100, color=VERMELHO, step="mid", alpha=0.85, label=f"ganho (> {config.CNV_GANHO})")
    ax.fill_between(x, 0, -perda.values * 100, color=AZUL, step="mid", alpha=0.85, label=f"perda (< {config.CNV_PERDA})")
    lim = pos.groupby("chrom", sort=False).apply(lambda d: (pos.index.get_loc(d.index[0]), pos.index.get_loc(d.index[-1])))
    for i, (c, (a, b)) in enumerate(lim.items()):
        if i % 2 == 0:
            ax.axvspan(a, b, color="#f2f4f8", zorder=0)
        ax.text((a + b) / 2, -102, c, ha="center", va="top", fontsize=7)
    ax.axhline(0, color="#1d2433", lw=0.6)
    ax.set_ylim(-100, 100)
    ax.set_xlim(0, len(pos))
    ax.set_xticks([])
    ax.set_ylabel("% dos tumores")
    ax.set_title(f"Frequência de ganhos e perdas por gene ao longo do genoma (n={nt} tumores)")
    ax.legend(loc="upper right")
    ax.grid(False)
    f1 = salvar(fig, pasta, "frequencia_genoma")

    # ---------------- heatmap amostras x genoma
    f2 = None
    if nt >= 2:
        passo = max(1, len(pos) // 1500)
        H = T.iloc[::passo].T
        ordem = fga.sort_values(ascending=False).index
        H = H.loc[ordem]
        fig, ax = plt.subplots(figsize=(16, min(12, 2 + 0.06 * nt)))
        ax.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for c, (a, b) in lim.items():
            ax.axvline(a / passo, color="white", lw=0.4)
            ax.text((a + b) / 2 / passo, -0.8, c, ha="center", fontsize=6)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel("tumores (ordenados pela fração alterada)")
        ax.set_title("Perfil de número de cópias (vermelho = ganho, azul = perda; log2 tumor/normal)", pad=16)
        ax.grid(False)
        f2 = salvar(fig, pasta, "heatmap_genoma")

    # ---------------- fração alterada e distribuição
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
    axs[0].hist(T.values[np.isfinite(T.values)].ravel(), bins=80, color=CINZA)
    axs[0].axvline(config.CNV_GANHO, color=VERMELHO, ls="--", lw=1)
    axs[0].axvline(config.CNV_PERDA, color=AZUL, ls="--", lw=1)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("log2(tumor/normal) por gene")
    axs[0].set_title("Distribuição dos valores")
    axs[1].hist(fga * 100, bins=20, color=AZUL, edgecolor="white")
    axs[1].set_xlabel("% dos genes alterados")
    axs[1].set_ylabel("tumores")
    axs[1].set_title(f"Fração do genoma alterada (mediana {fmt(100 * fga.median())}%)")
    g = anot.loc[tum, "estadio"]
    grupos = [k for k in ["I", "II", "III", "IV"] if (g == k).sum() >= 2]
    p_est = np.nan
    if len(grupos) >= 2:
        dados = [fga[g == k] * 100 for k in grupos]
        axs[2].boxplot(dados, showfliers=False)
        axs[2].set_xticks(range(1, len(grupos) + 1))
        axs[2].set_xticklabels(grupos)
        for i, d in enumerate(dados, 1):
            axs[2].scatter(np.random.default_rng(1).normal(i, 0.05, len(d)), d, s=12, color=VERMELHO, alpha=0.6)
        p_est = stats.kruskal(*dados).pvalue
        axs[2].set_title(f"Fração alterada por estádio (p={p_est:.2g})")
        axs[2].set_ylabel("% genes alterados")
    else:
        axs[2].axis("off")
    f3 = salvar(fig, pasta, "fracao_alterada")

    # ---------------- genes recorrentes
    top_g = ganho.sort_values(ascending=False).head(20)
    top_p = perda.sort_values(ascending=False).head(20)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.barh(top_g.index[::-1], top_g.values[::-1] * 100, color=VERMELHO)
    a1.set_title("Genes mais frequentemente ganhos")
    a1.set_xlabel("% dos tumores")
    a2.barh(top_p.index[::-1], top_p.values[::-1] * 100, color=AZUL)
    a2.set_title("Genes mais frequentemente perdidos")
    a2.set_xlabel("% dos tumores")
    for a in (a1, a2):
        a.tick_params(axis="y", labelsize=8)
        a.grid(axis="y", visible=False)
    f4 = salvar(fig, pasta, "genes_recorrentes")

    # braços cromossômicos aproximados: média por cromossomo
    braco = pd.DataFrame({"ganho": ganho, "perda": perda, "chrom": pos["chrom"]}).groupby("chrom").mean().reindex(
        [c for c in ORDEM_CHR if c in set(pos["chrom"])])
    res = {"tumores": nt, "normais": len(nor), "genes": int(len(M)), "fga_mediana": float(fga.median()),
           "top_ganhos": {k: float(v) for k, v in top_g.head(10).items()},
           "top_perdas": {k: float(v) for k, v in top_p.head(10).items()},
           "cromossomo_ganho": braco["ganho"].idxmax() if len(braco) else None,
           "cromossomo_perda": braco["perda"].idxmax() if len(braco) else None,
           "p_estadio": None if p_est != p_est else float(p_est)}

    rel.kpi("tumores", nt)
    rel.kpi("genes", len(M))
    rel.kpi("fração alterada mediana", pct(fga.median()))
    rel.kpi("segmentos/tumor (mediana)", int(nseg.median()) if len(nseg) else 0)
    rel.achado(f"Mediana de {pct(fga.median())} dos genes com ganho ou perda por tumor (limiares log2 de "
               f"{config.CNV_GANHO} e {config.CNV_PERDA}).")
    if len(braco):
        rel.achado(f"O cromossomo com mais ganhos em média é o {res['cromossomo_ganho']} "
                   f"({pct(braco['ganho'].max())} dos tumores por gene) e o com mais perdas é o {res['cromossomo_perda']} "
                   f"({pct(braco['perda'].max())}).")
    rel.achado("Genes mais ganhos: " + ", ".join(f"<i>{k}</i> ({fmt(100 * v, 0)}%)" for k, v in top_g.head(6).items()) +
               ". Genes mais perdidos: " + ", ".join(f"<i>{k}</i> ({fmt(100 * v, 0)}%)" for k, v in top_p.head(6).items()) + ".")
    if p_est == p_est:
        rel.achado(f"Fração alterada por estádio: Kruskal-Wallis p={p_est:.2g}.")
    rel.nota("O valor por gene é o do segmento que cobre o meio do gene (log2 tumor/normal, hg19). Sem correção de "
             "pureza e ploidia: tumores com pouca célula tumoral têm valores atenuados. Para chamadas discretas "
             "(−2 a 2), use o GISTIC2 do PanCanAtlas ou o ABSOLUTE.")

    rel.secao("Paisagem ao longo do genoma")
    rel.figura(f1, "<b>Como ler:</b> para cada gene, na ordem dos cromossomos, o percentual de tumores com ganho (para "
               "cima, vermelho) e com perda (para baixo, azul). Picos largos indicam eventos de braço ou cromossomo "
               "inteiro; picos estreitos, amplificações ou deleções focais.", pasta)
    if f2:
        rel.figura(f2, "<b>Como ler:</b> cada linha é um tumor e cada coluna uma região do genoma. Tumores no topo têm "
                   "mais genoma alterado.", pasta)
    rel.secao("Carga de alterações")
    rel.figura(f3, "<b>Como ler:</b> distribuição dos valores log2 (as linhas tracejadas são os limiares), fração do "
               "genoma alterada por tumor e sua relação com o estádio.", pasta)
    rel.secao("Genes recorrentes")
    rel.figura(f4, "<b>Como ler:</b> genes com ganho ou perda no maior número de tumores. Genes vizinhos aparecem "
               "juntos porque compartilham o mesmo segmento.", pasta)
    rel.tabela(tabela_html(tab_genes.sort_values("% ganho", ascending=False).head(25)), "Top 25 genes por frequência de ganho")
    rel.tabela(tabela_html(tab_genes.sort_values("% perda", ascending=False).head(25)), "Top 25 genes por frequência de perda")
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo["cnv"] = res
    return res
