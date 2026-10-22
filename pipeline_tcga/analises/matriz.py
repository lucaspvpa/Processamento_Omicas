"""Exploração genérica de uma matriz contínua (features x amostras).

Usada por RNA-seq, miRNA, metilação, RPPA, ATAC-seq, CPTAC e pelas contagens (em log-CPM).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..graficos import (AZUL, CINZA, COR_ESTADIO, COR_TIPO, VERMELHO, embutir_2d, fdr_bh, pca, salvar,
                        tabela_html)
from ..relatorio import fmt, pct


COR_ESTADIO_AMOSTRA = {**{k: v for k, v in COR_ESTADIO.items() if k != "NA"}, "sem estádio": COR_ESTADIO["NA"],
                       "amostra normal": "#6c7a91"}


def estadio_amostra(anot: pd.DataFrame) -> pd.Series:
    """Estádio do paciente só para tumores; amostras normais ficam numa categoria própria."""
    e = anot["estadio"].replace({"NA": "sem estádio"})
    return e.where(anot["tipo"] == "Tumor", "amostra normal")


def _rotular(ax, x: pd.Series, y: pd.Series, nomes, lado: int):
    """Rótulos empilhados num dos lados do gráfico, ligados ao ponto por uma linha fina (sem sobreposição)."""
    nomes = sorted(nomes, key=lambda n: -y[n])
    if not nomes:
        return
    ymax = float(y.max())
    passo = 0.055 * ymax
    xlim = max(abs(float(x.min())), abs(float(x.max())))
    xt = lado * xlim * 1.02
    yt = ymax
    for n in nomes:
        yt = min(yt, float(y[n]))
        ax.annotate(str(n)[:18], (x[n], y[n]), xytext=(xt, yt), fontsize=7, ha="left" if lado > 0 else "right",
                    va="center", arrowprops={"arrowstyle": "-", "color": "#8d99ae", "lw": 0.5})
        yt -= passo
    ax.set_xlim(-xlim * 1.6, xlim * 1.6)


def _cores_anot(anot: pd.DataFrame, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=anot.index)
    out["tipo"] = anot["tipo"].map(COR_TIPO).fillna(CINZA)
    out["estádio"] = estadio_amostra(anot).map(COR_ESTADIO_AMOSTRA).fillna("#d9d9d9")
    return out.loc[cols]


def _scatter_cat(ax, xy: pd.DataFrame, cat: pd.Series, paleta: dict, titulo: str, rot=("PC1", "PC2")):
    for k in [k for k in paleta if k in set(cat)] + [k for k in sorted(set(cat)) if k not in paleta]:
        m = cat == k
        ax.scatter(xy.loc[m].iloc[:, 0], xy.loc[m].iloc[:, 1], s=26, alpha=0.85, color=paleta.get(k, CINZA),
                   label=f"{k} ({m.sum()})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(rot[0])
    ax.set_ylabel(rot[1])
    ax.set_title(titulo)
    ax.legend(fontsize=8, loc="best")


def explorar(ctx, M: pd.DataFrame, chave: str, rel, pasta: str, unidade: str, nome_feat: str = "genes",
             n_top: int = 2000, diferencial: bool = True, prefixo: str = "") -> dict:
    p = prefixo
    M = M.dropna(how="all")
    M = M.loc[:, M.notna().any()]
    anot = ctx.anotar(M.columns)
    tum = anot.index[anot["tipo"] == "Tumor"]
    nor = anot.index[anot["tipo"] == "Normal"]
    res = {"features": int(M.shape[0]), "amostras": int(M.shape[1]), "tumor": len(tum), "normal": len(nor),
           "pct_faltante": float(M.isna().mean().mean())}
    rel.kpi(f"{nome_feat}", res["features"])
    rel.kpi("amostras tumorais", len(tum))
    rel.kpi("amostras normais", len(nor))
    rel.kpi("valores faltantes", pct(res["pct_faltante"]))
    if M.shape[1] < 2:
        rel.achado("Há menos de duas amostras; a exploração multivariada não foi feita.")
        return res

    # ---------------- distribuição
    ordem = M.median().sort_values().index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    cores = [COR_TIPO[anot.loc[s, "tipo"]] for s in ordem]
    if M.shape[1] <= 150:
        bp = a1.boxplot([M[s].dropna().values for s in ordem], showfliers=False, patch_artist=True, widths=0.7)
        for b, c in zip(bp["boxes"], cores):
            b.set_facecolor(c)
            b.set_alpha(0.75)
        a1.set_xticks([])
    else:
        q = M[ordem].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        x = np.arange(len(ordem))
        a1.fill_between(x, q.loc[0.05], q.loc[0.95], color=CINZA, alpha=0.25, label="5–95%")
        a1.fill_between(x, q.loc[0.25], q.loc[0.75], color=AZUL, alpha=0.35, label="IQR")
        a1.plot(x, q.loc[0.5], color="#1d2433", lw=1, label="mediana")
        a1.legend()
    a1.set_xlabel(f"amostras ordenadas pela mediana (n={M.shape[1]})")
    a1.set_ylabel(unidade)
    a1.set_title("Distribuição por amostra (vermelho = tumor, azul = normal)")
    lo, hi = np.nanpercentile(M.values, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    for s in M.columns[:400]:
        h, _ = np.histogram(M[s].dropna(), bins=bins, density=True)
        a2.plot((bins[:-1] + bins[1:]) / 2, h, color=COR_TIPO[anot.loc[s, "tipo"]], alpha=0.25, lw=0.8)
    a2.set_xlabel(unidade)
    a2.set_ylabel("densidade")
    a2.set_title("Densidade por amostra")
    f_dist = salvar(fig, pasta, p + "distribuicao_amostras")

    # ---------------- média x desvio
    media, dp = M.mean(axis=1), M.std(axis=1)
    top = dp.dropna().nlargest(min(n_top, M.shape[0])).index
    falt_feat = M.isna().mean(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    a1.scatter(media, dp, s=3, color=CINZA, alpha=0.4, rasterized=True)
    a1.scatter(media[top], dp[top], s=3, color=VERMELHO, alpha=0.5, rasterized=True, label=f"top {len(top)} mais variáveis")
    a1.set_xlabel(f"média ({unidade})")
    a1.set_ylabel("desvio-padrão")
    a1.legend()
    a1.set_title(f"Média x variabilidade dos {nome_feat}")
    if falt_feat.max() > 0:
        a2.hist(falt_feat * 100, bins=40, color=AZUL)
        a2.set_xlabel(f"% de amostras faltantes por {nome_feat[:-1] if nome_feat.endswith('s') else nome_feat}")
        a2.set_ylabel(nome_feat)
        a2.set_yscale("log")
    else:
        a2.text(0.5, 0.5, "Nenhum valor faltante", ha="center", va="center", fontsize=12, color=CINZA,
                transform=a2.transAxes)
        a2.set_xticks([])
        a2.set_yticks([])
        a2.grid(False)
    a2.set_title("Faltantes por feature")
    f_mv = salvar(fig, pasta, p + "media_variancia")
    res["top_variaveis"] = list(top[:30])
    pd.DataFrame({"media": media, "dp": dp, "pct_faltante": falt_feat}).loc[top].to_csv(
        os.path.join(pasta, p + "top_variaveis.csv"))

    # ---------------- PCA e UMAP
    X = M.loc[top].T
    X = X.loc[:, X.isna().mean() < 0.5]
    f_pca = f_umap = None
    if X.shape[0] >= 4 and X.shape[1] >= 3:
        sc, var, cargas = pca(X, 10)
        res["var_pc"] = [float(v) for v in var[:5]]
        sc.to_csv(os.path.join(pasta, p + "pca_scores.csv"))
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
        axs[0].bar(range(1, len(var) + 1), var * 100, color=AZUL)
        axs[0].set_xlabel("componente")
        axs[0].set_ylabel("% da variância")
        axs[0].set_title("Variância explicada")
        xy = sc[["PC1", "PC2"]]
        rot = (f"PC1 ({fmt(var[0] * 100)}%)", f"PC2 ({fmt(var[1] * 100) if len(var) > 1 else 0}%)")
        if xy.shape[1] == 1:
            xy = xy.assign(PC2=0)
        _scatter_cat(axs[1], xy, anot.loc[sc.index, "tipo"], COR_TIPO, "PCA por tipo de amostra", rot)
        _scatter_cat(axs[2], xy, estadio_amostra(anot.loc[sc.index]), COR_ESTADIO_AMOSTRA, "PCA por estádio", rot)
        f_pca = salvar(fig, pasta, p + "pca")
        top_load = cargas["PC1"].abs().nlargest(10).index
        res["pc1_cargas"] = list(top_load)
        from ..graficos import padronizar
        emb, metodo = embutir_2d(padronizar(X).to_numpy())
        if emb is not None:
            e = pd.DataFrame(emb, index=X.index, columns=["d1", "d2"])
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
            _scatter_cat(axs[0], e, anot.loc[e.index, "tipo"], COR_TIPO, f"{metodo} por tipo", (f"{metodo} 1", f"{metodo} 2"))
            _scatter_cat(axs[1], e, estadio_amostra(anot.loc[e.index]), COR_ESTADIO_AMOSTRA, f"{metodo} por estádio",
                         (f"{metodo} 1", f"{metodo} 2"))
            f_umap = salvar(fig, pasta, p + "umap")
            res["metodo_2d"] = metodo

    # ---------------- correlação entre amostras
    f_cor = f_hm = None
    Xf = M.loc[top]
    Xf = Xf.loc[Xf.isna().mean(axis=1) < 0.5]
    Xf = Xf.T.fillna(Xf.mean(axis=1)).T
    if Xf.shape[1] >= 3:
        C = Xf.corr()
        res["cor_mediana_entre_amostras"] = float(np.nanmedian(C.values[np.triu_indices_from(C, 1)]))
        med_amostra = (C.sum() - 1) / (len(C) - 1)
        q1, q3 = med_amostra.quantile([0.25, 0.75])
        outl = med_amostra[med_amostra < q1 - 3 * (q3 - q1)].index.tolist()
        res["outliers"] = outl
        tam = min(14, 4 + 0.08 * len(C))
        g = sns.clustermap(C, cmap="mako", col_colors=_cores_anot(anot, C.columns), xticklabels=False,
                           yticklabels=len(C) <= 60, figsize=(tam, tam), cbar_kws={"label": "Pearson"})
        g.ax_heatmap.set_ylabel("")
        g.fig.suptitle("Correlação entre amostras (features mais variáveis)", y=1.01)
        f_cor = salvar(g.fig, pasta, p + "correlacao_amostras")

        h = Xf.loc[Xf.var(axis=1).nlargest(min(50, len(Xf))).index]
        hz = h.sub(h.mean(axis=1), axis=0).div(h.std(axis=1).replace(0, 1), axis=0).clip(-3, 3)
        g = sns.clustermap(hz, cmap="RdBu_r", center=0, col_colors=_cores_anot(anot, hz.columns), xticklabels=False,
                           yticklabels=True, figsize=(min(16, 6 + 0.06 * hz.shape[1]), 11),
                           cbar_kws={"label": "z-score"})
        g.ax_heatmap.tick_params(axis="y", labelsize=7)
        g.fig.suptitle(f"Os {len(hz)} {nome_feat} mais variáveis (z-score por linha)", y=1.01)
        f_hm = salvar(g.fig, pasta, p + "heatmap_top")

    # ---------------- tumor x normal
    f_volc = None
    if diferencial and len(tum) >= 3 and len(nor) >= 3:
        A, B = M[tum].to_numpy(dtype=float), M[nor].to_numpy(dtype=float)
        t, pv = stats.ttest_ind(A, B, axis=1, equal_var=False, nan_policy="omit")
        pv = np.asarray(pv, dtype=float)
        dif = np.nanmean(A, axis=1) - np.nanmean(B, axis=1)
        de = pd.DataFrame({"diferenca_tumor_menos_normal": dif, "t": np.asarray(t, dtype=float), "p": pv,
                           "fdr": fdr_bh(pv)}, index=M.index).dropna(subset=["p"])
        de = de.sort_values("p")
        de.to_csv(os.path.join(pasta, p + "tumor_vs_normal.csv"))
        sig = de[de["fdr"] < 0.05]
        res["de_fdr05"] = int(len(sig))
        res["de_up"] = int((sig["diferenca_tumor_menos_normal"] > 0).sum())
        res["de_down"] = int((sig["diferenca_tumor_menos_normal"] < 0).sum())
        res["de_top_up"] = list(sig[sig["diferenca_tumor_menos_normal"] > 0].index[:10])
        res["de_top_down"] = list(sig[sig["diferenca_tumor_menos_normal"] < 0].index[:10])
        pares = set(anot.loc[tum, "paciente"]) & set(anot.loc[nor, "paciente"])
        res["pares_tumor_normal"] = len(pares)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cor = np.where(de["fdr"] < 0.05, np.where(de.iloc[:, 0] > 0, VERMELHO, AZUL), CINZA)
        ax.scatter(de.iloc[:, 0], -np.log10(de["p"]), s=5, c=cor, alpha=0.6, rasterized=True)
        y = -np.log10(de["p"])
        forca = de.iloc[:, 0].abs() * y
        for lado, sel in ((1, sig.iloc[:, 0] > 0), (-1, sig.iloc[:, 0] < 0)):
            _rotular(ax, de.iloc[:, 0], y, forca[sig.index[sel]].nlargest(6).index, lado)
        ax.set_xlabel(f"diferença média tumor − normal ({unidade})")
        ax.set_ylabel("−log10(p)")
        ax.set_title(f"Tumor ({len(tum)}) x normal ({len(nor)}): {len(sig)} com FDR<5%")
        f_volc = salvar(fig, pasta, p + "tumor_vs_normal_volcano")
        rel_tab = sig.head(30)

    # ---------------- associação com a clínica (só tumores)
    assoc = associacao_clinica(ctx, M[tum] if len(tum) else M, top)
    if assoc is not None:
        assoc.to_csv(os.path.join(pasta, p + "associacao_clinica_pcs.csv"))
        res["assoc"] = assoc.to_dict()

    # ---------------- texto
    rel.secao("Distribuição dos valores")
    rel.figura(f_dist, f"<b>Como ler:</b> à esquerda, a distribuição de {unidade} em cada amostra, ordenadas pela "
               "mediana; amostras muito deslocadas das demais podem ter problema técnico ou de normalização. À direita, "
               "a densidade de cada amostra sobreposta (vermelho = tumor, azul = normal).", pasta)
    rel.figura(f_mv, f"<b>Como ler:</b> cada ponto é um {nome_feat[:-1] if nome_feat.endswith('s') else nome_feat}. "
               f"Em vermelho, os {len(top)} mais variáveis, usados na PCA, no UMAP e nos heatmaps. À direita, quantas "
               "amostras faltam por feature.", pasta)
    rel.secao("Estrutura das amostras")
    if f_pca:
        rel.figura(f_pca, "<b>Como ler:</b> a PCA resume as features mais variáveis em poucos eixos. Amostras próximas "
                   "têm perfis parecidos. Separação entre tumor e normal é esperada; separação por estádio indica que "
                   "a ômica carrega informação de progressão.", pasta)
    if f_umap:
        rel.figura(f_umap, f"<b>Como ler:</b> {res.get('metodo_2d', 'UMAP')} é uma projeção não linear das mesmas "
                   "features. Distâncias globais não são interpretáveis, mas grupos bem separados sugerem subtipos.", pasta)
    if f_cor:
        rel.figura(f_cor, "<b>Como ler:</b> correlação de Pearson entre pares de amostras, com agrupamento hierárquico. "
                   "A barra colorida de cima indica tipo (vermelho tumor, azul normal) e estádio.", pasta)
    if f_hm:
        rel.figura(f_hm, f"<b>Como ler:</b> os {nome_feat} mais variáveis, padronizados por linha (vermelho = acima da "
                   "média do gene, azul = abaixo). Blocos de cor mostram grupos de amostras com programas em comum.", pasta)
    if f_volc:
        rel.secao("Tumor x normal")
        rel.figura(f_volc, "<b>Como ler:</b> eixo x = diferença média entre tumor e normal; eixo y = significância. "
                   "Vermelho = maior no tumor, azul = maior no normal (FDR &lt; 5%, teste t de Welch).", pasta)
        if len(rel_tab):
            rel.tabela(tabela_html(rel_tab, 30), "Features mais diferentes entre tumor e normal")
    if assoc is not None and len(assoc):
        rel.secao("Associação com variáveis clínicas")
        rel.texto("p-valores da associação dos três primeiros componentes principais (só tumores) com estádio "
                  "(Kruskal-Wallis), sexo (Mann-Whitney), idade (Spearman) e sobrevida global (Cox univariado).")
        rel.tabela(tabela_html(assoc.apply(lambda c: c.map(lambda x: float(f"{x:.3g}") if pd.notna(x) else x)), 10))

    _achados(rel, res, nome_feat, unidade)
    return res


def associacao_clinica(ctx, M: pd.DataFrame, top) -> pd.DataFrame | None:
    X = M.loc[[t for t in top if t in M.index]].T
    X = X.loc[:, X.isna().mean() < 0.5]
    if X.shape[0] < 8 or X.shape[1] < 3:
        return None
    sc, var, _ = pca(X, 3)
    anot = ctx.anotar(sc.index)
    clin = ctx.clin.reindex(anot["paciente"].values)
    clin.index = sc.index
    out = {}
    for pc in sc.columns:
        v = sc[pc]
        r = {}
        g = anot["estadio"]
        grupos = [v[g == k].values for k in ["I", "II", "III", "IV"] if (g == k).sum() >= 3]
        r["estádio (Kruskal)"] = stats.kruskal(*grupos).pvalue if len(grupos) >= 2 else np.nan
        sx = anot["sexo"]
        cats = [k for k in sx.unique() if k not in ("NA", None) and (sx == k).sum() >= 3]
        r["sexo (Mann-Whitney)"] = stats.mannwhitneyu(v[sx == cats[0]], v[sx == cats[1]]).pvalue if len(cats) == 2 else np.nan
        idade = pd.to_numeric(anot["idade"], errors="coerce")
        ok = idade.notna()
        r["idade (Spearman)"] = stats.spearmanr(v[ok], idade[ok]).pvalue if ok.sum() >= 8 else np.nan
        try:
            from lifelines import CoxPHFitter
            d = pd.DataFrame({"t": pd.to_numeric(clin["OS.time"], errors="coerce"),
                              "e": pd.to_numeric(clin["OS"], errors="coerce"), "x": (v - v.mean()) / v.std()}).dropna()
            if len(d) >= 10 and d["e"].sum() >= 3:
                r["sobrevida OS (Cox)"] = CoxPHFitter().fit(d, "t", "e").summary.loc["x", "p"]
            else:
                r["sobrevida OS (Cox)"] = np.nan
        except Exception:
            r["sobrevida OS (Cox)"] = np.nan
        out[f"{pc} ({fmt(100 * var[int(pc[2:]) - 1])}%)"] = r
    return pd.DataFrame(out).T


def _achados(rel, res, nome_feat, unidade):
    rel.achado(f"{fmt(res['features'], 0)} {nome_feat} medidos em {res['amostras']} amostras "
               f"({res['tumor']} tumorais e {res['normal']} normais); {pct(res['pct_faltante'])} dos valores estão faltando.")
    if "var_pc" in res:
        v = res["var_pc"]
        rel.achado(f"Os dois primeiros componentes principais explicam {fmt(100 * v[0])}% e "
                   f"{fmt(100 * v[1]) if len(v) > 1 else '0'}% da variância das features mais variáveis.")
    if res.get("cor_mediana_entre_amostras") is not None:
        rel.achado(f"Correlação mediana entre amostras: {fmt(res['cor_mediana_entre_amostras'], 2)}.")
    if res.get("outliers"):
        rel.achado("Amostras com correlação muito abaixo das demais (possíveis outliers técnicos): "
                   + ", ".join(res["outliers"][:10]) + ".")
    if "de_fdr05" in res:
        rel.achado(f"Tumor x normal: {res['de_fdr05']} {nome_feat} diferentes com FDR &lt; 5% "
                   f"({res['de_up']} maiores no tumor, {res['de_down']} maiores no normal; "
                   f"{res.get('pares_tumor_normal', 0)} pacientes têm o par tumor-normal).")
        if res["de_top_up"]:
            rel.achado("Mais aumentados no tumor: " + ", ".join(res["de_top_up"][:8]) + ".")
        if res["de_top_down"]:
            rel.achado("Mais reduzidos no tumor: " + ", ".join(res["de_top_down"][:8]) + ".")
    elif res.get("normal", 0) < 3:
        rel.achado("Há menos de 3 amostras normais, então a comparação tumor x normal não foi feita.")
    a = res.get("assoc")
    if a:
        sig = [(pc, var, p) for var, d in a.items() for pc, p in d.items() if pd.notna(p) and p < 0.05]
        if sig:
            rel.achado("Associações clínicas com p &lt; 0,05 nos componentes principais: " +
                       "; ".join(f"{pc.split(' ')[0]} com {var.split(' ')[0]} (p={p:.2g})" for pc, var, p in sig[:6]) + ".")
        else:
            rel.achado("Nenhum dos três primeiros componentes principais se associou a estádio, sexo, idade ou "
                       "sobrevida com p &lt; 0,05.")
