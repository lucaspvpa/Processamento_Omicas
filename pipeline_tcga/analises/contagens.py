"""Contagens brutas do GDC (RNA-seq STAR e miRNA BCGSC).

Foco no que importa para um modelo de contagem: tamanho de biblioteca, sobredispersão
(Poisson x binomial negativa), excesso de zeros e composição por tipo de gene.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..graficos import AZUL, CINZA, COR_TIPO, LARANJA, VERDE, VERMELHO, salvar
from ..relatorio import Relatorio, fmt, pct
from . import matriz


def estimar_dispersao(C: pd.DataFrame):
    """Dispersão por gene pelo método dos momentos sobre contagens normalizadas por biblioteca."""
    lib = C.sum()
    fator = lib / np.exp(np.log(lib[lib > 0]).mean())
    N = C.div(fator, axis=1)
    mu = N.mean(axis=1)
    var = N.var(axis=1)
    alfa = ((var - mu) / mu.pow(2)).where(mu > 0)
    return mu, var, alfa.clip(lower=1e-4), fator


def analisar(ctx, om, tipo: str = "rna") -> dict:
    chave = "rna_contagem" if tipo == "rna" else "mirna_contagem"
    pasta = ctx.pasta_omica(chave)
    nome = "RNA-seq" if tipo == "rna" else "miRNA-seq"
    rel = Relatorio(f"{ctx.cancer} · {nome}: contagens brutas (GDC)",
                    "STAR - Counts (GENCODE v36)" if tipo == "rna" else "BCGSC miRNA Profiling (miRBase v21)")
    rel.fonte(om.fonte)
    for n_ in om.notas:
        rel.nota(n_)
    C = om.matriz.fillna(0)
    genes = om.extra.get("genes")
    anot = ctx.anotar(C.columns)
    lib = C.sum()
    det = (C >= 10).sum()
    ctx.metricas.setdefault(f"biblioteca_{tipo}_milhoes", {}).update((lib / 1e6).to_dict())
    pd.DataFrame({"tamanho_biblioteca": lib, "features_com_10+_leituras": det}).to_csv(
        os.path.join(pasta, "biblioteca_por_amostra.csv"))

    # ---------------- biblioteca
    o = lib.sort_values()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    a1.bar(range(len(o)), o.values / 1e6, color=[COR_TIPO[anot.loc[s, "tipo"]] for s in o.index], width=0.9)
    a1.set_xticks([])
    a1.set_ylabel("milhões de leituras")
    a1.set_xlabel(f"amostras (vermelho = tumor, azul = normal; n={len(o)})")
    a1.set_title(f"Tamanho da biblioteca (mediana {fmt(lib.median() / 1e6)} milhões)")
    a2.scatter(lib / 1e6, det, s=16, c=[COR_TIPO[anot.loc[s, 'tipo']] for s in lib.index])
    a2.set_xlabel("milhões de leituras")
    a2.set_ylabel("features com ≥10 leituras")
    a2.set_title("Profundidade x features detectadas")
    f1 = salvar(fig, pasta, "biblioteca")

    # ---------------- composição
    f2 = None
    comp = None
    if tipo == "rna" and genes is not None and "gene_type" in genes and genes["gene_type"].notna().any():
        gt = genes["gene_type"].reindex(C.index).fillna("outro")
        comp = C.groupby(gt).sum()
        comp = comp.div(comp.sum())
        principais = comp.mean(axis=1).nlargest(6).index
        comp = pd.concat([comp.loc[principais], comp.drop(principais).sum().to_frame("outros").T])
        fig, ax = plt.subplots(figsize=(14, 3.6))
        base = np.zeros(comp.shape[1])
        ordem = comp.loc[principais[0]].sort_values().index
        for k, cor in zip(comp.index, [AZUL, LARANJA, VERDE, VERMELHO, "#6c4f8c", "#2a9d8f", CINZA]):
            ax.bar(range(len(ordem)), comp.loc[k, ordem], bottom=base, color=cor, width=1, label=k)
            base += comp.loc[k, ordem].values
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("fração das leituras")
        ax.set_title("Composição das leituras por tipo de gene")
        ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        f2 = salvar(fig, pasta, "composicao_tipo_gene")

    # genes que dominam a biblioteca
    frac = C.div(lib, axis=1)
    top_dom = frac.mean(axis=1).nlargest(20)
    nomes = genes["gene_name"].reindex(top_dom.index) if genes is not None and "gene_name" in genes else None
    rotulos = [f"{nomes[g]} ({g.split('.')[0]})" if nomes is not None and isinstance(nomes.get(g), str) else g for g in top_dom.index]
    top20 = frac.loc[frac.mean(axis=1).nlargest(20).index].sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    a1.barh(rotulos[::-1], top_dom.values[::-1] * 100, color=AZUL)
    a1.tick_params(axis="y", labelsize=7)
    a1.set_xlabel("% médio das leituras")
    a1.set_title("Features que mais consomem leituras")
    a1.grid(axis="y", visible=False)
    a2.hist(top20 * 100, bins=20, color=LARANJA, edgecolor="white")
    a2.set_xlabel("% das leituras nas 20 features acima")
    a2.set_ylabel("amostras")
    a2.set_title("Concentração da biblioteca")
    f3 = salvar(fig, pasta, "features_dominantes")

    # ---------------- sobredispersão e zeros
    expressos = C.index[(C > 0).sum(axis=1) >= max(3, 0.1 * C.shape[1])]
    Ce = C.loc[expressos]
    mu, var, alfa, fator = estimar_dispersao(Ce)
    alfa_med = float(alfa[mu > 10].median()) if (mu > 10).any() else float(alfa.median())
    ok = (mu > 0) & (var > 0)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    axs[0].scatter(mu[ok], var[ok], s=2, alpha=0.3, color=CINZA, rasterized=True)
    xs = np.logspace(np.log10(max(mu[ok].min(), 1e-2)), np.log10(mu[ok].max()), 100)
    axs[0].plot(xs, xs, color=AZUL, lw=2, label="Poisson: var = μ")
    axs[0].plot(xs, xs + alfa_med * xs ** 2, color=VERMELHO, lw=2, label=f"Binomial negativa: var = μ + {alfa_med:.2f}μ²")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("média (contagem normalizada)")
    axs[0].set_ylabel("variância")
    axs[0].set_title("Média x variância por feature")
    axs[0].legend(fontsize=8)
    axs[1].hist(np.log10(alfa[mu > 1]), bins=50, color=VERMELHO, alpha=0.8)
    axs[1].set_xlabel("log10(dispersão α)")
    axs[1].set_ylabel("features")
    axs[1].set_title(f"Dispersão (mediana α={alfa_med:.2f} para μ>10)")
    zeros = (Ce == 0).mean(axis=1)
    mu_bruto = Ce.mean(axis=1)
    axs[2].scatter(mu_bruto, zeros, s=2, alpha=0.3, color=CINZA, rasterized=True, label="observado")
    mm = np.logspace(-2, np.log10(max(1, mu_bruto.max())), 200)
    axs[2].plot(mm, np.exp(-mm), color=AZUL, lw=2, label="esperado Poisson")
    axs[2].plot(mm, (1 + alfa_med * mm) ** (-1 / alfa_med), color=VERMELHO, lw=2, label="esperado BN")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("média (contagem bruta)")
    axs[2].set_ylabel("fração de zeros")
    axs[2].set_title("Excesso de zeros?")
    axs[2].legend(fontsize=8)
    f4 = salvar(fig, pasta, "sobredispersao_zeros")
    # zeros observados x esperados pela BN gene a gene
    esperado_bn = (1 + alfa * mu_bruto) ** (-1 / alfa)
    excesso = float(np.nanmedian((zeros - esperado_bn)[mu_bruto > 1]))
    pd.DataFrame({"media": mu, "variancia": var, "dispersao_alfa": alfa, "fracao_zeros": zeros}).to_csv(
        os.path.join(pasta, "dispersao_por_feature.csv"))

    res = {"amostras": int(C.shape[1]), "features": int(C.shape[0]), "features_expressas": int(len(expressos)),
           "biblioteca_mediana": float(lib.median()), "biblioteca_min": float(lib.min()), "biblioteca_max": float(lib.max()),
           "dispersao_mediana": alfa_med, "excesso_zeros_mediano": excesso,
           "pct_top20": float(top20.median())}
    if comp is not None and "protein_coding" in comp.index:
        res["fracao_protein_coding"] = float(comp.loc["protein_coding"].median())

    # ---------------- exploração em log-CPM
    cpm = C.div(lib, axis=1) * 1e6
    manter = (cpm >= 1).mean(axis=1) >= 0.2
    logcpm = np.log2(cpm.loc[manter] + 1)
    om.extra["logcpm"] = logcpm
    rel.kpi("tamanho mediano", fmt(lib.median() / 1e6) + " M")
    rel.kpi("dispersão mediana α", fmt(alfa_med, 2))
    rel.achado(f"Bibliotecas de {fmt(lib.min() / 1e6)} a {fmt(lib.max() / 1e6)} milhões de leituras (mediana "
               f"{fmt(lib.median() / 1e6)}). {fmt(len(expressos), 0)} de {fmt(C.shape[0], 0)} features têm leitura em "
               "pelo menos 10% das amostras.")
    rel.achado(f"A variância cresce bem acima da média: a dispersão mediana da binomial negativa é α={alfa_med:.2f} "
               f"(para features com média &gt; 10). Isso descarta um modelo Poisson e indica binomial negativa "
               "(ou Poisson-lognormal) na verossimilhança.")
    if excesso > 0.05:
        rel.achado(f"Há excesso de zeros em relação à binomial negativa (mediana de +{pct(excesso)} por feature); "
                   "vale testar um modelo com inflação de zeros ou um componente de detecção.")
    else:
        rel.achado(f"A fração de zeros é compatível com a binomial negativa (diferença mediana {pct(excesso)}); "
                   "inflação de zeros não parece necessária.")
    rel.achado(f"Em média {pct(top20.median())} das leituras de cada amostra vão para as 20 features mais abundantes"
               + (f" ({', '.join(str(nomes[g]) for g in top_dom.index[:3] if nomes is not None and isinstance(nomes.get(g), str))})." if nomes is not None else "."))
    if "fracao_protein_coding" in res:
        rel.achado(f"Genes codificadores de proteína recebem {pct(res['fracao_protein_coding'])} das leituras (mediana).")

    rel.secao("Biblioteca")
    rel.figura(f1, "<b>Como ler:</b> número total de leituras atribuídas a features em cada amostra e quantas features "
               "passam de 10 leituras. Bibliotecas muito menores que as demais pedem atenção (ou um fator de "
               "tamanho no modelo).", pasta)
    if f2:
        rel.figura(f2, "<b>Como ler:</b> fração das leituras por tipo de gene em cada amostra. Muito mitocondrial ou "
                   "rRNA indica degradação ou problema de preparo.", pasta)
    rel.figura(f3, "<b>Como ler:</b> features que concentram mais leituras. Quando poucos genes dominam (ex.: albumina "
               "em fígado), a normalização por total de leituras fica enviesada; prefira TMM ou mediana das razões.", pasta)
    rel.secao("Sobredispersão e zeros")
    rel.figura(f4, "<b>Como ler:</b> à esquerda, se os pontos seguissem a linha azul o Poisson bastaria; a curva "
               "vermelha é a binomial negativa ajustada. No meio, a distribuição da dispersão por feature (para "
               "priors hierárquicos). À direita, a fração de zeros observada contra o esperado por cada modelo.", pasta)
    rel.secao("Exploração em log2(CPM+1)", f"Features com CPM ≥ 1 em pelo menos 20% das amostras "
              f"({fmt(int(manter.sum()), 0)} features).")
    res["exploracao"] = matriz.explorar(ctx, logcpm, chave, rel, pasta, "log2(CPM+1)",
                                        "genes" if tipo == "rna" else "miRNAs", prefixo="logcpm_")
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo[chave] = res
    return res
