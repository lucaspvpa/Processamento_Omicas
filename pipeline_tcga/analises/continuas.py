"""Ômicas contínuas: RNA-seq PanCan, miRNA, metilação, RPPA, ATAC-seq e proteoma CPTAC."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..graficos import AZUL, COR_TIPO, LARANJA, VERMELHO, salvar
from ..relatorio import Relatorio, fmt, pct
from . import matriz

TEXTO = {
    "rna": ("RNA-seq (PanCanAtlas)", "Illumina HiSeq, RSEM normalizado e corrigido por lote (EB++), log2(x+1)",
            "log2(norm+1)", "genes",
            "Estes valores já vêm normalizados e com correção de lote entre plataformas e centros. Servem bem para "
            "modelos gaussianos ou redes neurais; para um modelo de contagem use as contagens brutas do GDC."),
    "mirna": ("miRNA-seq (PanCanAtlas)", "miRNA maduro, normalizado e corrigido por lote (EB), log2(x+1)",
              "log2(norm+1)", "miRNAs",
              "Fita madura (5p/3p) já normalizada. As contagens brutas por precursor estão no relatório de contagens."),
    "rppa": ("Proteínas por RPPA", "Reverse Phase Protein Array (MD Anderson), normalizado e corrigido por lote",
             "valor normalizado", "anticorpos",
             "O RPPA mede cerca de 200 proteínas e fosfoproteínas escolhidas por relevância em vias de sinalização. "
             "Anticorpos com sufixo _pS/_pT/_pY medem a forma fosforilada. Os valores são relativos (centrados), "
             "não absolutos."),
    "atac": ("Cromatina acessível (ATAC-seq)", "Corces et al., Science 2018; picos em log2((contagem+5)/milhão), normalizados por quantil",
             "log2 CPM", "picos",
             "Só cerca de 400 tumores de 23 cânceres têm ATAC-seq, sem amostras normais. Guardamos os picos mais "
             "variáveis para a exploração."),
    "cptac": ("Proteoma por espectrometria de massa (CPTAC)", "CPTAC, razão log2 em relação a uma referência comum",
              "log2 razão", "proteínas",
              "Proteômica por espectrometria de massa tem muitos valores faltantes (proteínas abaixo do limite de "
              "detecção). Os faltantes não são aleatórios: tendem a ser proteínas pouco abundantes."),
}


def analisar(ctx, om) -> dict:
    chave = om.chave
    titulo, sub, unidade, nome_feat, nota = TEXTO[chave]
    pasta = ctx.pasta_omica(chave)
    rel = Relatorio(f"{ctx.cancer} · {titulo}", sub)
    rel.fonte(om.fonte)
    rel.nota(nota)
    for n_ in om.notas:
        rel.nota(n_)
    M = om.matriz
    n_top = 2000 if chave in ("rna", "atac", "cptac") else min(500, M.shape[0])
    res = matriz.explorar(ctx, M, chave, rel, pasta, unidade, nome_feat, n_top=n_top,
                          diferencial=chave != "atac")
    if chave == "rna":
        expr = (M > 1).mean()
        res["genes_expressos_mediana"] = float(expr.median())
        rel.achado(f"Em mediana, {pct(expr.median())} dos genes têm expressão acima de log2(norm+1)=1 em cada amostra.")
    if chave in ("mirna", "rna"):
        top = M.mean(axis=1).nlargest(15)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(top.index[::-1], top.values[::-1], color=AZUL)
        ax.set_xlabel(f"média ({unidade})")
        ax.set_title(f"{nome_feat.capitalize()} mais abundantes")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
        f = salvar(fig, pasta, "mais_abundantes")
        rel.secao("Mais abundantes")
        rel.figura(f, f"<b>Como ler:</b> os {nome_feat} com maior média em todas as amostras.", pasta)
        res["mais_abundantes"] = list(top.index[:10])
    if chave == "rppa":
        fosfo = [a for a in M.index if "_p" in a]
        res["fosfo"] = len(fosfo)
        rel.achado(f"{len(fosfo)} dos {M.shape[0]} anticorpos medem formas fosforiladas.")
    if chave == "cptac":
        falt = M.isna().mean(axis=1)
        res["proteinas_completas"] = int((falt == 0).sum())
        rel.achado(f"{fmt(int((falt == 0).sum()), 0)} proteínas foram quantificadas em todas as amostras; "
                   f"{fmt(int((falt > 0.5).sum()), 0)} faltam em mais da metade.")
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo[chave] = res
    return res


def analisar_metilacao(ctx, om) -> dict:
    pasta = ctx.pasta_omica("metilacao")
    rel = Relatorio(f"{ctx.cancer} · Metilação do DNA (450k)", "Illumina Infinium HumanMethylation450, valores beta (0 a 1)")
    rel.fonte(om.fonte)
    for n_ in om.notas:
        rel.nota(n_)
    rel.nota(f"O array tem ~485 mil CpGs. Para não ocupar espaço, guardamos estatísticas de todos os CpGs e só os "
             f"{fmt(config.TOP_CPGS, 0)} mais variáveis para PCA, UMAP e heatmaps.")
    stats_ = om.extra["stats"]
    hist = om.extra["hist"]
    anot_sondas = om.extra.get("anot")
    M = om.matriz
    anot = ctx.anotar(hist.columns)

    # ---------------- distribuição global
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    x = hist.index.astype(float)
    dens = hist / hist.sum()
    for s in dens.columns:
        axs[0].plot(x, dens[s], color=COR_TIPO[anot.loc[s, "tipo"]], alpha=0.35, lw=0.9)
    axs[0].set_xlabel("beta")
    axs[0].set_ylabel("fração dos CpGs")
    axs[0].set_title("Distribuição de beta por amostra")
    axs[1].hist(stats_["media"].dropna(), bins=60, color=AZUL)
    axs[1].set_xlabel("beta médio do CpG")
    axs[1].set_ylabel("CpGs")
    axs[1].set_title(f"Beta médio de todos os {fmt(len(stats_), 0)} CpGs")
    axs[2].hist(stats_["dp"].dropna(), bins=60, color=LARANJA)
    axs[2].axvline(stats_["dp"].nlargest(config.TOP_CPGS).min(), color=VERMELHO, ls="--", lw=1,
                   label=f"corte dos top {fmt(config.TOP_CPGS, 0)}")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("desvio-padrão do beta")
    axs[2].set_title("Variabilidade por CpG")
    axs[2].legend()
    f1 = salvar(fig, pasta, "distribuicao_beta")

    # metilação global por amostra
    glob = (dens.mul(x, axis=0)).sum()
    hiper = dens.loc[x > 0.8].sum()
    ctx.metricas.setdefault("beta_medio_global", {}).update(glob.to_dict())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, v, t in ((axs[0], glob, "Beta médio global"), (axs[1], hiper, "Fração de CpGs com beta > 0,8")):
        grupos = [g for g in ["Tumor", "Normal"] if (anot["tipo"] == g).any()]
        dados = [v[anot["tipo"] == g] for g in grupos]
        ax.boxplot(dados, showfliers=False)
        ax.set_xticks(range(1, len(grupos) + 1))
        ax.set_xticklabels([f"{g} ({len(d)})" for g, d in zip(grupos, dados)])
        for i, (g, d) in enumerate(zip(grupos, dados), 1):
            ax.scatter(np.random.default_rng(2).normal(i, 0.05, len(d)), d, s=14, color=COR_TIPO[g], alpha=0.7)
        ax.set_title(t)
    f2 = salvar(fig, pasta, "metilacao_global")

    falt = stats_["faltante"]
    res_glob = {"cpgs_total": int(len(stats_)), "cpgs_sem_faltante": int((falt == 0).sum()),
                "beta_global_tumor": float(glob[anot["tipo"] == "Tumor"].median()) if (anot["tipo"] == "Tumor").any() else None,
                "beta_global_normal": float(glob[anot["tipo"] == "Normal"].median()) if (anot["tipo"] == "Normal").any() else None,
                "fracao_cpgs_intermediarios": float(((stats_["media"] > 0.2) & (stats_["media"] < 0.8)).mean())}

    # anotação dos CpGs variáveis
    f3 = None
    if anot_sondas is not None and "gene" in anot_sondas:
        a = anot_sondas.reindex(M.index)
        com_gene = a["gene"].notna() & (a["gene"] != ".")
        res_glob["top_cpgs_com_gene"] = float(com_gene.mean())
        crom = a["chrom"].str.replace("chr", "").value_counts()
        ordem = [c for c in [str(i) for i in range(1, 23)] + ["X", "Y"] if c in crom.index]
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(ordem, crom[ordem], color=AZUL)
        ax.set_xlabel("cromossomo")
        ax.set_ylabel("CpGs")
        ax.set_title(f"Onde estão os {fmt(len(M), 0)} CpGs mais variáveis ({pct(com_gene.mean())} anotados a um gene)")
        f3 = salvar(fig, pasta, "cpgs_variaveis_cromossomos")

    rel.kpi("CpGs medidos", res_glob["cpgs_total"])
    rel.kpi("CpGs sem faltante", res_glob["cpgs_sem_faltante"])
    rel.achado("A distribuição de beta é bimodal: a maioria dos CpGs está quase toda desmetilada (perto de 0, típico "
               "de ilhas CpG em promotores) ou quase toda metilada (perto de 1, corpo gênico e regiões intergênicas). "
               f"{pct(res_glob['fracao_cpgs_intermediarios'])} dos CpGs têm beta médio intermediário (0,2 a 0,8).")
    if res_glob["beta_global_tumor"] is not None and res_glob["beta_global_normal"] is not None:
        d = res_glob["beta_global_tumor"] - res_glob["beta_global_normal"]
        rel.achado(f"Beta médio global: {fmt(res_glob['beta_global_tumor'], 3)} nos tumores e "
                   f"{fmt(res_glob['beta_global_normal'], 3)} nos normais "
                   f"({'hipometilação global no tumor, como é comum em câncer' if d < 0 else 'sem hipometilação global no tumor'}).")
    rel.secao("Distribuição global")
    rel.figura(f1, "<b>Como ler:</b> à esquerda, a distribuição de beta em cada amostra (vermelho tumor, azul normal), "
               "calculada sobre todos os CpGs. No meio, o beta médio de cada CpG. À direita, a variabilidade de cada "
               "CpG e o corte usado para escolher os mais variáveis.", pasta)
    rel.figura(f2, "<b>Como ler:</b> metilação média de cada amostra e fração de CpGs muito metilados, tumor x normal.", pasta)
    if f3:
        rel.figura(f3, "<b>Como ler:</b> distribuição por cromossomo dos CpGs mais variáveis.", pasta)
    rel.secao("Exploração dos CpGs mais variáveis")
    res = matriz.explorar(ctx, M, "metilacao", rel, pasta, "beta", "CpGs", n_top=5000)
    res.update(res_glob)
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo["metilacao"] = res
    return res
