"""Mutações somáticas (MC3): carga mutacional, tipos de efeito, espectro de substituições e oncoplot."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import stats

from .. import config
from ..graficos import AZUL, CINZA, LARANJA, VERDE, VERMELHO, ROXO, salvar, tabela_html
from ..relatorio import Relatorio, fmt, pct

COR_EFEITO = {
    "Missense_Mutation": VERDE, "Nonsense_Mutation": VERMELHO, "Frame_Shift_Del": "#1d3557",
    "Frame_Shift_Ins": ROXO, "In_Frame_Del": "#e9c46a", "In_Frame_Ins": "#f4a261", "Splice_Site": LARANJA,
    "Translation_Start_Site": "#9c6644", "Nonstop_Mutation": "#2a9d8f", "Silent": "#cfd8dc",
    "Múltiplas": "#000000",
}
SUBST = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
COR_SUBST = {"C>A": "#03bcee", "C>G": "#010101", "C>T": "#e32926", "T>A": "#cac9c9", "T>C": "#a1ce63", "T>G": "#ebc6c4"}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def classe_subst(ref: str, alt: str):
    if len(ref) != 1 or len(alt) != 1 or ref not in COMP or alt not in COMP or ref == alt:
        return None
    if ref in ("G", "A"):
        ref, alt = COMP[ref], COMP[alt]
    return f"{ref}>{alt}"


def analisar(ctx, om, amostras_seq) -> dict:
    pasta = ctx.pasta_omica("mutacao")
    rel = Relatorio(f"{ctx.cancer} · Mutações somáticas", "MC3 v0.2.8 (PanCanAtlas), exoma, chamadas consenso")
    rel.fonte(om.fonte)
    mut = om.extra["mutacoes"].copy()
    mut["DNA_VAF"] = pd.to_numeric(mut["DNA_VAF"], errors="coerce")
    mut["nao_silenciosa"] = mut["effect"].isin(config.EFEITOS_NAO_SILENCIOSOS)
    amostras = sorted(set(amostras_seq) | set(mut["sample"]))
    n = len(amostras)

    # ---------------- carga mutacional
    cont = mut.groupby("sample").agg(total=("gene", "size"), nao_silenciosas=("nao_silenciosa", "sum"))
    cont = cont.reindex(amostras).fillna(0).astype(int)
    cont["TMB_por_Mb"] = cont["nao_silenciosas"] / config.TAMANHO_EXOMA_MB
    cont.to_csv(os.path.join(pasta, "carga_mutacional_por_amostra.csv"))
    ctx.metricas.setdefault("TMB_por_Mb", {}).update(cont["TMB_por_Mb"].to_dict())
    tmb_med = float(cont["TMB_por_Mb"].median())
    hiper = cont.index[cont["TMB_por_Mb"] >= 10].tolist()

    efeitos = mut["effect"].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 1]})
    o = cont.sort_values("TMB_por_Mb", ascending=False)
    tab_ef = mut.pivot_table(index="sample", columns="effect", values="gene", aggfunc="size").reindex(o.index).fillna(0)
    base = np.zeros(len(o))
    for ef in [e for e in efeitos.index if e in tab_ef.columns][:10]:
        a1.bar(range(len(o)), tab_ef[ef], bottom=base, color=COR_EFEITO.get(ef, CINZA), width=0.9, label=ef)
        base += tab_ef[ef].values
    a1.set_yscale("symlog", linthresh=10)
    a1.set_xticks([])
    a1.set_xlabel(f"amostras ordenadas pela carga (n={n})")
    a1.set_ylabel("mutações")
    a1.set_title(f"Mutações por amostra (TMB mediano {fmt(tmb_med, 2)}/Mb)")
    a1.legend(fontsize=7, ncol=2)
    ev = efeitos.iloc[:12][::-1]
    a2.barh(ev.index, ev.values, color=[COR_EFEITO.get(e, CINZA) for e in ev.index])
    for i, v in enumerate(ev.values):
        a2.text(v, i, f" {v}", va="center", fontsize=8)
    a2.set_title("Tipos de efeito (todas as amostras)")
    a2.grid(axis="y", visible=False)
    f1 = salvar(fig, pasta, "carga_mutacional_e_efeitos")

    # ---------------- espectro de substituições
    snv = mut.assign(classe=[classe_subst(r, a) for r, a in zip(mut["reference"], mut["alt"])]).dropna(subset=["classe"])
    esp = snv["classe"].value_counts().reindex(SUBST).fillna(0)
    por = snv.pivot_table(index="sample", columns="classe", values="gene", aggfunc="size").reindex(columns=SUBST).fillna(0)
    por = por.div(por.sum(axis=1).replace(0, np.nan), axis=0).reindex(o.index).fillna(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    a1.bar(esp.index, esp.values / max(1, esp.sum()) * 100, color=[COR_SUBST[s] for s in esp.index])
    a1.set_ylabel("% das SNVs")
    a1.set_title(f"Espectro de substituições (n={int(esp.sum())} SNVs)")
    base = np.zeros(len(por))
    for s in SUBST:
        a2.bar(range(len(por)), por[s], bottom=base, color=COR_SUBST[s], width=1, label=s)
        base += por[s].values
    a2.set_xticks([])
    a2.set_ylim(0, 1)
    a2.set_xlabel("amostras (mesma ordem da carga mutacional)")
    a2.set_ylabel("fração")
    a2.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    f2 = salvar(fig, pasta, "espectro_substituicoes")

    # ---------------- genes mais mutados e oncoplot
    ns = mut[mut["nao_silenciosa"]]
    freq = ns.groupby("gene")["sample"].nunique().sort_values(ascending=False)
    freq_pct = freq / max(1, n)
    tg = pd.DataFrame({"amostras_mutadas": freq, "% das amostras": (freq_pct * 100).round(1)})
    tg.to_csv(os.path.join(pasta, "genes_mais_mutados.csv"))
    top = list(freq.index[:25])
    f3 = None
    if top:
        efe = ns[ns["gene"].isin(top)].groupby(["gene", "sample"])["effect"].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else "Múltiplas")
        mat = efe.unstack().reindex(index=top, columns=amostras)
        bin_ = mat.notna().astype(int)
        ordem_am = sorted(amostras, key=lambda s: tuple(-bin_[s].values), reverse=False)
        mat = mat[ordem_am]
        cats = list(COR_EFEITO)
        cod = mat.apply(lambda c: c.map(lambda v: cats.index(v) + 1 if v in cats else (0 if pd.isna(v) else len(cats))))
        cmap = ListedColormap(["#f1f3f6"] + [COR_EFEITO[c] for c in cats] + [CINZA])
        fig = plt.figure(figsize=(min(18, 5 + 0.12 * n), 0.32 * len(top) + 3))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[6, 1], hspace=0.05, wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
        ax0.bar(range(len(ordem_am)), cont.loc[ordem_am, "nao_silenciosas"], color=CINZA, width=0.9)
        ax0.set_ylabel("não silenc.")
        ax0.tick_params(axis="x", bottom=False, labelbottom=False)
        ax0.set_title(f"Oncoplot: {len(top)} genes mais mutados (mutações não silenciosas)")
        ax1.imshow(cod.values, aspect="auto", cmap=cmap, vmin=0, vmax=len(cats) + 1, interpolation="nearest")
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top, fontsize=8, style="italic")
        ax1.set_xticks([])
        ax1.set_xlabel("amostras")
        ax1.grid(False)
        ax2.barh(range(len(top)), freq_pct[top].values * 100, color=AZUL, height=0.7)
        for i, v in enumerate(freq_pct[top].values * 100):
            ax2.text(v, i, f" {v:.0f}%", va="center", fontsize=7)
        ax2.tick_params(axis="y", left=False, labelleft=False)
        ax2.set_xlabel("% amostras")
        ax2.grid(axis="y", visible=False)
        presentes = [c for c in cats if (mat == c).any().any()]
        fig.legend(handles=[Patch(color=COR_EFEITO[c], label=c) for c in presentes], loc="lower center",
                   ncol=min(6, len(presentes)), fontsize=8, bbox_to_anchor=(0.45, -0.04))
        f3 = salvar(fig, pasta, "oncoplot")

    # ---------------- VAF e idade
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    a1.hist(mut["DNA_VAF"].dropna(), bins=40, color=AZUL, edgecolor="white")
    a1.set_xlabel("fração alélica (VAF)")
    a1.set_ylabel("mutações")
    a1.set_title(f"VAF (mediana {fmt(mut['DNA_VAF'].median(), 2)})")
    anot = ctx.anotar(cont.index)
    idade = pd.to_numeric(anot["idade"], errors="coerce")
    ok = idade.notna()
    rho = p_idade = np.nan
    if ok.sum() >= 8:
        rho, p_idade = stats.spearmanr(idade[ok], cont.loc[ok, "TMB_por_Mb"])
        a2.scatter(idade[ok], cont.loc[ok, "TMB_por_Mb"], s=18, color=VERMELHO, alpha=0.7)
        a2.set_yscale("symlog", linthresh=0.1)
        a2.set_title(f"TMB x idade (Spearman ρ={rho:.2f}, p={p_idade:.2g})")
    a2.set_xlabel("idade ao diagnóstico")
    a2.set_ylabel("TMB (/Mb)")
    f4 = salvar(fig, pasta, "vaf_e_idade")

    # TMB por estádio
    f5 = None
    g = anot["estadio"]
    grupos = [k for k in ["I", "II", "III", "IV"] if (g == k).sum() >= 2]
    p_est = np.nan
    if len(grupos) >= 2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        dados = [cont.loc[g == k, "TMB_por_Mb"] for k in grupos]
        ax.boxplot(dados, showfliers=False)
        ax.set_xticks(range(1, len(grupos) + 1))
        ax.set_xticklabels(grupos)
        for i, d in enumerate(dados, 1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d, s=12, color=VERMELHO, alpha=0.6)
        p_est = stats.kruskal(*dados).pvalue if all(len(d) >= 2 for d in dados) else np.nan
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"TMB por estádio (Kruskal p={p_est:.2g})")
        ax.set_ylabel("TMB (/Mb)")
        f5 = salvar(fig, pasta, "tmb_por_estadio")

    res = {"amostras": n, "mutacoes": int(len(mut)), "nao_silenciosas": int(mut["nao_silenciosa"].sum()),
           "tmb_mediano": tmb_med, "hipermutados": hiper, "top_genes": {k: float(v) for k, v in freq_pct.iloc[:20].items()},
           "espectro": (esp / max(1, esp.sum())).round(4).to_dict(), "rho_idade": float(rho) if rho == rho else None}

    rel.kpi("amostras sequenciadas", n)
    rel.kpi("mutações", len(mut))
    rel.kpi("TMB mediano (/Mb)", fmt(tmb_med, 2))
    rel.kpi("hipermutados (≥10/Mb)", len(hiper))
    rel.achado(f"{fmt(len(mut), 0)} mutações em {n} tumores, das quais {fmt(int(mut['nao_silenciosa'].sum()), 0)} "
               f"alteram proteína. A carga mutacional mediana é {fmt(tmb_med, 2)} mutações não silenciosas por Mb "
               f"(exoma de ~{config.TAMANHO_EXOMA_MB:.0f} Mb).")
    if hiper:
        rel.achado(f"{len(hiper)} amostra(s) hipermutada(s) (≥ 10/Mb): {', '.join(hiper[:8])}.")
    if len(freq):
        rel.achado("Genes mais mutados: " + ", ".join(f"<i>{g}</i> ({fmt(100 * freq_pct[g], 0)}%)" for g in freq.index[:8]) + ".")
    if esp.sum():
        dom = esp.idxmax()
        rel.achado(f"A substituição mais frequente é {dom} ({pct(esp[dom] / esp.sum())} das SNVs). Predomínio de C>T "
                   "costuma refletir desaminação de citosinas metiladas (assinatura de envelhecimento); C>A sugere "
                   "tabaco ou dano oxidativo; T>C pode indicar exposições específicas.")
    if rho == rho:
        rel.achado(f"Correlação entre idade e TMB: ρ={rho:.2f} (p={p_idade:.2g}).")
    rel.nota("O MC3 inclui só amostras tumorais com exoma sequenciado e só mutações que passaram nos filtros de "
             "consenso. Amostras sem nenhuma mutação não aparecem no arquivo e entram aqui com carga zero quando "
             "estão na lista de amostras.")

    rel.secao("Carga mutacional")
    rel.figura(f1, "<b>Como ler:</b> à esquerda, cada barra é um tumor, empilhada pelo tipo de efeito (escala log). À "
               "direita, o total de mutações por tipo de efeito na coorte.", pasta)
    if f5:
        rel.figura(f5, "<b>Como ler:</b> carga mutacional por estádio patológico.", pasta)
    rel.secao("Espectro de substituições")
    rel.figura(f2, "<b>Como ler:</b> as seis classes de substituição de base, com a base de referência convertida para "
               "pirimidina (C ou T). À direita, a composição de cada tumor.", pasta)
    rel.secao("Genes mais mutados")
    if f3:
        rel.figura(f3, "<b>Como ler:</b> cada coluna é um tumor e cada linha um gene; a cor indica o tipo de mutação. "
                   "No topo, o número de mutações não silenciosas do tumor; à direita, a frequência do gene.", pasta)
    rel.tabela(tabela_html(tg.head(30)), "Genes com mais tumores mutados")
    rel.secao("Fração alélica e idade")
    rel.figura(f4, "<b>Como ler:</b> a fração alélica (VAF) mostra a proporção de leituras com a mutação; picos perto "
               "de 0,5 indicam mutações clonais heterozigotas em tumores puros. À direita, carga mutacional por idade.", pasta)
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo["mutacao"] = res
    return res
