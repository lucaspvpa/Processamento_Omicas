"""Comparações entre ômicas dentro de um câncer.

1. Concordância RNA PanCan x contagens do GDC
2. Efeitos em cis: CNV -> mRNA, metilação -> mRNA, ATAC (promotor) -> mRNA
3. mRNA -> proteína (RPPA e CPTAC) e RPPA x CPTAC
4. miRNA x mRNA
5. Mutação -> expressão do próprio gene
6. Métricas de amostra cruzadas (TMB, fração alterada, metilação global)
7. Integração: similaridade entre ômicas (RV), concordância de agrupamentos (ARI) e agrupamento conjunto
"""
from __future__ import annotations

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .. import config
from ..dados import eh_tumor, paciente
from ..graficos import (AZUL, CINZA, COR_ESTADIO, CORES, VERMELHO, corr_spearman_matriz, embutir_2d, pca,
                        rv_coef, salvar, spearman_linhas, tabela_html)
from ..relatorio import Relatorio, fmt, pct

RPPA_GENE = {
    "X1433EPSILON": "YWHAE", "X1433BETA": "YWHAB", "X1433ZETA": "YWHAZ", "X4EBP1": "EIF4EBP1", "X53BP1": "TP53BP1",
    "ACC1": "ACACA", "ACC": "ACACA", "AKT": "AKT1", "AMPKALPHA": "PRKAA1", "BAK": "BAK1", "BCLXL": "BCL2L1",
    "BECLIN": "BECN1", "BETACATENIN": "CTNNB1", "BIM": "BCL2L11", "CKIT": "KIT", "CMET": "MET", "CMYC": "MYC",
    "CRAF": "RAF1", "CASPASE7CLEAVEDD198": "CASP7", "CAVEOLIN1": "CAV1", "CD31": "PECAM1", "CD49B": "ITGA2",
    "CHK1": "CHEK1", "CHK2": "CHEK2", "CIAP": "BIRC2", "CLAUDIN7": "CLDN7", "COLLAGENVI": "COL6A1",
    "CYCLINB1": "CCNB1", "CYCLIND1": "CCND1", "CYCLINE1": "CCNE1", "CYCLINE2": "CCNE2", "DJ1": "PARK7",
    "ECADHERIN": "CDH1", "ERALPHA": "ESR1", "ERK2": "MAPK1", "FIBRONECTIN": "FN1", "FOXO3A": "FOXO3",
    "GSK3ALPHABETA": "GSK3A", "GSK3": "GSK3B", "HER2": "ERBB2", "HER3": "ERBB3", "HSP70": "HSPA1A",
    "JNK2": "MAPK9", "JNK": "MAPK8", "KU80": "XRCC5", "LKB1": "STK11", "MAPK": "MAPK1", "MEK1": "MAP2K1",
    "MIG6": "ERRFI1", "NCADHERIN": "CDH2", "NFKBP65": "RELA", "PCADHERIN": "CDH3", "P27": "CDKN1B",
    "P38MAPK": "MAPK14", "P38": "MAPK14", "P53": "TP53", "P70S6K1": "RPS6KB1", "P70S6K": "RPS6KB1",
    "P90RSK": "RPS6KA1", "PAI1": "SERPINE1", "PAXILLIN": "PXN", "PI3KP110ALPHA": "PIK3CA", "PKCALPHA": "PRKCA",
    "PKCDELTA": "PRKCD", "PKCPANBETAII": "PRKCB", "PR": "PGR", "PRAS40": "AKT1S1", "S6": "RPS6", "SHC": "SHC1",
    "STAT5ALPHA": "STAT5A", "STATHMIN": "STMN1", "TUBERIN": "TSC2", "VEGFR2": "KDR", "YAP": "YAP1", "YB1": "YBX1",
    "ANNEXINVII": "ANXA7", "BAP1C4": "BAP1", "CD20": "MS4A1", "EIF4G": "EIF4G1", "HEREGULIN": "NRG1",
    "MYOSINIIA": "MYH9", "P21": "CDKN1A", "PI3KP85": "PIK3R1", "RAB11": "RAB11A", "RAPTOR": "RPTOR",
    "SCD1": "SCD", "SF2": "SRSF1", "TAZ": "WWTR1", "TRANSGLUTAMINASE": "TGM2", "P62LCKLIGAND": "SQSTM1",
    "ANNEXIN1": "ANXA1", "GCN5L2": "KAT2A", "ADAR1": "ADAR", "JAB1": "COPS5", "CASPASE8": "CASP8",
    "PARPCLEAVED": "PARP1", "RB": "RB1", "SMAC": "DIABLO", "SNAIL": "SNAI1", "CABL": "ABL1", "CASPASE3": "CASP3",
    "CD26": "DPP4", "IGF1R": "IGF1R", "P16INK4A": "CDKN2A", "SHP2": "PTPN11", "PDL1": "CD274",
    "COMPLEXIISUBUNIT30": "SDHB", "GYGGLYCOGENIN1": "GYG1", "GYS": "GYS1", "HIF1ALPHA": "HIF1A", "PKM2": "PKM",
    "PYGBAB2": "PYGB", "CASPASE9": "CASP9", "LCN2A": "LCN2", "NRF2": "NFE2L2", "PARPAB3": "PARP1",
    "THYMIDILATESYNTHASE": "TYMS", "TTF1": "NKX2-1", "CHROMOGRANINANTERM": "CHGA", "CK5": "KRT5",
    "NAPSINA": "NAPSA", "P63": "TP63", "SYNAPTOPHYSIN": "SYP", "ALPHACATENIN": "CTNNA1", "CJUN": "JUN",
    "OXPHOSCOMPLEXVSUBUNITB": "ATP5F1B", "NDRG1": "NDRG1", "MRE11": "MRE11",
}


ROTULO_METRICA = {"TMB_por_Mb": "log10(TMB + 0,1)", "fracao_genoma_alterada": "fração alterada (CNV)",
                  "beta_medio_global": "beta médio global", "biblioteca_rna_milhoes": "biblioteca RNA (milhões)",
                  "biblioteca_mirna_milhoes": "biblioteca miRNA (milhões)"}


def por_paciente(M: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mantém uma amostra tumoral por paciente (a de menor código) e renomeia as colunas para o paciente."""
    if M is None or M.shape[1] == 0:
        return None
    cols = sorted([c for c in M.columns if eh_tumor(c)])
    vistos, manter = set(), []
    for c in cols:
        if paciente(c) not in vistos:
            vistos.add(paciente(c))
            manter.append(c)
    out = M[manter].copy()
    out.columns = [paciente(c) for c in manter]
    return out[~out.index.duplicated()]


def _hist_cor(ax, r: pd.Series, titulo: str, rotulo: str):
    r = r.dropna()
    ax.hist(r, bins=50, color=AZUL, alpha=0.85)
    ax.axvline(0, color="#1d2433", lw=0.8)
    ax.axvline(r.median(), color=VERMELHO, ls="--", lw=1.2, label=f"mediana {r.median():.2f}")
    ax.set_xlabel(rotulo)
    ax.set_ylabel("pares")
    ax.set_title(titulo)
    ax.legend()


def _cis(rel, pasta, nome_arq, A, B, titulo, xr, explicacao, res, chave, sinal_esperado=1, rotulos=None):
    r = spearman_linhas(A, B).dropna(subset=["rho"])
    if len(r) < 5:
        return None
    r.to_csv(os.path.join(pasta, nome_arq + ".csv"))
    sig = r[r["fdr"] < 0.05]
    res[chave] = {"pares": int(len(r)), "mediana_rho": float(r["rho"].median()),
                  "sig_fdr05": int(len(sig)), "sig_no_sentido_esperado": int((np.sign(sig["rho"]) == sinal_esperado).sum()),
                  "amostras": int(r["n"].median())}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"wspace": 0.5})
    _hist_cor(a1, r["rho"], titulo, xr)
    top = r.sort_values("rho", ascending=sinal_esperado < 0).head(12)
    nomes = [str(rotulos.get(i, i)) if rotulos is not None else str(i) for i in top.index]
    a2.barh(nomes[::-1], top["rho"][::-1], color=VERMELHO if sinal_esperado > 0 else AZUL)
    a2.set_xlabel("Spearman ρ")
    a2.set_title("Pares mais fortes no sentido esperado")
    a2.tick_params(axis="y", labelsize=8)
    a2.grid(axis="y", visible=False)
    f = salvar(fig, pasta, nome_arq)
    rel.figura(f, explicacao + f" Mediana ρ = {r['rho'].median():.2f}; {len(sig)} de {len(r)} pares com FDR &lt; 5% "
               f"(n mediano = {int(r['n'].median())} pacientes).", pasta)
    return r


def analisar(ctx, omicas: dict, resumo_omicas: dict) -> dict:
    pasta = ctx.pasta_omica("20_comparativo")
    rel = Relatorio(f"{ctx.cancer} · Comparações entre ômicas", "Relações entre camadas moleculares nos mesmos pacientes")
    res = {}
    P = {k: por_paciente(om.matriz) for k, om in omicas.items() if om is not None and om.matriz is not None
         and k not in ("mutacao",)}
    if "rna_contagem" in omicas and omicas["rna_contagem"] is not None and "logcpm" in omicas["rna_contagem"].extra:
        P["rna_logcpm"] = por_paciente(omicas["rna_contagem"].extra["logcpm"])
    rna = P.get("rna")
    if rna is None and P.get("rna_logcpm") is not None:
        rna = _logcpm_por_simbolo(P["rna_logcpm"], omicas["rna_contagem"].extra.get("genes"))
    if rna is not None:
        rna = rna.loc[rna.var(axis=1) > 0]

    # ---------------- 1. PanCan x GDC
    rel.secao("RNA-seq PanCan x contagens brutas do GDC")
    if P.get("rna") is not None and P.get("rna_logcpm") is not None and omicas["rna_contagem"].extra.get("genes") is not None:
        g = _logcpm_por_simbolo(P["rna_logcpm"], omicas["rna_contagem"].extra["genes"])
        A, B = P["rna"].align(g, join="inner", axis=None)
        if A.shape[0] > 100 and A.shape[1] >= 3:
            por_amostra = pd.Series({c: stats.spearmanr(A[c], B[c], nan_policy="omit")[0] for c in A.columns})
            por_gene = spearman_linhas(A, B)["rho"]
            res["pancan_vs_gdc"] = {"genes": int(A.shape[0]), "pacientes": int(A.shape[1]),
                                    "rho_amostra_mediana": float(por_amostra.median()),
                                    "rho_gene_mediana": float(por_gene.median())}
            fig, axs = plt.subplots(1, 3, figsize=(16, 4))
            c0 = por_amostra.sort_values().index[len(por_amostra) // 2]
            axs[0].scatter(B[c0], A[c0], s=2, alpha=0.3, color=CINZA, rasterized=True)
            axs[0].set_xlabel("GDC: log2(CPM+1)")
            axs[0].set_ylabel("PanCan: log2(norm+1)")
            axs[0].set_title(f"Paciente mediano {c0}")
            axs[1].hist(por_amostra, bins=20, color=AZUL)
            axs[1].set_title(f"ρ entre fontes por paciente (mediana {por_amostra.median():.3f})")
            axs[1].set_xlabel("Spearman ρ (todos os genes)")
            _hist_cor(axs[2], por_gene, "ρ entre fontes por gene", "Spearman ρ (entre pacientes)")
            f = salvar(fig, pasta, "pancan_vs_gdc")
            rel.figura(f, "<b>Como ler:</b> a mesma amostra quantificada pelo PanCanAtlas (RSEM, hg19, corrigido por "
                       "lote) e pelo GDC (STAR, hg38, contagem bruta em log-CPM). Correlação alta por paciente indica "
                       "que as duas fontes medem o mesmo perfil; correlação baixa por gene aparece em genes pouco "
                       "expressos ou com anotação diferente entre GENCODE v36 e o RSEM antigo.", pasta)
    else:
        rel.texto("Não há as duas fontes de RNA-seq para comparar nesta execução.")

    # ---------------- 2. efeitos em cis
    rel.secao("Efeitos em cis sobre a expressão")
    if rna is not None and P.get("cnv") is not None:
        _cis(rel, pasta, "cnv_vs_mrna", P["cnv"], rna, "CNV x mRNA do mesmo gene", "Spearman ρ",
             "<b>Como ler:</b> para cada gene, a correlação entre número de cópias e expressão nos mesmos pacientes. "
             "Ganhos que aumentam a expressão (efeito de dosagem) deslocam a distribuição para a direita.",
             res, "cnv_mrna")
    if rna is not None and P.get("metilacao") is not None and omicas["metilacao"].extra.get("anot") is not None:
        an = omicas["metilacao"].extra["anot"].reindex(P["metilacao"].index)
        gene = an["gene"].fillna(".").str.split(",").str[0]
        m = P["metilacao"][(gene != ".") & gene.isin(rna.index)]
        if len(m) > 20:
            gm = gene[m.index]
            B = rna.reindex(gm.values)
            B.index = m.index
            _cis(rel, pasta, "metilacao_vs_mrna", m, B, "Metilação do CpG x mRNA do gene anotado", "Spearman ρ",
                 "<b>Como ler:</b> cada par é um CpG variável e o gene anotado a ele. Metilação de promotor costuma "
                 "silenciar o gene (ρ negativo); no corpo do gene a relação pode ser positiva.",
                 res, "metilacao_mrna", sinal_esperado=-1,
                 rotulos={cg: f"{cg} ({g})" for cg, g in gm.items()})
    atac = omicas.get("atac")
    if rna is not None and atac is not None and atac.extra.get("promotor") is not None and atac.extra.get("promotor_map") is not None:
        pm = atac.extra["promotor_map"]
        colg = "gene" if "gene" in pm.columns else pm.columns[0]
        pr = por_paciente(atac.extra["promotor"])
        if pr is not None:
            g = pm[colg].reindex(pr.index).fillna(".").astype(str).str.split(",").str[0]
            m = pr[g.isin(rna.index)]
            if len(m) > 20:
                B = rna.reindex(g[m.index].values)
                B.index = m.index
                _cis(rel, pasta, "atac_vs_mrna", m, B, "Acessibilidade do promotor x mRNA", "Spearman ρ",
                     "<b>Como ler:</b> promotores mais abertos (mais sinal de ATAC) tendem a ter genes mais "
                     "expressos (ρ positivo).", res, "atac_mrna")

    # ---------------- 3. proteína
    rel.secao("Do mRNA à proteína")
    if rna is not None and P.get("rppa") is not None:
        rp = P["rppa"]
        total = [a for a in rp.index if "_p" not in a]
        genes = {a: RPPA_GENE.get(a, a) for a in total}
        genes = {a: g for a, g in genes.items() if g in rna.index}
        if len(genes) >= 5:
            A = rp.loc[list(genes)]
            B = rna.reindex(list(genes.values()))
            B.index = A.index
            _cis(rel, pasta, "rppa_vs_mrna", A, B, "Proteína (RPPA) x mRNA", "Spearman ρ",
                 f"<b>Como ler:</b> {len(genes)} anticorpos de proteína total ligados ao gene correspondente. "
                 "Correlação mRNA-proteína costuma ser moderada (mediana ~0,3-0,5) por regulação pós-transcricional.",
                 res, "rppa_mrna", rotulos={a: (a if a == g else f"{a} ({g})") for a, g in genes.items()})
    if rna is not None and P.get("cptac") is not None:
        _cis(rel, pasta, "cptac_vs_mrna", P["cptac"], rna, "Proteína (CPTAC) x mRNA", "Spearman ρ",
             "<b>Como ler:</b> correlação entre abundância da proteína por espectrometria de massa e o mRNA do mesmo gene.",
             res, "cptac_mrna")
    if P.get("rppa") is not None and P.get("cptac") is not None:
        rp = P["rppa"]
        total = {a: RPPA_GENE.get(a, a) for a in rp.index if "_p" not in a}
        total = {a: g for a, g in total.items() if g in P["cptac"].index}
        if len(total) >= 5:
            B = P["cptac"].reindex(list(total.values()))
            B.index = list(total)
            _cis(rel, pasta, "rppa_vs_cptac", rp.loc[list(total)], B, "RPPA x CPTAC (mesma proteína)", "Spearman ρ",
                 "<b>Como ler:</b> concordância entre as duas tecnologias de proteína.", res, "rppa_cptac")

    # ---------------- 4. miRNA x mRNA
    if rna is not None and P.get("mirna") is not None:
        rel.secao("miRNA x mRNA")
        mi = P["mirna"].loc[P["mirna"].var(axis=1).nlargest(100).index]
        mr = rna.loc[rna.var(axis=1).nlargest(2000).index]
        comuns = mi.columns.intersection(mr.columns)
        if len(comuns) >= 8:
            C = corr_spearman_matriz(mi[comuns], mr[comuns])
            longo = C.stack().rename("rho").reset_index()
            longo.columns = ["miRNA", "gene", "rho"]
            neg = longo.nsmallest(30, "rho")
            neg.to_csv(os.path.join(pasta, "mirna_mrna_pares_negativos.csv"), index=False)
            res["mirna_mrna"] = {"pares": int(len(longo)), "mediana_rho": float(longo["rho"].median()),
                                 "frac_rho_menor_-0.5": float((longo["rho"] < -0.5).mean()), "pacientes": len(comuns)}
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2))
            _hist_cor(a1, longo["rho"], "Todos os pares (100 miRNAs x 2000 genes)", "Spearman ρ")
            sub = C.loc[neg["miRNA"].unique()[:15], neg["gene"].unique()[:20]]
            sns.heatmap(sub, cmap="RdBu_r", center=0, ax=a2, cbar_kws={"label": "ρ"}, xticklabels=True, yticklabels=True)
            a2.tick_params(labelsize=7)
            a2.set_xlabel("")
            a2.set_ylabel("")
            a2.set_title("Pares mais anticorrelacionados")
            f = salvar(fig, pasta, "mirna_vs_mrna")
            rel.figura(f, "<b>Como ler:</b> miRNAs reprimem seus alvos, então pares com ρ negativo forte são candidatos a "
                       "interação miRNA-alvo (confirme em bases como miRTarBase/TargetScan). Com poucos pacientes, "
                       "correlações extremas aparecem por acaso.", pasta)
            rel.tabela(tabela_html(neg.head(15).set_index("miRNA")), "Pares miRNA-gene mais negativos")

    # ---------------- 5. mutação -> expressão
    mut = omicas.get("mutacao")
    if rna is not None and mut is not None and not mut.vazia:
        rel.secao("Mutação e expressão do próprio gene")
        mm = mut.extra["mutacoes"]
        mm = mm[mm["effect"].isin(config.EFEITOS_NAO_SILENCIOSOS)]
        seq = set(paciente(s) for s in mut.extra.get("amostras_seq", mm["sample"]))
        pacs = [p for p in rna.columns if p in seq]
        mutados = mm.assign(p=mm["sample"].str[:12]).groupby("gene")["p"].apply(set)
        expresso = rna[pacs].mean(axis=1).gt(1) & rna[pacs].var(axis=1).gt(0)
        cand = [(g, s & set(pacs)) for g, s in mutados.items()
                if g in rna.index and expresso[g] and 3 <= len(s & set(pacs)) <= len(pacs) - 3]
        cand = sorted(cand, key=lambda x: -len(x[1]))[:8]
        linhas = []
        if cand:
            ncol = min(4, len(cand))
            nlin = int(np.ceil(len(cand) / ncol))
            fig, axs = plt.subplots(nlin, ncol, figsize=(3.2 * ncol + 1, 3.6 * nlin), squeeze=False)
            axs = axs.ravel()
            for ax in axs[len(cand):]:
                ax.axis("off")
            for ax, (g, s) in zip(axs, cand):
                v = rna.loc[g, pacs]
                mt = v[[p in s for p in pacs]]
                wt = v[[p not in s for p in pacs]]
                p = stats.mannwhitneyu(mt, wt).pvalue
                linhas.append({"gene": g, "mutados": len(mt), "selvagens": len(wt),
                               "diferença mediana": float(mt.median() - wt.median()), "p": p})
                ax.boxplot([wt, mt], showfliers=False)
                ax.set_xticks([1, 2])
                ax.set_xticklabels([f"WT\n{len(wt)}", f"mut\n{len(mt)}"])
                ax.scatter(np.random.default_rng(3).normal(2, 0.05, len(mt)), mt, s=12, color=VERMELHO)
                ax.set_title(f"{g}\np={p:.2g}", style="italic")
            for ax in axs[::ncol]:
                ax.set_ylabel("expressão (mRNA)")
            fig.tight_layout()
            f = salvar(fig, pasta, "mutacao_vs_expressao")
            tab = pd.DataFrame(linhas).set_index("gene")
            tab.to_csv(os.path.join(pasta, "mutacao_vs_expressao.csv"))
            res["mutacao_expressao"] = tab.to_dict(orient="index")
            rel.figura(f, "<b>Como ler:</b> expressão do gene em tumores com e sem mutação não silenciosa nele. "
                       "Mutações truncantes (nonsense, frameshift) costumam reduzir a expressão por decaimento do mRNA "
                       "(NMD); mutações ativadoras costumam não mudar a expressão.", pasta)
        else:
            rel.texto("Nenhum gene tem pelo menos 3 tumores mutados e 3 não mutados com RNA-seq.")

    # ---------------- 6. métricas de amostra
    met = pd.DataFrame({k: pd.Series(v) for k, v in ctx.metricas.items()})
    if len(met):
        met = met[[isinstance(i, str) and eh_tumor(i) for i in met.index]]
        met.index = [paciente(i) for i in met.index]
        met = met[~met.index.duplicated()]
        cols = [c for c in met.columns if met[c].notna().sum() >= 5]
        if len(cols) >= 2:
            rel.secao("Métricas de amostra cruzadas")
            met.to_csv(os.path.join(pasta, "metricas_por_paciente.csv"))
            C = met[cols].corr(method="spearman")
            n = len(cols)
            vis = met[cols].copy()
            if "TMB_por_Mb" in vis:
                vis["TMB_por_Mb"] = np.log10(vis["TMB_por_Mb"] + 0.1)  # carga mutacional é muito assimétrica
            fig, axs = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
            for i, j in itertools.product(range(n), range(n)):
                ax = axs[i, j]
                if i == j:
                    ax.hist(vis[cols[i]].dropna(), bins=15, color=AZUL)
                else:
                    ax.scatter(vis[cols[j]], vis[cols[i]], s=8, color=VERMELHO, alpha=0.6)
                    ax.text(0.03, 0.97, f"ρ={C.iloc[i, j]:.2f}", transform=ax.transAxes, fontsize=8, va="top",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
                if j == 0:
                    ax.set_ylabel(ROTULO_METRICA.get(cols[i], cols[i][:22]), fontsize=8)
                if i == n - 1:
                    ax.set_xlabel(ROTULO_METRICA.get(cols[j], cols[j][:22]), fontsize=8)
                ax.tick_params(labelsize=7)
            f = salvar(fig, pasta, "metricas_cruzadas")
            rel.figura(f, "<b>Como ler:</b> relação entre métricas resumidas de cada tumor (carga mutacional, fração do "
                       "genoma alterada, metilação global, tamanho de biblioteca). ρ de Spearman no canto.", pasta)
            res["metricas_cor"] = C.round(3).to_dict()

    # ---------------- 7. integração
    rel.secao("Integração: as ômicas contam a mesma história?")
    blocos = {}
    tops = {"rna": 2000, "rna_logcpm": 2000, "mirna": 300, "metilacao": 5000, "cnv": 2000, "rppa": 300,
            "atac": 2000, "cptac": 2000}
    for k, n_top in tops.items():
        X = P.get(k)
        if X is None or X.shape[1] < 8:
            continue
        X = X.loc[X.isna().mean(axis=1) < 0.3]
        X = X.loc[X.var(axis=1).nlargest(min(n_top, len(X))).index].T
        sc, var, _ = pca(X, 10)
        blocos[k] = sc
    rot = {"rna_logcpm": "RNA contagens GDC", **config.ROTULO}
    if len(blocos) >= 2:
        ks = list(blocos)
        RV = pd.DataFrame(np.nan, index=[rot[k] for k in ks], columns=[rot[k] for k in ks])
        ARI = RV.copy()
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        rotulos = {}
        for k in ks:
            nk = 2 if len(blocos[k]) < 40 else 3
            rotulos[k] = pd.Series(KMeans(nk, n_init=20, random_state=0).fit_predict(blocos[k]), index=blocos[k].index)
        for a, b in itertools.product(ks, ks):
            comuns = blocos[a].index.intersection(blocos[b].index)
            if len(comuns) < 8:
                continue
            RV.loc[rot[a], rot[b]] = rv_coef(blocos[a].loc[comuns].to_numpy(), blocos[b].loc[comuns].to_numpy())
            ARI.loc[rot[a], rot[b]] = adjusted_rand_score(rotulos[a][comuns], rotulos[b][comuns])
        RV.to_csv(os.path.join(pasta, "rv_entre_omicas.csv"))
        ARI.to_csv(os.path.join(pasta, "ari_agrupamentos.csv"))
        res["rv"] = RV.round(3).to_dict()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.heatmap(RV.astype(float), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=a1, cbar_kws={"label": "RV"})
        a1.set_title("Similaridade da estrutura dos pacientes (RV)")
        sns.heatmap(ARI.astype(float), annot=True, fmt=".2f", cmap="magma", vmin=-0.1, vmax=1, ax=a2, cbar_kws={"label": "ARI"})
        a2.set_title("Concordância de agrupamentos k-means (ARI)")
        for a in (a1, a2):
            a.tick_params(axis="x", rotation=45, labelsize=8)
            a.tick_params(axis="y", labelsize=8)
        f = salvar(fig, pasta, "similaridade_entre_omicas")
        rel.figura(f, "<b>Como ler:</b> à esquerda, o coeficiente RV compara as matrizes de distância entre pacientes de "
                   "duas ômicas (1 = mesma estrutura, 0 = sem relação), usando os 10 primeiros componentes de cada uma. "
                   "À direita, o quanto os agrupamentos k-means de cada ômica coincidem (0 = acaso). Ômicas com RV alto "
                   "trazem informação redundante; RV baixo indica informação complementar, que é o que a integração "
                   "multiômica aproveita.", pasta)

        # agrupamento conjunto
        # RNA PanCan e contagens do GDC são o mesmo sequenciamento: entra só um no agrupamento conjunto
        cobertura = sorted([k for k in ks if not (k == "rna_logcpm" and "rna" in ks)], key=lambda k: -len(blocos[k]))
        escolhidas = list(cobertura)
        while len(escolhidas) > 2:
            comuns = set.intersection(*[set(blocos[k].index) for k in escolhidas])
            if len(comuns) >= max(10, 0.5 * len(blocos[cobertura[0]])):
                break
            escolhidas.pop()
        comuns = sorted(set.intersection(*[set(blocos[k].index) for k in escolhidas]))
        if len(comuns) >= 10:
            partes = []
            for k in escolhidas:
                b = blocos[k].loc[comuns]
                b = (b - b.mean()) / np.linalg.svd(b - b.mean(), compute_uv=False)[0]
                partes.append(b.add_prefix(k + "_"))
            J = pd.concat(partes, axis=1)
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            melhor, lab = -1, None
            for kk in range(2, min(6, len(comuns) // 5 + 1)):
                l_ = KMeans(kk, n_init=30, random_state=0).fit_predict(J)
                s = silhouette_score(J, l_)
                if s > melhor:
                    melhor, lab = s, l_
            if lab is None:
                lab = KMeans(2, n_init=30, random_state=0).fit_predict(J)
            grupos = pd.Series(lab, index=comuns, name="grupo_integrado") + 1
            grupos.to_csv(os.path.join(pasta, "grupos_integrados.csv"))
            sc, var, _ = pca(J, 2)
            emb, metodo = (embutir_2d(J.to_numpy()) if len(comuns) >= 50 else
                           (sc[["PC1", "PC2"]].to_numpy(), "PCA"))
            anot = ctx.anotar([c + "-01" for c in comuns])
            anot.index = comuns
            fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
            for g_ in sorted(grupos.unique()):
                m = grupos == g_
                axs[0].scatter(sc.loc[m, "PC1"], sc.loc[m, "PC2"], s=26, color=CORES[g_ - 1], label=f"grupo {g_} ({m.sum()})")
            axs[0].set_title("PCA conjunta por grupo integrado")
            axs[0].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
            axs[0].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
            axs[0].legend()
            fig.suptitle("Ômicas combinadas: " + ", ".join(rot[k] for k in escolhidas), fontsize=9, y=1.02)
            if emb is not None:
                for est, cor in COR_ESTADIO.items():
                    m = (anot["estadio"] == est).values
                    if m.any():
                        axs[1].scatter(emb[m, 0], emb[m, 1], s=26, color=cor, label=f"{'sem estádio' if est == 'NA' else est} ({m.sum()})", edgecolor="#555", lw=0.3)
                axs[1].set_title(f"{metodo} conjunta por estádio")
                axs[1].set_xlabel("PC1" if metodo == "PCA" else f"{metodo} 1")
                axs[1].set_ylabel("PC2" if metodo == "PCA" else f"{metodo} 2")
                axs[1].legend(fontsize=8)
            p_lr = None
            try:
                from lifelines import KaplanMeierFitter
                from lifelines.statistics import multivariate_logrank_test
                cl = ctx.clin.reindex(comuns)
                t = pd.to_numeric(cl["OS.time"], errors="coerce") / 30.44
                e = pd.to_numeric(cl["OS"], errors="coerce")
                ok = t.notna() & e.notna()
                for g_ in sorted(grupos.unique()):
                    m = ok & (grupos == g_)
                    if m.sum() >= 2:
                        KaplanMeierFitter().fit(t[m], e[m], label=f"grupo {g_} (n={m.sum()})").plot_survival_function(
                            ax=axs[2], ci_show=False, color=CORES[g_ - 1])
                p_lr = multivariate_logrank_test(t[ok], grupos[ok], e[ok]).p_value
                axs[2].set_title(f"Sobrevida global por grupo (log-rank p={p_lr:.2g})")
                axs[2].set_xlabel("meses")
            except Exception:
                axs[2].axis("off")
            f = salvar(fig, pasta, "agrupamento_integrado")
            res["integrado"] = {"omicas": escolhidas, "pacientes": len(comuns), "k": int(grupos.nunique()),
                                "silhueta": float(melhor), "p_logrank": None if p_lr is None else float(p_lr)}
            rel.figura(f, f"<b>Como ler:</b> {len(comuns)} pacientes com todas estas ômicas: "
                       f"{', '.join(rot[k] for k in escolhidas)}. Cada ômica foi resumida em 10 componentes e "
                       "ponderada para pesar igual (estilo MFA). O k-means escolheu "
                       f"{grupos.nunique()} grupos pela silhueta ({melhor:.2f}). À direita, a sobrevida de cada grupo. "
                       "É um ponto de partida exploratório para o seu modelo, não um resultado final.", pasta)

    # ---------------- 8. associação clínica por ômica
    linhas = []
    for k, r in resumo_omicas.items():
        a = (r or {}).get("assoc") or ((r or {}).get("exploracao") or {}).get("assoc")
        if not a:
            continue
        for var, d in a.items():
            for pc, p in d.items():
                if pc.startswith("PC1"):
                    linhas.append({"ômica": config.ROTULO.get(k, k), "variável": var, "p": p})
    if linhas:
        T = pd.DataFrame(linhas).pivot_table(index="ômica", columns="variável", values="p")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(T) + 1.8))
        sns.heatmap(-np.log10(T.astype(float)), annot=T.map(lambda x: f"{x:.2g}") if hasattr(T, "map") else True,
                    fmt="", cmap="Reds", vmin=0, vmax=4, ax=ax, cbar_kws={"label": "−log10(p)"})
        ax.set_title("PC1 de cada ômica x variáveis clínicas (p-valor)")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("")
        ax.set_xlabel("")
        f = salvar(fig, pasta, "pc1_vs_clinica")
        rel.secao("Qual ômica se associa à clínica?")
        rel.figura(f, "<b>Como ler:</b> p-valores da associação do primeiro componente principal de cada ômica (só "
                   "tumores) com estádio, sexo, idade e sobrevida global. Quanto mais escuro, mais forte a associação.", pasta)

    _achados(rel, res)
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo["comparativo"] = res
    return res


def _logcpm_por_simbolo(L: pd.DataFrame, genes: pd.DataFrame | None) -> pd.DataFrame:
    if genes is None or "gene_name" not in genes:
        return L
    nomes = genes["gene_name"].reindex(L.index)
    L = L[nomes.notna().values]
    L.index = nomes.dropna().values
    return L.groupby(level=0).mean()


def _achados(rel, res):
    if "pancan_vs_gdc" in res:
        r = res["pancan_vs_gdc"]
        rel.achado(f"PanCan x GDC: correlação mediana por paciente de {r['rho_amostra_mediana']:.3f} em "
                   f"{fmt(r['genes'], 0)} genes; por gene, {r['rho_gene_mediana']:.2f}.")
    for k, t in (("cnv_mrna", "CNV x mRNA"), ("metilacao_mrna", "Metilação x mRNA"), ("atac_mrna", "ATAC x mRNA"),
                 ("rppa_mrna", "RPPA x mRNA"), ("cptac_mrna", "CPTAC x mRNA"), ("rppa_cptac", "RPPA x CPTAC")):
        if k in res:
            r = res[k]
            rel.achado(f"{t}: mediana ρ={r['mediana_rho']:.2f} em {fmt(r['pares'], 0)} pares; {r['sig_fdr05']} com FDR &lt; 5% "
                       f"({r['sig_no_sentido_esperado']} no sentido esperado).")
    if "mirna_mrna" in res:
        r = res["mirna_mrna"]
        rel.achado(f"miRNA x mRNA: {pct(r['frac_rho_menor_-0.5'])} dos pares têm ρ &lt; −0,5.")
    if "integrado" in res:
        r = res["integrado"]
        p = r["p_logrank"]
        rel.achado(f"Agrupamento integrado de {r['pacientes']} pacientes em {r['k']} grupos (silhueta {r['silhueta']:.2f})"
                   + (f"; sobrevida diferente entre grupos com log-rank p={p:.2g}." if p is not None else "."))
