"""Comparações entre cânceres. Lê o que cada câncer já gravou em resultados/<CANCER> e monta resultados/PANCAN.

Pode ser rodado a qualquer momento: usa todos os cânceres processados até agora.
"""
from __future__ import annotations

import html
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .analises.clinico import CAMPOS_CDR, matriz_estado
from .analises.disponibilidade import desenhar_upset
from .dados import eh_tumor
from .graficos import AZUL, CINZA, COR_ESTADIO, CORES, VERMELHO, estadio_grupo, salvar, tabela_html
from .relatorio import Relatorio

RELACOES = [("pancan_vs_gdc", "rho_amostra_mediana", "RNA PanCan x GDC (por paciente)"),
            ("cnv_mrna", "mediana_rho", "CNV x mRNA"), ("metilacao_mrna", "mediana_rho", "Metilação x mRNA"),
            ("atac_mrna", "mediana_rho", "ATAC x mRNA"), ("rppa_mrna", "mediana_rho", "RPPA x mRNA"),
            ("cptac_mrna", "mediana_rho", "CPTAC x mRNA"), ("rppa_cptac", "mediana_rho", "RPPA x CPTAC"),
            ("mirna_mrna", "mediana_rho", "miRNA x mRNA")]
METRICAS = [("TMB_por_Mb", "Carga mutacional (mutações não silenciosas/Mb)", True),
            ("fracao_genoma_alterada", "Fração dos genes com ganho ou perda", False),
            ("beta_medio_global", "Metilação global (beta médio)", False),
            ("biblioteca_rna_milhoes", "Tamanho da biblioteca RNA-seq (milhões de leituras)", False)]


def _ler(caminho, **kw):
    try:
        return pd.read_csv(caminho, **kw) if os.path.exists(caminho) else None
    except Exception:
        return None


def canceres_processados(dir_res: str) -> list[str]:
    out = []
    for c in sorted(os.listdir(dir_res)) if os.path.isdir(dir_res) else []:
        if c != "PANCAN" and os.path.exists(os.path.join(dir_res, c, "resumo.json")):
            out.append(c)
    return out


def _boxplot_por_cancer(ax, dados: dict, titulo: str, log: bool = False, cor=AZUL):
    dados = {k: v.dropna() for k, v in dados.items() if len(v.dropna())}
    ordem = sorted(dados, key=lambda k: dados[k].median())
    vals = [dados[k] for k in ordem]
    ax.boxplot(vals, showfliers=False, widths=0.6)
    rng = np.random.default_rng(0)
    for i, v in enumerate(vals, 1):
        ax.scatter(rng.normal(i, 0.07, len(v)), v, s=5, color=cor, alpha=0.35, rasterized=True)
    ax.set_xticks(range(1, len(ordem) + 1))
    ax.set_xticklabels([f"{k}\n({len(dados[k])})" for k in ordem], fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_title(titulo)
    return ordem


def gerar(dir_res: str, log) -> str | None:
    canceres = canceres_processados(dir_res)
    if not canceres:
        log("Nenhum câncer processado ainda; pan-câncer não gerado.")
        return None
    pasta = os.path.join(dir_res, "PANCAN")
    os.makedirs(pasta, exist_ok=True)
    resumos = {}
    for c in canceres:
        with open(os.path.join(dir_res, c, "resumo.json"), encoding="utf-8") as f:
            resumos[c] = json.load(f)
    completos = [c for c in canceres if resumos[c].get("modo") == "completo"]
    rel = Relatorio("Pan-câncer · comparação entre os cânceres processados",
                    f"{len(canceres)} cânceres: {', '.join(canceres)}")
    rel.kpi("cânceres", len(canceres))
    rel.kpi("com análise molecular completa", len(completos))

    # ---------------- disponibilidade
    qts, tabs = [], []
    for c in canceres:
        q = _ler(os.path.join(dir_res, c, "00_disponibilidade", "quantidade_por_omica.csv"))
        t = _ler(os.path.join(dir_res, c, "00_disponibilidade", "disponibilidade_por_paciente.csv"), index_col=0)
        if q is not None:
            qts.append(q.assign(cancer=c))
        if t is not None:
            tabs.append(t.assign(cancer=c))
    if qts:
        Q = pd.concat(qts)
        piv = Q.pivot_table(index="cancer", columns="chave", values="pacientes (tumor)", aggfunc="first")
        piv = piv[[o["chave"] for o in config.OMICAS if o["chave"] in piv.columns]]
        pivn = Q.pivot_table(index="cancer", columns="chave", values="amostras normais", aggfunc="first")[piv.columns]
        piv.columns = [config.ROTULO[c] for c in piv.columns]
        pivn.columns = piv.columns
        piv.to_csv(os.path.join(pasta, "pacientes_por_omica_e_cancer.csv"))
        pivn.to_csv(os.path.join(pasta, "amostras_normais_por_omica_e_cancer.csv"))
        coorte = piv.iloc[:, 0].replace(0, np.nan)
        frac = piv.div(coorte, axis=0)
        fig, ax = plt.subplots(figsize=(1.0 * piv.shape[1] + 3, 0.38 * len(piv) + 2))
        sns.heatmap(frac, annot=piv.astype(int), fmt="d", cmap="Blues", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "fração da coorte"}, linewidths=0.5, linecolor="white", annot_kws={"fontsize": 8})
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=40)
        ax.set_title("Pacientes com amostra tumoral em cada ômica (cor = fração da coorte)")
        f = salvar(fig, pasta, "pacientes_por_omica")
        rel.secao("Quantos pacientes têm cada ômica")
        rel.figura(f, "<b>Como ler:</b> número de pacientes por câncer com amostra tumoral em cada ômica; a cor é a "
                   "fração da coorte do TCGA-CDR. RNA-seq PanCan e contagens do GDC são os mesmos sequenciamentos "
                   "processados de formas diferentes; o mesmo vale para miRNA.", pasta)
        rel.tabela(tabela_html(piv.astype(int), max_linhas=40), "Pacientes (tumor) por ômica")
        rel.tabela(tabela_html(pivn.astype(int), max_linhas=40), "Amostras normais por ômica")
    if tabs:
        T = pd.concat(tabs)
        T.to_csv(os.path.join(pasta, "disponibilidade_todos_pacientes.csv"))
        cols = [o["chave"] for o in config.OMICAS if o["chave"] in T.columns]
        fig, combos = desenhar_upset(T[cols].astype(bool), f"Todos os {len(T)} pacientes dos cânceres processados")
        f = salvar(fig, pasta, "upset_pancancer")
        rel.secao("Combinações de ômicas")
        rel.figura(f, "<b>Como ler:</b> cada coluna é uma combinação exata de ômicas e a barra de cima é o número de "
                   "pacientes que têm exatamente aquela combinação. A barra vermelha marca quem tem todas as ômicas "
                   "presentes. À esquerda, o total de cada ômica.", pasta)
        nucleo = [c for c in ["mutacao", "cnv", "rna", "mirna", "metilacao"] if c in cols]
        n_om = T[cols].drop(columns=["clinico"], errors="ignore").sum(axis=1)
        dist = pd.crosstab(T["cancer"], n_om)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.32 * len(dist) + 2.2))
        esq = np.zeros(len(dist))
        cmap = plt.get_cmap("viridis", max(dist.columns) + 1)
        fr = dist.div(dist.sum(axis=1), axis=0)
        for k in dist.columns:
            a1.barh(dist.index, fr[k], left=esq, color=cmap(k), label=str(k))
            esq += fr[k].values
        a1.set_xlabel("fração dos pacientes")
        a1.set_title("Quantas ômicas moleculares cada paciente tem (cor = nº de ômicas)", pad=30)
        a1.legend(ncol=len(dist.columns), bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0.3,
                  fontsize=8, handlelength=1.2, columnspacing=1.0)
        a1.grid(axis="y", visible=False)
        todos = T.groupby("cancer")[nucleo].apply(lambda d: int(d.all(axis=1).sum()))
        tam = T.groupby("cancer").size()
        a2.barh(tam.index, tam.values, color="#dfe6ee", label="coorte")
        a2.barh(todos.index, todos.values, color=VERMELHO, label="com " + " + ".join(config.ROTULO[c] for c in nucleo))
        a2.set_title("Pacientes com as 5 ômicas do núcleo multiômico", pad=30)
        a2.legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0.3, fontsize=8)
        for a in (a1, a2):
            a.set_ylim(-0.6, len(dist) - 0.4)
        a2.grid(axis="y", visible=False)
        f = salvar(fig, pasta, "omicas_por_paciente")
        rel.figura(f, "<b>Como ler:</b> à esquerda, a distribuição do número de ômicas moleculares por paciente. "
                   "À direita, quantos pacientes de cada câncer têm o núcleo usado na maior parte dos estudos "
                   "multiômicos (mutação, CNV, RNA-seq, miRNA e metilação).", pasta)
        galeria = "".join(
            f'<a href="../{c}/index.html" style="display:inline-block;width:260px;margin:6px;text-align:center">'
            f'<img src="../{c}/00_disponibilidade/upset_combinacoes.png" style="width:100%;border:1px solid #e3e6ec;'
            f'border-radius:6px" loading="lazy"><br>{html.escape(c)}</a>' for c in canceres)
        rel.secao("UpSet de cada câncer")
        rel.bruto(f"<div>{galeria}</div>")

    # ---------------- clínico
    clins = []
    for c in canceres:
        d = _ler(os.path.join(dir_res, c, "01_clinico", "clinico_tcga_cdr.csv"), dtype=str, index_col=0)
        if d is not None:
            clins.append(d.assign(cancer=c))
    if clins:
        C = pd.concat(clins)
        rel.secao("Dados clínicos")
        campos = [c for c in CAMPOS_CDR if c in C.columns]
        est = matriz_estado(C[campos])
        falt = (est == 1).groupby(C["cancer"]).mean() * 100
        falt.to_csv(os.path.join(pasta, "faltantes_clinicos_por_cancer.csv"))
        fig, ax = plt.subplots(figsize=(0.42 * len(campos) + 3, 0.36 * len(falt) + 2.6))
        sns.heatmap(falt, cmap="Reds", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "% ausente"},
                    annot=len(falt) <= 12, fmt=".0f", annot_kws={"fontsize": 7})
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.set_title("Campos clínicos ausentes por câncer (TCGA-CDR)")
        f = salvar(fig, pasta, "faltantes_clinicos")
        rel.figura(f, "<b>Como ler:</b> percentual de pacientes sem o campo. Campos que não se aplicam ao câncer "
                   "(por exemplo, estádio patológico em leucemia) contam à parte e não aparecem como ausentes.", pasta)

        estg = C["ajcc_pathologic_tumor_stage"].map(estadio_grupo)
        tab = pd.crosstab(C["cancer"], estg)
        tab = tab[[k for k in ["I", "II", "III", "IV", "NA"] if k in tab.columns]]
        tab.to_csv(os.path.join(pasta, "estadio_por_cancer.csv"))
        fr = tab.div(tab.sum(axis=1), axis=0)
        idade = pd.to_numeric(C["age_at_initial_pathologic_diagnosis"], errors="coerce")
        fig, axs = plt.subplots(1, 3, figsize=(18, 0.3 * len(tab) + 2.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
        esq = np.zeros(len(fr))
        for k in fr.columns:
            axs[0].barh(fr.index, fr[k], left=esq, color=COR_ESTADIO[k], label="sem estádio" if k == "NA" else k)
            esq += fr[k].values
        axs[0].legend(ncol=5, bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0.3, fontsize=8)
        axs[0].set_title("Estádio patológico (AJCC)", pad=30)
        axs[0].grid(axis="y", visible=False)
        ordem = idade.groupby(C["cancer"]).median().sort_values().index
        axs[1].boxplot([idade[C["cancer"] == k].dropna() for k in ordem], vert=False, showfliers=False)
        axs[1].set_yticks(range(1, len(ordem) + 1))
        axs[1].set_yticklabels(ordem)
        axs[1].set_xlabel("anos")
        axs[1].set_title("Idade ao diagnóstico", pad=30)
        sexo = pd.crosstab(C["cancer"], C["gender"].str.upper()).reindex(fr.index)
        sexo = sexo.div(sexo.sum(axis=1), axis=0)
        esq = np.zeros(len(sexo))
        for k, cor in zip(sexo.columns, [VERMELHO, AZUL, CINZA]):
            axs[2].barh(sexo.index, sexo[k], left=esq, color=cor, label=k.lower())
            esq += sexo[k].fillna(0).values
        axs[2].legend(ncol=3, bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0.3, fontsize=8)
        axs[2].set_title("Sexo", pad=30)
        axs[2].grid(axis="y", visible=False)
        axs[0].set_ylim(-0.6, len(fr) - 0.4)
        axs[1].set_ylim(0.4, len(ordem) + 0.6)
        axs[2].set_ylim(-0.6, len(sexo) - 0.4)
        f = salvar(fig, pasta, "estadio_idade_sexo")
        rel.figura(f, "<b>Como ler:</b> composição de estádio, idade e sexo de cada coorte. Cânceres sem estádio "
                   "patológico (cérebro, sangue, alguns sarcomas) aparecem cinza à esquerda.", pasta)

        try:
            from lifelines import KaplanMeierFitter
            fig, ax = plt.subplots(figsize=(9, 5))
            linhas = []
            for i, c in enumerate(canceres):
                d = C[C["cancer"] == c]
                t = pd.to_numeric(d["OS.time"], errors="coerce") / 365.25
                e = pd.to_numeric(d["OS"], errors="coerce")
                ok = t.notna() & e.notna()
                if ok.sum() < 5:
                    continue
                km = KaplanMeierFitter().fit(t[ok], e[ok], label=c)
                km.plot_survival_function(ax=ax, ci_show=False, color=CORES[i % len(CORES)],
                                          ls="-" if i < len(CORES) else "--")
                s5 = km.survival_function_at_times(5).iloc[0]
                linhas.append({"câncer": c, "pacientes": int(ok.sum()), "óbitos": int(e[ok].sum()),
                               "mediana OS (anos)": km.median_survival_time_, "OS em 5 anos": s5})
            ax.set_xlabel("anos")
            ax.set_ylabel("sobrevida global")
            ax.set_xlim(0, 10)
            ax.legend(ncol=2 if len(canceres) > 10 else 1, fontsize=7, bbox_to_anchor=(1, 1), loc="upper left")
            ax.set_title("Sobrevida global por câncer (Kaplan-Meier)")
            f = salvar(fig, pasta, "sobrevida_por_cancer")
            rel.figura(f, "<b>Como ler:</b> curvas de sobrevida global de cada coorte, até 10 anos.", pasta)
            if linhas:
                S = pd.DataFrame(linhas).set_index("câncer").sort_values("OS em 5 anos")
                S.to_csv(os.path.join(pasta, "sobrevida_por_cancer.csv"))
                rel.tabela(tabela_html(S, max_linhas=40), "Sobrevida global")
        except Exception as e:
            log(f"  pan-câncer: sobrevida falhou: {e}")

    # ---------------- métricas moleculares
    mets = []
    for c in completos:
        m = _ler(os.path.join(dir_res, c, "metricas_por_amostra.csv"), index_col=0)
        if m is not None:
            mets.append(m.assign(cancer=c))
    if mets:
        Mt = pd.concat(mets)
        Mt = Mt[[isinstance(i, str) and eh_tumor(i) for i in Mt.index]]
        Mt.to_csv(os.path.join(pasta, "metricas_por_amostra_tumoral.csv"))
        disp = [(k, t, lg) for k, t, lg in METRICAS if k in Mt.columns and Mt[k].notna().sum() >= 3]
        if disp:
            rel.secao("Métricas moleculares por câncer")
            fig, axs = plt.subplots(len(disp), 1, figsize=(max(8, 0.55 * len(completos) + 3), 3.4 * len(disp)))
            axs = np.atleast_1d(axs)
            for ax, (k, t, lg) in zip(axs, disp):
                d = {c: Mt.loc[Mt["cancer"] == c, k] for c in completos}
                if lg:
                    d = {c: v.where(v > 0, 0.01) for c, v in d.items()}
                _boxplot_por_cancer(ax, d, t, log=lg, cor=VERMELHO if k == "TMB_por_Mb" else AZUL)
            f = salvar(fig, pasta, "metricas_moleculares")
            rel.figura(f, "<b>Como ler:</b> cada ponto é um tumor; cânceres ordenados pela mediana. A carga mutacional "
                       "está em escala log (tumores com zero mutações aparecem em 0,01).", pasta)
            tab = Mt.groupby("cancer")[[k for k, _, _ in disp]].median()
            tab.columns = [t.split(" (")[0] for _, t, _ in disp]
            rel.tabela(tabela_html(tab, max_linhas=40), "Medianas por câncer")

    # ---------------- relações entre ômicas
    linhas = []
    for c in completos:
        comp = resumos[c].get("comparativo") or {}
        r = {"câncer": c}
        for k, campo, rot in RELACOES:
            if k in comp and comp[k].get(campo) is not None:
                r[rot] = comp[k][campo]
        if len(r) > 1:
            linhas.append(r)
    if linhas:
        R = pd.DataFrame(linhas).set_index("câncer")
        R.to_csv(os.path.join(pasta, "relacoes_entre_omicas.csv"))
        fig, ax = plt.subplots(figsize=(0.9 * R.shape[1] + 3, 0.4 * len(R) + 2))
        sns.heatmap(R.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                    cbar_kws={"label": "ρ mediano"})
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Correlação mediana entre camadas do mesmo gene")
        f = salvar(fig, pasta, "relacoes_entre_omicas")
        rel.secao("Relações entre ômicas")
        rel.figura(f, "<b>Como ler:</b> para cada câncer, a correlação de Spearman mediana entre pares gene a gene "
                   "(CNV e mRNA, metilação e mRNA, proteína e mRNA...). Positivo em CNV x mRNA indica efeito de dosagem; "
                   "negativo em metilação x mRNA indica silenciamento.", pasta)
    rvs = []
    for c in completos:
        rv = (resumos[c].get("comparativo") or {}).get("rv")
        if rv:
            M = pd.DataFrame(rv)
            for a in M.index:
                for b in M.columns:
                    if a < b and pd.notna(M.loc[a, b]):
                        rvs.append({"câncer": c, "par": f"{a} x {b}", "RV": M.loc[a, b]})
    if rvs:
        RV = pd.DataFrame(rvs).pivot_table(index="câncer", columns="par", values="RV")
        RV.to_csv(os.path.join(pasta, "rv_entre_omicas_por_cancer.csv"))
        fig, ax = plt.subplots(figsize=(0.55 * RV.shape[1] + 3, 0.4 * len(RV) + 3.5))
        sns.heatmap(RV, annot=len(RV) <= 10, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "RV"},
                    annot_kws={"fontsize": 7})
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.set_title("Similaridade entre ômicas (RV) em cada câncer")
        f = salvar(fig, pasta, "rv_entre_omicas")
        rel.figura(f, "<b>Como ler:</b> quanto a estrutura dos pacientes em uma ômica se parece com a de outra "
                   "(1 = idêntica). Pares com RV baixo trazem informação complementar.", pasta)

    # ---------------- resumo por ômica
    linhas = []
    for c in completos:
        for o in config.OMICAS:
            r = resumos[c].get(o["chave"])
            if not isinstance(r, dict) or o["chave"] in ("clinico",):
                continue
            ex = r.get("exploracao") if isinstance(r.get("exploracao"), dict) else r
            var_pc = ex.get("var_pc") or []
            linhas.append({"câncer": c, "ômica": o["rotulo"], "features": ex.get("features", r.get("genes")),
                           "amostras": r.get("amostras", ex.get("amostras")),
                           "% faltante": None if ex.get("pct_faltante") is None else 100 * ex["pct_faltante"],
                           "PC1 (% var.)": 100 * var_pc[0] if var_pc else None,
                           "tumor x normal (FDR<5%)": ex.get("de_fdr05")})
    if linhas:
        O = pd.DataFrame(linhas)
        O.to_csv(os.path.join(pasta, "resumo_por_omica_e_cancer.csv"), index=False)
        rel.secao("Resumo de cada ômica por câncer")
        rel.tabela(tabela_html(O.set_index("câncer"), max_linhas=400, casas=1))

    links = " · ".join(f'<a href="../{c}/index.html">{c}</a>' for c in canceres)
    rel.nota("Cânceres incluídos: " + links + ". Para acrescentar outros, rode o pipeline com mais cânceres; esta "
             "página é refeita com todos os que já têm resultados.")
    if len(completos) < len(canceres):
        rel.nota("Cânceres só com disponibilidade e clínico (rodados com --so-disponibilidade) entram nas tabelas de "
                 "disponibilidade e clínico, mas não nas métricas moleculares.")
    caminho = rel.salvar(os.path.join(pasta, "index.html"), '<nav><a href="../index.html">← Todos os cânceres</a></nav>')
    return caminho


def indice_geral(dir_res: str):
    """resultados/index.html com a lista de cânceres e o link para o pan-câncer."""
    canceres = canceres_processados(dir_res)
    linhas = []
    for c in canceres:
        with open(os.path.join(dir_res, c, "resumo.json"), encoding="utf-8") as f:
            r = json.load(f)
        disp = r.get("disponibilidade") or {}
        linhas.append({"câncer": f'<a href="{c}/index.html">{c}</a>', "pacientes": disp.get("pacientes"),
                       "ômicas disponíveis": len(disp.get("omicas_disponiveis") or []),
                       "pacientes com todas": disp.get("todas_disponiveis"), "análise": r.get("modo"),
                       "gerado em": r.get("data")})
    rel = Relatorio("Resultados TCGA por câncer", "PanCanAtlas 2018 + contagens brutas do GDC + CPTAC")
    rel.bruto('<p><a href="PANCAN/index.html"><b>→ Comparação pan-câncer</b></a></p>')
    if linhas:
        t = pd.DataFrame(linhas).to_html(classes="tab", border=0, escape=False, index=False)
        rel.tabela(t)
    rel.salvar(os.path.join(dir_res, "index.html"))
