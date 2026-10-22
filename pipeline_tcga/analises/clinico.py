"""Dados clínicos: completude dos campos, estádio, distribuição dos metadados e sobrevida."""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ..graficos import AZUL, CINZA, COR_ESTADIO, LARANJA, VERMELHO, barras_categoria, estadio_grupo, salvar, tabela_html
from ..relatorio import Relatorio, fmt, pct

AUSENTE = {"", "nan", "none", "[not available]", "[unknown]", "[not evaluated]", "[discrepancy]",
           "not reported", "unknown", "--", "'--", "not available", "[not reported]", "na", "n/a"}
NAO_SE_APLICA = {"[not applicable]", "not applicable"}
CAMPOS_CDR = ["age_at_initial_pathologic_diagnosis", "gender", "race", "ajcc_pathologic_tumor_stage",
              "clinical_stage", "histological_type", "histological_grade", "initial_pathologic_dx_year",
              "menopause_status", "vital_status", "tumor_status", "last_contact_days_to", "death_days_to",
              "cause_of_death", "new_tumor_event_type", "new_tumor_event_dx_days_to",
              "treatment_outcome_first_course", "margin_status", "residual_tumor",
              "OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]
DESFECHOS = [("OS", "Sobrevida global (OS)"), ("DSS", "Sobrevida específica da doença (DSS)"),
             ("DFI", "Intervalo livre de doença (DFI)"), ("PFI", "Intervalo livre de progressão (PFI)")]


def estado(v) -> int:
    """0 = presente, 1 = ausente, 2 = não se aplica."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 1
    s = str(v).strip().lower()
    if s in NAO_SE_APLICA:
        return 2
    if s in AUSENTE:
        return 1
    return 0


def matriz_estado(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(estado))


def figura_faltantes(est: pd.DataFrame, titulo: str):
    est = est.loc[:, est.columns[np.argsort((est == 1).mean().values, kind="stable")]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, max(4, 0.26 * est.shape[1] + 1.5)),
                                 gridspec_kw={"width_ratios": [2.4, 1]})
    cmap = ListedColormap(["#2f6690", "#e9ecef", "#e9c46a"])
    a1.imshow(est.T.values, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    a1.set_yticks(range(est.shape[1]))
    a1.set_yticklabels([c[:42] for c in est.columns], fontsize=8)
    a1.set_xticks([])
    a1.set_xlabel(f"pacientes (n={est.shape[0]})")
    a1.set_title(titulo)
    a1.grid(False)
    falt = (est == 1).mean() * 100
    nsa = (est == 2).mean() * 100
    y = np.arange(est.shape[1])
    a2.barh(y, falt.values, color=VERMELHO, height=0.6, label="ausente")
    a2.barh(y, nsa.values, left=falt.values, color="#e9c46a", height=0.6, label="não se aplica")
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.set_xlim(0, 100)
    a1.set_ylim(est.shape[1] - 0.5, -0.5)
    a2.set_ylim(est.shape[1] - 0.5, -0.5)
    a2.set_xlabel("% dos pacientes")
    a2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    a2.grid(axis="y", visible=False)
    return fig


def km(ax, tempos, eventos, rotulo, cor):
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(tempos, eventos, label=f"{rotulo} (n={len(tempos)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=cor, show_censors=True,
                               censor_styles={"ms": 4, "marker": "|"})
    return kmf


def analisar(ctx, gdc: pd.DataFrame, fonte_gdc: str) -> dict:
    pasta = ctx.pasta_omica("clinico")
    clin = ctx.clin.copy()
    clin.to_csv(os.path.join(pasta, "clinico_tcga_cdr.csv"))
    if gdc is not None and len(gdc):
        gdc = gdc[gdc.index.isin(clin.index)] if clin.index.isin(gdc.index).any() else gdc
        gdc.to_csv(os.path.join(pasta, "clinico_gdc.csv"))
    rel = Relatorio(f"{ctx.cancer} · Dados clínicos", "TCGA-CDR (PanCanAtlas) e metadados atuais do GDC")
    rel.fonte("TCGA-CDR, Liu et al., Cell 2018 (Survival_SupplementalTable_S1)")
    rel.fonte(f"GDC: {fonte_gdc}")
    n = len(clin)
    rel.kpi("pacientes", n)
    res = {"pacientes": n}

    # ---------------- completude
    campos = [c for c in CAMPOS_CDR if c in clin.columns]
    est = matriz_estado(clin[campos])
    falt_cdr = pd.DataFrame({"% ausente": (est == 1).mean() * 100, "% não se aplica": (est == 2).mean() * 100})
    falt_cdr.index.name = "campo"
    falt_cdr.to_csv(os.path.join(pasta, "faltantes_cdr.csv"))
    res["faltantes_cdr"] = falt_cdr["% ausente"].round(1).to_dict()
    f1 = salvar(figura_faltantes(est, "Completude dos campos do TCGA-CDR"), pasta, "faltantes_cdr")
    criticos = falt_cdr[falt_cdr["% ausente"] > 50].index.tolist()
    completos = falt_cdr[(falt_cdr["% ausente"] == 0) & (falt_cdr["% não se aplica"] == 0)].index.tolist()

    f2 = None
    if gdc is not None and len(gdc):
        admin = re.compile(r"(^|[._])(id|uuid|case_id|sample|submitter_id|state|created_datetime|updated_datetime)([._]|$)"
                           r"|_id\.|xena")
        estg = matriz_estado(gdc[[c for c in gdc.columns if not admin.search(c)]])
        manter = [c for c in estg.columns if (estg[c] != 1).any() or c.endswith("diagnoses")]
        estg = estg[manter]
        falt_gdc = pd.DataFrame({"% ausente": (estg == 1).mean() * 100, "% não se aplica": (estg == 2).mean() * 100})
        falt_gdc.index.name = "campo"
        falt_gdc.sort_values("% ausente").to_csv(os.path.join(pasta, "faltantes_gdc.csv"))
        res["faltantes_gdc"] = falt_gdc["% ausente"].round(1).to_dict()
        if estg.shape[1] > 60:
            estg = estg[falt_gdc.sort_values("% ausente", ascending=False).index[:60]]
        f2 = salvar(figura_faltantes(estg, "Completude dos campos do GDC (atual)"), pasta, "faltantes_gdc")

    # ---------------- estádio
    st = clin.get("ajcc_pathologic_tumor_stage", pd.Series(dtype=str))
    grupos = st.map(estadio_grupo)
    cont_g = grupos.value_counts().reindex(["I", "II", "III", "IV", "NA"]).fillna(0).astype(int)
    res["estadio"] = cont_g.to_dict()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(cont_g.index, cont_g.values, color=[COR_ESTADIO[k] for k in cont_g.index])
    for i, v in enumerate(cont_g.values):
        a1.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    a1.set_title("Estádio patológico AJCC (agrupado)")
    a1.set_ylabel("pacientes")
    a1.set_xticks(range(5))
    a1.set_xticklabels(["I", "II", "III", "IV", "sem\nregistro"])
    barras_categoria(a2, st.where(est.get("ajcc_pathologic_tumor_stage", pd.Series(0, index=st.index)) == 0),
                     "Estádio patológico detalhado", max_cat=14)
    f3 = salvar(fig, pasta, "estadio_patologico")

    tnm = [c for c in gdc.columns if c.startswith(("ajcc_pathologic_t.", "ajcc_pathologic_n.", "ajcc_pathologic_m."))] \
        if gdc is not None and len(gdc) else []
    f3b = None
    if tnm:
        fig, axs = plt.subplots(1, len(tnm), figsize=(4.2 * len(tnm), 3.4))
        axs = np.atleast_1d(axs)
        for ax, c in zip(axs, tnm):
            barras_categoria(ax, gdc[c].where(matriz_estado(gdc[[c]])[c] == 0), c.split(".")[0].replace("ajcc_pathologic_", "AJCC ").upper())
        f3b = salvar(fig, pasta, "tnm_gdc")

    # ---------------- distribuição dos metadados
    idade = pd.to_numeric(clin.get("age_at_initial_pathologic_diagnosis"), errors="coerce")
    res["idade_mediana"] = float(idade.median()) if idade.notna().any() else None
    painel = [("gender", "Sexo"), ("race", "Raça"), ("histological_type", "Tipo histológico"),
              ("histological_grade", "Grau histológico"), ("vital_status", "Estado vital"),
              ("tumor_status", "Status do tumor"), ("treatment_outcome_first_course", "Resposta ao 1º tratamento"),
              ("residual_tumor", "Tumor residual")]
    painel = [p for p in painel if p[0] in clin.columns]
    nlin = int(np.ceil((len(painel) + 1) / 3))
    fig, axs = plt.subplots(nlin, 3, figsize=(15, 3.3 * nlin))
    axs = axs.ravel()
    axs[0].hist(idade.dropna(), bins=15, color=AZUL, edgecolor="white")
    axs[0].set_title(f"Idade ao diagnóstico (mediana {fmt(res['idade_mediana'], 0)} anos)")
    axs[0].set_xlabel("anos")
    for ax, (c, t) in zip(axs[1:], painel):
        barras_categoria(ax, clin[c].where(matriz_estado(clin[[c]])[c] == 0), t)
    for ax in axs[len(painel) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    f4 = salvar(fig, pasta, "distribuicao_metadados")
    for c, _ in painel:
        res[f"dist_{c}"] = clin[c].where(matriz_estado(clin[[c]])[c] == 0).fillna("(ausente)").value_counts().to_dict()

    f4b = None
    if gdc is not None and len(gdc):
        extras = [c for c in ["primary_diagnosis.diagnoses", "tumor_grade.diagnoses", "tissue_or_organ_of_origin.diagnoses",
                              "treatment_type.treatments.diagnoses", "prior_malignancy.diagnoses",
                              "ethnicity.demographic", "alcohol_history.exposures"] if c in gdc.columns]
        if extras:
            nlin = int(np.ceil(len(extras) / 3))
            fig, axs = plt.subplots(nlin, 3, figsize=(15, 3.3 * nlin))
            axs = np.atleast_1d(axs).ravel()
            for ax, c in zip(axs, extras):
                barras_categoria(ax, gdc[c].where(matriz_estado(gdc[[c]])[c] == 0), c.split(".")[0], cor=LARANJA)
            for ax in axs[len(extras):]:
                ax.axis("off")
            fig.tight_layout()
            f4b = salvar(fig, pasta, "distribuicao_metadados_gdc")

    # ---------------- sobrevida
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    surv = {}
    for ax, (d, nome) in zip(axs, DESFECHOS):
        if d not in clin.columns:
            ax.axis("off")
            continue
        t = pd.to_numeric(clin[f"{d}.time"], errors="coerce") / 30.44
        e = pd.to_numeric(clin[d], errors="coerce")
        ok = t.notna() & e.notna()
        if ok.sum() < 3:
            ax.text(0.5, 0.5, "dados insuficientes", ha="center", transform=ax.transAxes)
            ax.set_title(nome)
            continue
        kmf = km(ax, t[ok], e[ok], "todos", AZUL)
        med = kmf.median_survival_time_
        surv[d] = {"n": int(ok.sum()), "eventos": int(e[ok].sum()),
                   "mediana_meses": None if not np.isfinite(med) else float(med),
                   "seguimento_mediano_meses": float(t[ok].median())}
        ax.set_title(nome)
        ax.set_xlabel("meses")
        ax.get_legend().remove() if ax.get_legend() else None
        ax.text(0.98, 0.95, f"n={ok.sum()} · eventos={int(e[ok].sum())}\nmediana: "
                f"{'não atingida' if not np.isfinite(med) else fmt(med, 1) + ' m'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axs[0].set_ylabel("probabilidade")
    f5 = salvar(fig, pasta, "sobrevida_km")
    res["sobrevida"] = surv

    # KM por estádio e por sexo (OS)
    f6, p_estadio, p_sexo = None, None, None
    if "OS" in clin.columns:
        from lifelines.statistics import multivariate_logrank_test
        t = pd.to_numeric(clin["OS.time"], errors="coerce") / 30.44
        e = pd.to_numeric(clin["OS"], errors="coerce")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        g = grupos.replace({"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV"})
        for ax, grp, tit in ((a1, g, "OS por estádio"), (a2, clin.get("gender"), "OS por sexo")):
            if grp is None:
                ax.axis("off")
                continue
            ok = t.notna() & e.notna() & grp.notna() & (grp != "NA")
            cats = [c for c in sorted(grp[ok].unique()) if (grp[ok] == c).sum() >= 3]
            if len(cats) < 2:
                ax.text(0.5, 0.5, "menos de 2 grupos com 3+ pacientes", ha="center", transform=ax.transAxes)
                ax.set_title(tit)
                continue
            ok = ok & grp.isin(cats)
            for c, cor in zip(cats, [AZUL, VERMELHO, LARANJA, CINZA]):
                m = ok & (grp == c)
                km(ax, t[m], e[m], str(c), cor)
            lr = multivariate_logrank_test(t[ok], grp[ok], e[ok])
            ax.set_title(f"{tit} (log-rank p={lr.p_value:.3g})")
            ax.set_xlabel("meses")
            if tit.startswith("OS por est"):
                p_estadio = float(lr.p_value)
            else:
                p_sexo = float(lr.p_value)
        a1.set_ylabel("probabilidade")
        f6 = salvar(fig, pasta, "sobrevida_por_grupo")
    res["p_logrank_estadio"], res["p_logrank_sexo"] = p_estadio, p_sexo

    # ---------------- relatório
    rel.kpi("idade mediana", fmt(res["idade_mediana"], 0) + " anos" if res["idade_mediana"] else "—")
    if "OS" in surv:
        rel.kpi("óbitos", surv["OS"]["eventos"])
        rel.kpi("seguimento mediano", fmt(surv["OS"]["seguimento_mediano_meses"], 1) + " meses")
    rel.kpi("com estádio AJCC", int(cont_g.drop("NA").sum()))
    rel.achado(f"A coorte {ctx.cancer} do PanCanAtlas tem <b>{n}</b> pacientes com dados clínicos curados (TCGA-CDR).")
    if cont_g.drop("NA").sum():
        top = cont_g.drop("NA").idxmax()
        rel.achado(f"{int(cont_g.drop('NA').sum())} de {n} pacientes ({pct(cont_g.drop('NA').sum() / n)}) têm estádio "
                   f"patológico AJCC; o grupo mais comum é o estádio {top} (n={cont_g[top]}).")
    else:
        rel.achado("Nenhum paciente tem estádio patológico AJCC; este câncer usa outro sistema de estadiamento.")
    if "OS" in surv:
        s = surv["OS"]
        med = "não foi atingida" if s["mediana_meses"] is None else f"é de {fmt(s['mediana_meses'], 1)} meses"
        rel.achado(f"Sobrevida global: {s['eventos']} óbitos em {s['n']} pacientes; a mediana {med} e o seguimento "
                   f"mediano é de {fmt(s['seguimento_mediano_meses'], 1)} meses.")
    if p_estadio is not None:
        rel.achado(f"Comparando estádios I-II com III-IV, o log-rank dá p={p_estadio:.3g} "
                   f"({'diferença significativa' if p_estadio < 0.05 else 'sem diferença significativa'} a 5%).")
    if criticos:
        rel.achado("Campos do CDR com mais de 50% de ausência: " + ", ".join(f"<code>{c}</code>" for c in criticos) + ".")
    if completos:
        rel.achado(f"{len(completos)} campos do CDR estão completos para todos os pacientes, incluindo "
                   + ", ".join(f"<code>{c}</code>" for c in completos[:6]) + ".")
    rel.nota("Para sobrevida, use as colunas OS/DSS/DFI/PFI do TCGA-CDR, que foram curadas e harmonizadas. "
             "Liu et al. recomendam PFI e OS para a maioria dos cânceres e alertam para seguimento curto em alguns tipos.")

    rel.secao("Completude dos dados clínicos")
    rel.figura(f1, "<b>Como ler:</b> cada coluna da esquerda é um paciente e cada linha um campo do TCGA-CDR. Azul = "
               "preenchido, cinza = ausente, amarelo = não se aplica (ex.: data do óbito em paciente vivo). À direita, "
               "o percentual ausente de cada campo. Campos ordenados do mais completo para o menos completo.", pasta)
    rel.tabela(tabela_html(falt_cdr.sort_values("% ausente", ascending=False).round(1), 40), "Percentual ausente por campo (TCGA-CDR)")
    if f2:
        rel.figura(f2, "<b>Como ler:</b> o mesmo para os campos atuais do GDC (Data Release mais recente). "
                   "O GDC tem mais campos que o CDR, mas muitos foram preenchidos só para alguns projetos.", pasta)
    rel.secao("Estádio patológico")
    rel.figura(f3, "<b>Como ler:</b> à esquerda, o estádio AJCC agrupado (subestádios A/B/C somados); “sem registro” "
               "inclui ausentes e cânceres que usam outro sistema. À direita, os valores exatos registrados.", pasta)
    if f3b:
        rel.figura(f3b, "<b>Como ler:</b> componentes T (tumor), N (linfonodos) e M (metástase) do AJCC registrados no GDC.", pasta)
    rel.secao("Distribuição dos metadados")
    rel.figura(f4, "<b>Como ler:</b> distribuição de cada variável clínica do CDR. Barras cinza = ausente ou agrupado em "
               "“outros”.", pasta)
    if f4b:
        rel.figura(f4b, "<b>Como ler:</b> variáveis adicionais do GDC (diagnóstico, grau, tratamento, exposições).", pasta)
    rel.secao("Sobrevida")
    rel.figura(f5, "<b>Como ler:</b> curvas de Kaplan-Meier para os quatro desfechos do CDR, em meses. A faixa é o "
               "intervalo de confiança de 95% e os tracinhos são pacientes censurados.", pasta)
    if f6:
        rel.figura(f6, "<b>Como ler:</b> sobrevida global separada por estádio (I-II vs III-IV) e por sexo, com o "
                   "p-valor do teste log-rank.", pasta)
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())
    ctx.resumo["clinico"] = res
    return res
