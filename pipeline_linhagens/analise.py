"""Análises descritivas de um banco de linhagens: disponibilidade, anotações e uma página por ômica."""
from __future__ import annotations

import html
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_tcga.analises.disponibilidade import desenhar_upset
from pipeline_tcga.graficos import AZUL, CORES, LARANJA, VERMELHO, barras_categoria, pca, salvar, tabela_html
from pipeline_tcga.relatorio import Relatorio, fmt, pct

UNIDADE = "linhagens"


def nav(nivel: int) -> str:
    sobe = "../" * nivel
    return f'<nav><a href="{sobe}index.html">Início</a><a href="{sobe}COMPARACAO/index.html">DepMap × Passports</a></nav>'


def cores_linhagem(lin: pd.Series, max_cat: int = 10) -> tuple[pd.Series, dict]:
    top = lin.fillna("(sem anotação)").value_counts().index[:max_cat]
    rot = lin.fillna("(sem anotação)").where(lambda x: x.isin(top), "(outras)")
    paleta = {k: CORES[i % len(CORES)] for i, k in enumerate(top)}
    paleta["(outras)"] = "#d0d4dc"
    return rot, paleta


# ---------------------------------------------------------------- ômica em formato de matriz

def analisar_matriz(X: pd.DataFrame, anot: pd.DataFrame, chave: str, rotulo: str, pasta: str, rel: Relatorio) -> dict:
    res = {"modelos": int(X.shape[0]), "variaveis": int(X.shape[1]),
           "faltantes": float(np.isnan(X.to_numpy()).mean())}
    rel.kpi("linhagens com dado", X.shape[0])
    rel.kpi("variáveis (genes, proteínas...)", X.shape[1])
    rel.kpi("valores faltantes", pct(res["faltantes"]))

    vals = X.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals):
        rng = np.random.default_rng(0)
        amostra = rng.choice(vals, size=min(300_000, len(vals)), replace=False)
        q = np.quantile(vals if len(vals) < 5e6 else amostra, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
        res["quantis"] = dict(zip(["min", "p1", "p25", "mediana", "p75", "p99", "max"], map(float, q)))
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(amostra, bins=120, color=AZUL)
        ax.set_xlabel("valor")
        ax.set_ylabel("frequência (amostra de valores)")
        ax.set_title(f"{rotulo}: distribuição dos valores")
        rel.secao("Distribuição dos valores")
        rel.figura(salvar(fig, pasta, "distribuicao"), "Histograma de uma amostra aleatória de até 300 mil valores "
                   "da matriz. Ajuda a ver a escala (log, beta, z-score) e se há valores extremos.", pasta)
        rel.tabela(tabela_html(pd.DataFrame([res["quantis"]])), "Quantis")

    # Faltantes por linhagem e por variável
    falt_lin = X.isna().mean(axis=1)
    falt_var = X.isna().mean(axis=0)
    if falt_var.max() > 0:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].hist(falt_lin, bins=40, color=AZUL)
        axs[0].set_xlabel("fração faltante por linhagem")
        axs[1].hist(falt_var, bins=40, color=LARANJA)
        axs[1].set_xlabel("fração faltante por variável")
        rel.secao("Valores faltantes")
        rel.figura(salvar(fig, pasta, "faltantes"), "Distribuição da fração de valores ausentes por linhagem e por "
                   "variável. Proteoma e fármacos costumam ter muitos ausentes, o que pesa na escolha da imputação.",
                   pasta)

    # Variáveis mais variáveis
    var = X.var().sort_values(ascending=False)
    top = pd.DataFrame({"variância": var.iloc[:25], "média": X[var.index[:25]].mean(),
                        "% faltante": 100 * falt_var[var.index[:25]]})
    top.index.name = "variável"
    top.to_csv(os.path.join(pasta, "variaveis_mais_variaveis.csv"))
    rel.secao("Variáveis com maior variância")
    rel.tabela(tabela_html(top.reset_index()))

    # PCA colorida pela linhagem
    lin = anot.reindex(X.index)["linhagem"]
    usar = var.index[:2000]
    Xp = X[usar]
    Xp = Xp.loc[:, Xp.notna().mean() >= 0.8]
    Xp = Xp.loc[Xp.notna().mean(axis=1) >= 0.5]
    if Xp.shape[0] >= 10 and Xp.shape[1] >= 5:
        sc, ev, _ = pca(Xp.fillna(Xp.mean()), n=10)
        rot, paleta = cores_linhagem(lin.reindex(sc.index))
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for k in list(paleta)[::-1]:
            m = rot == k
            ax.scatter(sc.loc[m, "PC1"], sc.loc[m, "PC2"], s=10, color=paleta[k], label=f"{k} ({m.sum()})",
                       alpha=0.8, lw=0)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
        ax.set_title(f"{rotulo}: PCA das {Xp.shape[1]} variáveis mais variáveis")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        rel.secao("Estrutura global (PCA)")
        rel.figura(salvar(fig, pasta, "pca_linhagem"), "Cada ponto é uma linhagem, colorida pelo tecido de origem "
                   "(10 mais frequentes). Agrupamento por tecido indica que a ômica carrega sinal de linhagem, "
                   "que precisa ser modelado (por exemplo como efeito aleatório) numa integração.", pasta)
        res["pca_var_pc1_pc2"] = [float(ev[0]), float(ev[1])]
        # Quanto da variância de cada PC é explicada pela linhagem (R² de uma ANOVA)
        r2 = {}
        for pc in sc.columns[:5]:
            g = pd.DataFrame({"y": sc[pc], "g": lin.reindex(sc.index).fillna("NA")})
            tot = ((g.y - g.y.mean()) ** 2).sum()
            entre = g.groupby("g").y.transform("mean")
            r2[pc] = float(((entre - g.y.mean()) ** 2).sum() / tot) if tot > 0 else np.nan
        res["r2_linhagem_pcs"] = r2
        rel.texto("Fração da variância de cada componente explicada pelo tecido de origem (R²): "
                  + ", ".join(f"{k} {pct(v)}" for k, v in r2.items()) + ".")

    if chave in ("crispr", "rnai"):
        res.update(essencialidade(X, anot, rotulo, pasta, rel))
    return res


def essencialidade(X: pd.DataFrame, anot: pd.DataFrame, rotulo: str, pasta: str, rel: Relatorio) -> dict:
    """Genes essenciais comuns e seletivos a partir do escore de efeito gênico (< -0,5 = dependência)."""
    dep = (X < -0.5).where(X.notna())
    frac = dep.mean()
    comuns = frac[frac >= 0.9].index
    seletivos = frac[(frac >= 0.02) & (frac <= 0.5)].index
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(frac.dropna(), bins=50, color=AZUL)
    ax.set_yscale("log")
    ax.set_xlabel("fração das linhagens dependentes do gene (efeito < -0,5)")
    ax.set_ylabel("genes")
    rel.secao("Dependências gênicas")
    rel.figura(salvar(fig, pasta, "dependencias"), f"{len(comuns)} genes são essenciais em ≥90% das linhagens "
               f"(essenciais comuns) e {len(seletivos)} em 2% a 50% (dependências seletivas, as mais "
               "interessantes para associar a outras ômicas).", pasta)
    pd.Series(sorted(comuns), name="gene").to_csv(os.path.join(pasta, "essenciais_comuns.csv"), index=False)
    sel = frac[seletivos].sort_values(ascending=False).rename("fração dependente").to_frame()
    sel.to_csv(os.path.join(pasta, "dependencias_seletivas.csv"))
    rel.tabela(tabela_html(sel.reset_index().rename(columns={"index": "gene"}).head(25)),
               "Dependências seletivas mais frequentes")
    return {"essenciais_comuns": int(len(comuns)), "dependencias_seletivas": int(len(seletivos))}


# ---------------------------------------------------------------- ômica em formato longo (mutação, fusão)

def analisar_longo(df: pd.DataFrame, anot: pd.DataFrame, rotulo: str, pasta: str, rel: Relatorio) -> dict:
    por_modelo = df.groupby("modelo").size()
    res = {"modelos": int(por_modelo.size), "eventos": int(len(df)),
           "eventos_por_modelo_mediana": float(por_modelo.median())}
    rel.kpi("linhagens com dado", por_modelo.size)
    rel.kpi("eventos (linhas)", len(df))
    rel.kpi("mediana por linhagem", float(por_modelo.median()))

    lin = anot.reindex(por_modelo.index)["linhagem"].fillna("(sem anotação)")
    ordem = lin.value_counts().index[:15]
    grupos = [np.log10(por_modelo[lin == k].values + 1) for k in ordem]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(grupos, showfliers=False, patch_artist=True, boxprops={"facecolor": "#dbe6f0"})
    ax.set_xticks(range(1, len(ordem) + 1))
    ax.set_xticklabels([str(k)[:22] for k in ordem], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("log10(eventos + 1)")
    ax.set_title(f"{rotulo}: eventos por linhagem, por tecido")
    rel.secao("Eventos por linhagem")
    rel.figura(salvar(fig, pasta, "eventos_por_linhagem"), "Número de eventos (mutações, fusões...) por linhagem, "
               "nos 15 tecidos mais frequentes. Para mutação, é uma medida aproximada da carga mutacional.", pasta)
    por_modelo.rename("eventos").to_csv(os.path.join(pasta, "eventos_por_linhagem.csv"))

    if "gene" in df.columns:
        genes = df.drop_duplicates(["modelo", "gene"]).groupby("gene").size().sort_values(ascending=False)
        top = (genes.iloc[:30] / por_modelo.size * 100).round(1).rename("% das linhagens").to_frame()
        top.insert(0, "linhagens", genes.iloc[:30])
        top.to_csv(os.path.join(pasta, "genes_mais_frequentes.csv"))
        rel.secao("Genes mais frequentes")
        rel.tabela(tabela_html(top.reset_index()))
        res["top_genes"] = genes.iloc[:10].to_dict()
    return res


# ---------------------------------------------------------------- panorama do banco

def anotacoes(anot: pd.DataFrame, bruto: pd.DataFrame, pasta: str, rel: Relatorio, colunas_chave: list[str]) -> dict:
    """Completude das colunas de anotação e distribuição das principais."""
    comp = bruto.replace({"": np.nan, "unknown": np.nan, "Unknown": np.nan}).notna().mean().sort_values()
    tab = (100 * comp).round(1).rename("% preenchido").to_frame()
    tab.index.name = "coluna"
    tab.to_csv(os.path.join(pasta, "anotacoes_completude.csv"))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.18 * len(tab))))
    ax.barh(tab.index, tab["% preenchido"], color=[VERMELHO if v < 50 else AZUL for v in tab["% preenchido"]])
    ax.set_xlabel("% das linhagens com a informação")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Completude das anotações")
    rel.secao("Anotações das linhagens (equivalente ao clínico)",
              "Em linhagens não há sobrevida nem estádio na maior parte dos casos. O que existe é a origem "
              "(tecido, doença, subtipo), dados do doador (sexo, idade, etnia), se veio de tumor primário ou "
              "metástase, e propriedades do cultivo.")
    rel.figura(salvar(fig, pasta, "anotacoes_completude"), "Fração de linhagens com cada coluna preenchida "
               "(vermelho = menos da metade).", pasta)
    presentes = [c for c in colunas_chave if c in bruto.columns]
    if presentes:
        n = len(presentes)
        ncol = 2
        nlin = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nlin, ncol, figsize=(11, 3.2 * nlin))
        axs = np.atleast_1d(axs).ravel()
        for ax, c in zip(axs, presentes):
            barras_categoria(ax, bruto[c].replace("", np.nan), c, max_cat=10)
            ax.set_xlabel(UNIDADE)
        for ax in axs[n:]:
            ax.axis("off")
        fig.tight_layout()
        rel.figura(salvar(fig, pasta, "anotacoes_distribuicoes"), "Distribuição das principais anotações.", pasta)
    return {"completude": tab["% preenchido"].to_dict()}


def disponibilidade(tab: pd.DataFrame, rotulos: dict, anot: pd.DataFrame, banco: str, pasta: str,
                    rel: Relatorio) -> dict:
    """tab: linhagens x ômicas (bool)."""
    t = tab.rename(columns=rotulos)
    t.to_csv(os.path.join(pasta, "disponibilidade_por_linhagem.csv"))
    fig, combos = desenhar_upset(t, f"{banco}: linhagens por combinação de ômicas", unidade=UNIDADE)
    rel.secao("Disponibilidade e sobreposição das ômicas")
    rel.figura(salvar(fig, pasta, "upset_combinacoes"), "UpSet: cada barra conta as linhagens com exatamente "
               "aquela combinação de ômicas (interseção exclusiva). À esquerda, o total por ômica.", pasta)
    comb = t.apply(lambda r: " + ".join(c for c in t.columns if r[c]) or "nenhuma", axis=1).value_counts()
    comb = comb.rename_axis("combinação").reset_index(name=UNIDADE)
    comb.to_csv(os.path.join(pasta, "combinacoes.csv"), index=False)

    ordem = [c for c in t.sum().sort_values(ascending=False).index if t[c].any()]
    acum, cur = [], pd.Series(True, index=t.index)
    for c in ordem:
        cur = cur & t[c]
        acum.append({"última adicionada": c, UNIDADE: int(cur.sum())})
    funil = pd.DataFrame(acum)
    funil.to_csv(os.path.join(pasta, "funil_intersecao.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(funil) + 1.2))
    y = np.arange(len(funil))[::-1]
    ax.barh(y, funil[UNIDADE], color=AZUL, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(["+ " + x for x in funil["última adicionada"]])
    for i, v in zip(y, funil[UNIDADE]):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("linhagens com todas as ômicas até esta linha")
    ax.set_title("Quantas linhagens sobram ao exigir cada ômica a mais")
    ax.grid(axis="y", visible=False)
    rel.figura(salvar(fig, pasta, "funil_intersecao"), "Interseção acumulada, da ômica mais comum para a menos "
               "comum.", pasta)
    rel.tabela(tabela_html(funil), "Funil de interseção")

    # Linhagens por tecido x ômica
    lin = anot.reindex(t.index)["linhagem"].fillna("(sem anotação)")
    porlin = t.groupby(lin).sum().astype(int)
    porlin.insert(0, "total", lin.value_counts())
    porlin = porlin.sort_values("total", ascending=False)
    porlin.index.name = "tecido"
    porlin.to_csv(os.path.join(pasta, "linhagens_por_tecido_e_omica.csv"))
    h = porlin.drop(columns="total").head(30)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * h.shape[1], 0.3 * len(h) + 1.6))
    ax.imshow(np.log1p(h.to_numpy()), aspect="auto", cmap="Blues")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.iat[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > h.to_numpy().max() * 0.35 else "#1d2433")
    ax.set_xticks(range(h.shape[1]))
    ax.set_xticklabels(h.columns, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(h.shape[0]))
    ax.set_yticklabels([str(k)[:30] for k in h.index], fontsize=8)
    ax.grid(False)
    ax.set_title("Linhagens por tecido e ômica (30 tecidos mais frequentes)")
    rel.secao("Linhagens por tipo de câncer")
    rel.figura(salvar(fig, pasta, "linhagens_por_tecido"), "Número de linhagens de cada tecido com cada ômica.",
               pasta)
    rel.tabela(tabela_html(porlin.reset_index(), max_linhas=40), "Linhagens por tecido e ômica")

    res = {"linhagens": int(len(t)), "por_omica": {k: int(v) for k, v in t.sum().items()},
           "todas": int(t[ordem].all(axis=1).sum()) if ordem else 0,
           "funil": funil.to_dict(orient="records"),
           "combinacoes_top": comb.head(15).to_dict(orient="records")}

    if "tcga" in anot.columns and anot["tcga"].notna().any():
        tc = anot.reindex(t.index)["tcga"]
        portc = t.groupby(tc).sum().astype(int)
        portc.insert(0, "total", tc.value_counts())
        portc = portc.sort_values("total", ascending=False)
        portc.index.name = "projeto TCGA"
        portc.to_csv(os.path.join(pasta, "linhagens_por_projeto_tcga.csv"))
        rel.secao("Correspondência com os projetos do TCGA",
                  "Mapeamento aproximado do subtipo Oncotree de cada linhagem para a sigla do TCGA, para "
                  "cruzar com o levantamento do TCGA (por exemplo, quantas linhagens de BRCA têm CRISPR e "
                  "RNA-seq para servir de validação).")
        rel.tabela(tabela_html(portc.reset_index(), max_linhas=40))
        res["por_tcga"] = portc["total"].to_dict()
    return res


def pagina_omica(rel: Relatorio, caminho: str) -> str:
    return rel.salvar(caminho, nav=nav(2))


def tabela_links(linhas: list[dict]) -> str:
    """Tabela HTML com link para a página de cada ômica."""
    cab = ["ômica", "arquivo", "versão", "formato", "linhagens", "variáveis", "página"]
    corpo = []
    for r in linhas:
        link = f'<a href="{html.escape(r["pagina"])}">abrir</a>' if r.get("pagina") else "—"
        corpo.append("<tr>" + "".join(
            f"<td class='t'>{html.escape(str(r.get(k) if r.get(k) is not None else '—'))}</td>"
            for k in ["rotulo", "arquivo", "versao", "formato"])
            + f"<td>{fmt(r.get('linhagens') or 0, 0)}</td><td>{fmt(r.get('variaveis') or 0, 0)}</td><td>{link}</td></tr>")
    return ("<table class='tab'><thead><tr>" + "".join(f"<th class='t'>{c}</th>" for c in cab)
            + "</tr></thead><tbody>" + "".join(corpo) + "</tbody></table>")
