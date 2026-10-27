"""Site resultados_espacial: só os estudos do CELLxGENE que têm single-cell E transcriptômica espacial.

Uso (da pasta que contém pipeline_espacial e pipeline_tcga):
    python -m pipeline_espacial.pagina --saida "C:/Users/lucas/OneDrive/Desktop/Doutorado/Datasets/Espacial"

Gera <saida>/resultados_espacial/ com:
    index.html                 origem dos dados, critérios, panorama e tabela dos estudos
    colecoes/<estudo>/         uma página por estudo: doadores, mapas espaciais, composição celular,
                               concordância single-cell x espacial do mesmo doador e botões de download
    downloads/                 tabelas (zip), manifesto com o link de cada h5ad e script para baixá-los
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import html
import json
import os
import re
import shutil
import unicodedata
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from pipeline_tcga.graficos import AZUL, CORES, LARANJA, VERMELHO, salvar, tabela_html, tema
from pipeline_tcga.relatorio import Relatorio, fmt

from .executar import COLS_ESPACIAL, COLS_OBS, RX_CANCER, abrir_census, ler_obs, tcga_de

S3 = "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com/cell-census/{versao}/h5ads/{ds}.h5ad"
DISCOVER = "https://cellxgene.cziscience.com/collections/{cid}"

# Outras ômicas descritas nos artigos (além do RNA single-cell e do espacial que estão no CELLxGENE).
# Só entram as que conferi no artigo; o restante aparece como "ver artigo".
OUTRAS_OMICAS = {
    "10.1038/s41591-024-03215-z": "snRNA-seq; MERFISH, ExSeq e CODEX (proteínas) nas mesmas biópsias; dados clínicos de tratamento",
    "10.1038/s41588-021-00911-1": "CITE-seq (proteínas de superfície) em parte dos tumores; subtipos clínicos (ER, HER2, TNBC)",
    "10.1038/s41586-022-05060-x": "snATAC-seq (cromatina acessível) dos mesmos corações; Visium",
    "10.1038/s41586-023-06311-1": "Multiome (snRNA + snATAC no mesmo núcleo); TCR/BCR",
    "10.1038/s41586-023-05869-0": "Multiome (snRNA + snATAC no mesmo núcleo)",
    "10.1126/science.abo0510": "CITE-seq; TCR/BCR (VDJ)",
    "10.1038/s41588-022-01243-4": "TCR/BCR (VDJ); snRNA-seq",
}


def slug(txt: str, n: int = 50) -> str:
    t = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:n].lower()


def botao(href: str, texto: str, cheio: bool = False, baixar: bool = False) -> str:
    extra = " download" if baixar else ""
    return (f'<a class="btn{" cheio" if cheio else ""}" href="{html.escape(href)}"{extra}>'
            f"{html.escape(texto)}</a>")


CSS_EXTRA = """<style>
.btn{display:inline-block;font-weight:600;padding:7px 13px;border:1px solid var(--acc);border-radius:6px;
text-decoration:none;margin:4px 8px 4px 0;background:var(--panel);font-size:14px}
.btn.cheio{background:var(--acc);color:var(--bg)}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;background:var(--th);margin-right:4px}
.tag.c{background:#f6d5d9;color:#7a1f2b}
td a{white-space:nowrap}
</style>"""


def tamanho(s: requests.Session, url: str) -> int | None:
    try:
        r = s.head(url, timeout=60)
        return int(r.headers.get("Content-Length")) if r.ok else None
    except Exception:
        return None


def mb(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x / 1e9:.1f} GB" if x >= 1e9 else f"{x / 1e6:.0f} MB"


# ---------------------------------------------------------------- expressão: pseudo-bulk por doador

class Pseudobulk:
    """Soma das contagens brutas de uma amostra de células/spots, por gene (Ensembl)."""

    def __init__(self, exp):
        self.X = exp.ms["RNA"].X["raw"]
        var = exp.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id"]).concat().to_pandas()
        self.n = int(var["soma_joinid"].max()) + 1
        self.genes = pd.Series(var["feature_id"].values, index=var["soma_joinid"].values).reindex(range(self.n))

    def __call__(self, ids) -> pd.Series:
        tot = np.zeros(self.n)
        for tbl in self.X.read(coords=(np.sort(np.asarray(ids, dtype=np.int64)),)).tables():
            tot += np.bincount(tbl["soma_dim_1"].to_numpy(), weights=tbl["soma_data"].to_numpy(), minlength=self.n)
        s = pd.Series(tot, index=self.genes.values)
        return s[s.index.notna()].groupby(level=0).sum()


def spearman(a: pd.Series, b: pd.Series) -> float:
    g = a.index.intersection(b.index)
    a, b = a[g], b[g]
    m = (a > 0) | (b > 0)
    if m.sum() < 200:
        return np.nan
    return float(pd.Series(a[m].values).rank().corr(pd.Series(b[m].values).rank()))


# ---------------------------------------------------------------- figuras de um estudo

def mapas(esp: pd.DataFrame, slides: list[str], pasta: str, cena=None) -> str | None:
    if not slides:
        return None
    n = len(slides)
    ncol = min(3, n)
    nlin = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=(5.2 * ncol, 4.6 * nlin), squeeze=False)
    tipos = esp.loc[esp["dataset_id"].isin(slides), "cell_type"].replace({"unknown": np.nan}).value_counts().index[:9]
    paleta = {k: CORES[i % len(CORES)] for i, k in enumerate(tipos)}
    for ax, sid in zip(axs.ravel(), slides):
        d = esp[esp["dataset_id"] == sid]
        if d["assay"].iat[0].startswith("Visium") and (d["in_tissue"] == 1).any():
            d = d[d["in_tissue"] == 1]
        if d["array_row"].isna().all() and cena is not None:
            # Slide-seqV2 não tem grade: as coordenadas ficam na cena espacial (obsl/loc)
            d = d.merge(cena(sid), on="soma_joinid", how="inner")
            xs, ys = d["x"], -d["y"]
        else:
            xs, ys = d["array_col"], -d["array_row"]
        cor = d["cell_type"].map(paleta).fillna("#c9ced8")
        tam = 5 if len(d) < 6000 else 0.8
        ax.scatter(xs, ys, c=cor.values, s=tam, lw=0)
        ax.set_title(f"doador {d['donor_id'].iat[0]} · {len(d):,} spots".replace(",", "."), fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for k, c in paleta.items():
        axs.ravel()[0].scatter([], [], color=c, label=k[:40], s=20)
    axs.ravel()[0].scatter([], [], color="#c9ced8", label="sem tipo (mistura no spot)", s=20)
    fig.legend(loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02 - 0.02 * nlin))
    fig.tight_layout()
    return salvar(fig, pasta, "mapas_espaciais")


def qc_slides(esp: pd.DataFrame, pasta: str) -> str | None:
    d = esp.copy()
    vis = d["assay"].str.startswith("Visium")
    d = d[~vis | (d["in_tissue"] == 1)]
    ordem = d.groupby("dataset_id")["raw_sum"].median().sort_values().index[-30:]
    if not len(ordem):
        return None
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.25 * len(ordem) + 1.6), sharey=True)
    for ax, col, rot in [(axs[0], "raw_sum", "log10(UMIs por spot)"), (axs[1], "nnz", "log10(genes detectados por spot)")]:
        dados = [np.log10(d.loc[d["dataset_id"] == s, col].values + 1) for s in ordem]
        ax.boxplot(dados, vert=False, showfliers=False, patch_artist=True, boxprops={"facecolor": "#dbe6f0"})
        ax.set_xlabel(rot)
    rot_y = [f"{d.loc[d['dataset_id'] == s, 'donor_id'].iat[0]} · {s[:6]}" for s in ordem]
    axs[0].set_yticks(range(1, len(ordem) + 1))
    axs[0].set_yticklabels(rot_y, fontsize=7)
    axs[0].set_title("Qualidade por lâmina (até 30)")
    fig.tight_layout()
    return salvar(fig, pasta, "qualidade_laminas")


def composicao_sc(sc: pd.DataFrame, pareados: set, pasta: str) -> str | None:
    if sc.empty:
        return None
    tab = pd.crosstab(sc["donor_id"], sc["cell_type"], normalize="index")
    top = tab.mean().sort_values(ascending=False).index[:12]
    t = tab[top].copy()
    t["outros"] = 1 - t.sum(axis=1)
    t = t.loc[sorted(t.index, key=lambda x: (x not in pareados, str(x)))].iloc[:40]
    fig, ax = plt.subplots(figsize=(11, 0.24 * len(t) + 2))
    esq = np.zeros(len(t))
    cores = CORES + ["#8ab17d", "#bc6c25", "#5e548e"]
    for i, c in enumerate(t.columns):
        ax.barh(range(len(t)), t[c], left=esq, color="#d9dce3" if c == "outros" else cores[i % len(cores)],
                label=c[:38], height=0.8)
        esq += t[c].values
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([("★ " if x in pareados else "") + str(x)[:28] for x in t.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fração das células")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_title("Composição celular do single-cell por doador (★ = tem espacial)")
    return salvar(fig, pasta, "composicao_singlecell")


def composicao_par(sc: pd.DataFrame, esp: pd.DataFrame, pareados: list, pasta: str):
    """Fração de cada tipo celular no single-cell x fração dos spots com esse rótulo no espacial."""
    linhas = []
    for d in pareados:
        a = sc.loc[sc["donor_id"] == d, "cell_type"].value_counts(normalize=True)
        e = esp.loc[(esp["donor_id"] == d) & (esp["cell_type"] != "unknown"), "cell_type"]
        if len(e) < 50:
            continue
        b = e.value_counts(normalize=True)
        tipos = a.index.union(b.index)
        for t in tipos:
            linhas.append({"doador": d, "tipo celular": t, "fração sc": float(a.get(t, 0)),
                           "fração espacial": float(b.get(t, 0))})
    if not linhas:
        return None, None
    df = pd.DataFrame(linhas)
    rho = df.groupby("doador").apply(lambda g: g["fração sc"].corr(g["fração espacial"], method="spearman"),
                                     include_groups=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (d, g) in enumerate(df.groupby("doador")):
        ax.scatter(g["fração sc"] + 1e-3, g["fração espacial"] + 1e-3, s=14, alpha=0.75, color=CORES[i % len(CORES)])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([1e-3, 1], [1e-3, 1], color="#999", lw=0.8, ls="--")
    ax.set_xlabel("fração no single-cell (+0,001)")
    ax.set_ylabel("fração dos spots no espacial (+0,001)")
    ax.set_title("Composição celular: single-cell × espacial (mesmo doador)")
    return salvar(fig, pasta, "composicao_sc_vs_espacial"), (df, rho)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Site com os estudos single-cell + espacial do CELLxGENE")
    ap.add_argument("--saida", default=os.getcwd())
    ap.add_argument("--versao", default="2025-11-10")
    ap.add_argument("--celulas", type=int, default=800, help="células/spots amostrados por doador no pseudo-bulk")
    ap.add_argument("--sem-expressao", action="store_true", help="pula a concordância de expressão (mais rápido)")
    args = ap.parse_args(argv)
    tema()

    def log(m):
        print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)

    raiz = os.path.join(args.saida, "resultados_espacial")
    if os.path.isdir(raiz):
        shutil.rmtree(raiz)
    os.makedirs(os.path.join(raiz, "downloads"))
    census, versao = abrir_census(args.versao)
    log(f"Census {versao}")
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    cid_de = ds.set_index("dataset_id")["collection_id"]

    esp = ler_obs(census["census_spatial_sequencing"]["homo_sapiens"]["obs"], ["soma_joinid"] + COLS_ESPACIAL)
    for c in esp.columns:
        if isinstance(esp[c].dtype, pd.CategoricalDtype):
            esp[c] = esp[c].astype(str)
    esp["collection_id"] = esp["dataset_id"].map(cid_de)
    ds_esp = set(esp["dataset_id"])
    ds_sc = ds[ds["collection_id"].isin(set(esp["collection_id"])) & ~ds["dataset_id"].isin(ds_esp)]
    log(f"lendo obs de {len(ds_sc)} datasets single-cell")
    sc = ler_obs(census["census_data"]["homo_sapiens"]["obs"], ["soma_joinid"] + COLS_OBS,
                 f"dataset_id in {ds_sc['dataset_id'].tolist()}")
    sc["collection_id"] = sc["dataset_id"].astype(str).map(cid_de).astype("category")
    colecoes = sorted(set(esp["collection_id"]) & set(sc["collection_id"]))
    log(f"{len(colecoes)} estudos com single-cell e espacial")

    # Tamanho dos h5ad
    alvo = ds[ds["collection_id"].isin(colecoes)].copy()
    alvo["modalidade"] = np.where(alvo["dataset_id"].isin(ds_esp), "espacial", "single-cell")
    alvo["url_h5ad"] = [S3.format(versao=versao, ds=d) for d in alvo["dataset_id"]]
    s = requests.Session()
    with cf.ThreadPoolExecutor(16) as ex:
        alvo["bytes"] = list(ex.map(lambda u: tamanho(s, u), alvo["url_h5ad"]))
    log(f"tamanhos de {len(alvo)} h5ad obtidos")

    espacial_cenas = census["census_spatial_sequencing"]["homo_sapiens"]["spatial"]

    def cena(sid):
        return espacial_cenas[sid]["obsl"]["loc"].read().concat().to_pandas()[["soma_joinid", "x", "y"]]

    pb_sc = pb_sp = None
    if not args.sem_expressao:
        pb_sc = Pseudobulk(census["census_data"]["homo_sapiens"])
        pb_sp = Pseudobulk(census["census_spatial_sequencing"]["homo_sapiens"])

    rng = np.random.default_rng(0)
    resumo, linhas_idx, manifesto = [], [], []
    for cid in colecoes:
        e = esp[esp["collection_id"] == cid]
        c = sc[sc["collection_id"] == cid].copy()
        for col in c.columns:
            if isinstance(c[col].dtype, pd.CategoricalDtype):
                c[col] = c[col].astype(str)
        meta = ds[ds["collection_id"] == cid].iloc[0]
        nome, doi = meta["collection_name"], meta["collection_doi"]
        sub = slug(nome)
        pasta = os.path.join(raiz, "colecoes", sub)
        os.makedirs(pasta, exist_ok=True)
        d_esp, d_sc = set(e["donor_id"]), set(c["donor_id"])
        pareados = sorted(d_esp & d_sc)
        doencas = sorted(set(e["disease"]) | set(c["disease"]))
        cancer = any(RX_CANCER.search(x) for x in doencas)
        log(f"{nome[:60]}: {len(pareados)} doadores pareados")

        rel = Relatorio(nome, f"CELLxGENE Census {versao} · {'câncer' if cancer else 'tecido não oncológico'}")
        rel.bruto(CSS_EXTRA)
        rel.fonte(f"{meta['collection_doi_label'] or ''} (doi:{doi}); curado pelo CZ CELLxGENE Discover")
        vis = e["assay"].str.startswith("Visium")
        spots = int((~vis | (e["in_tissue"] == 1)).sum())
        rel.kpi("doadores com sc + espacial", len(pareados))
        rel.kpi("doadores só espacial", len(d_esp - d_sc))
        rel.kpi("doadores só single-cell", len(d_sc - d_esp))
        rel.kpi("lâminas espaciais", e["dataset_id"].nunique())
        rel.kpi("spots no tecido", spots)
        rel.kpi("células single-cell", len(c))
        rel.kpi("tipos celulares (sc)", c["cell_type"].nunique())
        ass_sc = c["assay"].value_counts()
        multi = "10x multiome" in ass_sc.index
        outras = OUTRAS_OMICAS.get(doi, "ver artigo")
        rel.secao("Origem e ômicas",
                  f"Artigo: <a href='https://doi.org/{html.escape(doi)}'>doi:{html.escape(doi)}</a> · "
                  f"<a href='{DISCOVER.format(cid=cid)}'>página do estudo no CELLxGENE Discover</a>. "
                  f"Doença: {html.escape('; '.join(doencas))}. Tecido: "
                  f"{html.escape('; '.join(sorted(set(e['tissue_general']))))}.")
        rel.tabela(tabela_html(pd.DataFrame([
            {"ômica": "Transcriptômica espacial", "tecnologia": "; ".join(sorted(set(e["assay"]))),
             "no CELLxGENE": "sim (contagens + coordenadas; imagem H&E no h5ad)"},
            {"ômica": "RNA single-cell / single-nucleus", "tecnologia": "; ".join(ass_sc.index),
             "no CELLxGENE": "sim (contagens brutas + tipo celular)"},
            {"ômica": "Outras (segundo o artigo)", "tecnologia": outras + (" · o RNA de núcleos multiome está aqui; o ATAC fica no depósito do artigo" if multi else ""),
             "no CELLxGENE": "não; ver o depósito indicado no artigo (GEO, ArrayExpress, HTAN etc.)"},
        ])))

        # Doadores
        def doad(df, mod):
            g = df.groupby("donor_id")
            return pd.DataFrame({f"{mod}": g.size(), f"lâminas/datasets {mod}": g["dataset_id"].nunique()})
        ed = doad(e[~vis | (e["in_tissue"] == 1)], "espacial")
        cd = doad(c, "single-cell")
        info = pd.concat([e, c])[["donor_id", "sex", "development_stage", "disease"]].drop_duplicates("donor_id").set_index("donor_id")
        dt_ = info.join(ed, how="outer").join(cd, how="outer")
        dt_["pareado"] = dt_.index.isin(pareados)
        dt_ = dt_.sort_values(["pareado", "espacial"], ascending=False)
        dt_.index.name = "doador"
        dt_ = dt_.rename(columns={"sex": "sexo", "development_stage": "idade/estágio", "disease": "doença",
                                  "espacial": "spots", "single-cell": "células"})
        dt_.to_csv(os.path.join(pasta, "doadores.csv"))
        rel.secao("Doadores", "Pareado = o mesmo doador tem single-cell e espacial no estudo. É o que permite "
                  "deconvoluir os spots com o single-cell do próprio paciente e treinar modelos conjuntos. "
                  "O CELLxGENE padroniza só sexo, idade e doença; estádio, tratamento e desfecho, quando existem, "
                  "estão em colunas próprias do h5ad ou no artigo.")
        rel.tabela(tabela_html(dt_.reset_index(), max_linhas=60))

        # Espacial
        slides = []
        for d in (pareados or sorted(d_esp)):
            g = e[e["donor_id"] == d]
            slides.append(g.groupby("dataset_id").size().idxmax())
            if len(slides) == 6:
                break
        rel.secao("Transcriptômica espacial")
        rel.figura(mapas(e, slides, pasta, cena), "Uma lâmina por doador (pareados primeiro), colorida pelo tipo celular "
                   "atribuído ao spot pelos autores. Spots cinza não têm rótulo: no Visium cada spot de 55 µm "
                   "mistura várias células.", pasta)
        rel.figura(qc_slides(e, pasta), "UMIs e genes detectados por spot em cada lâmina (só spots dentro do "
                   "tecido). Lâminas com mediana muito baixa merecem filtro antes da análise.", pasta)
        lam = e.groupby("dataset_id").agg(doador=("donor_id", "first"), tecnologia=("assay", "first"),
                                          spots=("dataset_id", "size"), umis_mediana=("raw_sum", "median"),
                                          genes_mediana=("nnz", "median"))
        lam.to_csv(os.path.join(pasta, "laminas.csv"))

        # Single-cell
        rel.secao("Single-cell")
        rel.figura(composicao_sc(c, set(pareados), pasta), "Proporção dos 12 tipos celulares mais frequentes em "
                   "cada doador (até 40 doadores).", pasta)
        top_ct = c["cell_type"].value_counts().head(20).rename("células").to_frame()
        top_ct.index.name = "tipo celular"
        rel.tabela(tabela_html(top_ct.reset_index()), "Tipos celulares mais frequentes")
        c.groupby(["donor_id", "cell_type"]).size().rename("células").to_csv(
            os.path.join(pasta, "composicao_singlecell.csv"))

        # Integração
        res_col = {"estudo": nome, "doi": doi, "cancer": cancer, "pareados": len(pareados),
                   "doadores_espacial": len(d_esp), "doadores_sc": len(d_sc), "laminas": int(e["dataset_id"].nunique()),
                   "spots": spots, "celulas_sc": len(c), "multiome": multi}
        if pareados:
            rel.secao("Single-cell × espacial no mesmo doador")
            f, out = composicao_par(c, e, pareados, pasta)
            if f:
                df, rho = out
                df.to_csv(os.path.join(pasta, "composicao_sc_vs_espacial.csv"), index=False)
                rel.figura(f, "Cada ponto é um tipo celular em um doador (cor = doador). Spearman mediano entre as "
                           f"duas frações: {np.nanmedian(rho):.2f}. Concordância alta indica que os rótulos do "
                           "espacial são coerentes com o single-cell do mesmo paciente.", pasta)
                res_col["composicao_rho"] = float(np.nanmedian(rho))
            if pb_sc is not None:
                pbs, pbe = {}, {}
                for d in sorted(d_esp | set(pareados)):
                    ide = e.loc[(e["donor_id"] == d) & (~vis | (e["in_tissue"] == 1)), "soma_joinid"].values
                    if len(ide):
                        pbe[d] = pb_sp(rng.choice(ide, min(args.celulas, len(ide)), replace=False))
                for d in pareados:
                    idc = c.loc[c["donor_id"] == d, "soma_joinid"].values
                    pbs[d] = pb_sc(rng.choice(idc, min(args.celulas, len(idc)), replace=False))
                lin = []
                for d in pareados:
                    mesmo = spearman(pbs[d], pbe[d])
                    outros = [spearman(pbs[d], pbe[o]) for o in pbe if o != d]
                    lin.append({"doador": d, "mesmo doador": mesmo,
                                "outros doadores (média)": float(np.nanmean(outros)) if outros else np.nan})
                conc = pd.DataFrame(lin)
                conc.to_csv(os.path.join(pasta, "concordancia_expressao.csv"), index=False)
                fig, ax = plt.subplots(figsize=(7, 0.3 * len(conc) + 1.6))
                y = np.arange(len(conc))
                ax.scatter(conc["mesmo doador"], y, color=VERMELHO, label="mesmo doador", zorder=3)
                ax.scatter(conc["outros doadores (média)"], y, color=AZUL, label="outros doadores (média)", zorder=3)
                ax.set_yticks(y)
                ax.set_yticklabels([str(x)[:28] for x in conc["doador"]], fontsize=7)
                ax.set_xlabel("Spearman entre pseudo-bulks (single-cell × espacial)")
                ax.legend(fontsize=8)
                rel.figura(salvar(fig, pasta, "concordancia_expressao"),
                           f"Expressão somada de {args.celulas} células e {args.celulas} spots amostrados por doador "
                           "(contagens brutas, genes Ensembl em comum). Se o vermelho fica à direita do azul, o "
                           "espacial se parece mais com o single-cell do próprio paciente do que com o de outros, "
                           "o sinal que um modelo pareado aproveita.", pasta)
                res_col["expressao_mesmo"] = float(conc["mesmo doador"].median())
                res_col["expressao_outros"] = float(conc["outros doadores (média)"].median())

        # Downloads
        a = alvo[alvo["collection_id"] == cid].copy()
        dn = (pd.concat([e, c]).groupby("dataset_id")["donor_id"]
              .agg(lambda x: "; ".join(sorted(set(x))[:4]) + (" ..." if len(set(x)) > 4 else "")))
        a["doadores"] = a["dataset_id"].map(dn)
        a = a.sort_values(["modalidade", "dataset_title"])
        a[["modalidade", "dataset_title", "dataset_id", "doadores", "dataset_total_cell_count", "bytes", "url_h5ad"]].to_csv(
            os.path.join(pasta, "manifesto_h5ad.csv"), index=False)
        manifesto.append(a.assign(estudo=nome))
        total = a["bytes"].dropna().sum()
        rel.secao("Baixar os dados",
                  f"{len(a)} arquivos h5ad (AnnData), {mb(total)} no total. Cada h5ad traz a matriz de contagens, "
                  "os metadados das células/spots e, no espacial, as coordenadas e a imagem do tecido. Abra com "
                  "scanpy (sc.read_h5ad) ou, no R, com zellkonverter/anndata.")
        rel.bruto("<p>" + botao("manifesto_h5ad.csv", "Manifesto (CSV com todos os links)", cheio=True, baixar=True)
                  + botao("../../downloads/baixar_h5ad.py", "Script para baixar tudo", baixar=True)
                  + botao("doadores.csv", "Tabela de doadores", baixar=True) + "</p>")
        corpo = "".join(
            f"<tr><td class='t'>{html.escape(r.modalidade)}</td><td class='t'>{html.escape(str(r.dataset_title))}</td>"
            f"<td class='t'>{html.escape(str(r.doadores))}</td><td>{fmt(int(r.dataset_total_cell_count), 0)}</td>"
            f"<td>{mb(r.bytes)}</td><td><a class='btn' href='{html.escape(r.url_h5ad)}'>baixar h5ad</a></td></tr>"
            for r in a.itertuples())
        rel.bruto("<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>modalidade</th>"
                  "<th class='t'>dataset</th><th class='t'>doadores</th><th>células/spots</th><th>tamanho</th>"
                  f"<th></th></tr></thead><tbody>{corpo}</tbody></table></div>")
        rel.salvar(os.path.join(pasta, "index.html"),
                   nav='<nav><a href="../../index.html">Início</a></nav>')
        resumo.append(res_col)
        linhas_idx.append({**res_col, "pagina": f"colecoes/{sub}/index.html", "outras": outras,
                           "tecnologia": "; ".join(sorted({a.split(" Spatial")[0] for a in set(e["assay"])})),
                           "doencas": "; ".join(d for d in doencas if d != "normal") or "normal (sadio)",
                           "tamanho": total, "tcga": "; ".join(sorted({t for t in map(tcga_de, doencas) if t}))})

    # ---------------------------------------------------------------- downloads gerais
    man = pd.concat(manifesto)
    man[["estudo", "modalidade", "dataset_title", "dataset_id", "doadores", "dataset_total_cell_count", "bytes",
         "url_h5ad"]].to_csv(os.path.join(raiz, "downloads", "manifesto_h5ad.csv"), index=False)
    shutil.copy(os.path.join(os.path.dirname(__file__), "baixar_h5ad.py"), os.path.join(raiz, "downloads"))
    idx = pd.DataFrame(linhas_idx).sort_values(["cancer", "pareados"], ascending=False)
    idx.drop(columns=["pagina"]).to_csv(os.path.join(raiz, "downloads", "estudos.csv"), index=False)
    with zipfile.ZipFile(os.path.join(raiz, "downloads", "tabelas_espacial.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, arqs in os.walk(raiz):
            for f in arqs:
                if f.endswith(".csv"):
                    p = os.path.join(base, f)
                    z.write(p, os.path.relpath(p, raiz))

    # ---------------------------------------------------------------- índice
    rel = Relatorio("Single-cell + transcriptômica espacial", f"CELLxGENE Census {versao} · estudos com as duas "
                    "modalidades")
    rel.bruto(CSS_EXTRA)
    rel.kpi("estudos", len(idx))
    rel.kpi("estudos de câncer", int(idx["cancer"].sum()))
    rel.kpi("doadores com sc + espacial", int(idx["pareados"].sum()))
    rel.kpi("lâminas espaciais", int(idx["laminas"].sum()))
    rel.kpi("células single-cell", int(idx["celulas_sc"].sum()))
    rel.kpi("dados (h5ad)", mb(man["bytes"].dropna().sum()))
    rel.secao("De onde vêm os dados")
    rel.texto("Todos os dados vêm do <b>CZ CELLxGENE Discover</b> (Chan Zuckerberg Initiative), que recebe os "
              "dados dos autores de cada artigo, confere e padroniza os metadados (tipo celular, tecido, doença, "
              "tecnologia, sexo e idade em ontologias como Cell Ontology, UBERON e MONDO) e distribui as matrizes "
              "em h5ad. O <b>Census</b> é a versão consolidada e versionada desse acervo, lida aqui pela API "
              "cellxgene_census. Desde 2025 ele inclui a coleção census_spatial_sequencing, com Visium e "
              "Slide-seqV2 humanos e de camundongo. A licença dos dados é a de cada estudo, em geral CC BY 4.0: "
              "cite o artigo de cada estudo ao usar os dados.")
    rel.texto("<b>Por que este banco.</b> É o único acervo aberto em que single-cell e espacial do mesmo estudo "
              "estão no mesmo formato, com os mesmos rótulos de tipo celular e o mesmo identificador de doador. O "
              "HTAN é mais rico em câncer, mas a parte aberta e harmonizada (como a coleção HTAPP de mama) já "
              "está aqui, e o restante exige cadastro e fica espalhado em vários portais.")
    rel.texto("<b>Critério de inclusão.</b> Entram os estudos que têm, no CELLxGENE, pelo menos um dataset "
              "espacial e um dataset single-cell humano. Foram excluídos 4 estudos só espaciais (colorretal, "
              "mama triplo-negativo, TCR de melanoma/rim e Slide-seqV2 de rim). Os estudos estão ordenados por "
              "câncer e depois pelo número de doadores pareados.")
    rel.secao("Baixar")
    rel.bruto("<p>" + botao("downloads/manifesto_h5ad.csv", "Manifesto de todos os h5ad (CSV)", cheio=True, baixar=True)
              + botao("downloads/baixar_h5ad.py", "Script para baixar (Python)", baixar=True)
              + botao("downloads/tabelas_espacial.zip", "Todas as tabelas desta análise (zip)", baixar=True)
              + botao("downloads/estudos.csv", "Resumo dos estudos (CSV)", baixar=True) + "</p>")
    rel.texto("O script baixa os h5ad do manifesto, com filtro opcional por estudo ou modalidade. Exemplo: "
              "<code>python baixar_h5ad.py --estudo HTAN --destino D:\\espacial</code>.")

    # Figura geral
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(idx) + 1.4))
    y = np.arange(len(idx))[::-1]
    so_e = idx["doadores_espacial"] - idx["pareados"]
    so_s = idx["doadores_sc"] - idx["pareados"]
    ax.barh(y, idx["pareados"], color=VERMELHO, label="sc + espacial")
    ax.barh(y, so_e, left=idx["pareados"], color=LARANJA, label="só espacial")
    ax.barh(y, so_s, left=idx["pareados"] + so_e, color="#c9d3e0", label="só single-cell")
    ax.set_yticks(y)
    ax.set_yticklabels([("● " if c else "") + n[:60] for n, c in zip(idx["estudo"], idx["cancer"])], fontsize=8)
    ax.set_xlabel("doadores")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Doadores por estudo (● = câncer)")
    rel.secao("Panorama")
    rel.figura(salvar(fig, raiz, "panorama_doadores"), "Vermelho: doadores com as duas modalidades.", raiz)

    corpo = ""
    for r in idx.itertuples():
        tag = "<span class='tag c'>câncer</span>" if r.cancer else ""
        extra = ""
        if isinstance(getattr(r, "expressao_mesmo", None), float) and not np.isnan(r.expressao_mesmo):
            extra = f"{r.expressao_mesmo:.2f} vs {r.expressao_outros:.2f}"
        corpo += (f"<tr><td class='t'><a href='{r.pagina}'>{html.escape(r.estudo)}</a> {tag}</td>"
                  f"<td class='t'>{html.escape(r.doencas)}</td><td class='t'>{html.escape(r.tecnologia)}</td>"
                  f"<td class='t'>{html.escape(r.outras)}</td><td>{r.pareados}</td><td>{r.laminas}</td>"
                  f"<td>{fmt(r.celulas_sc, 0)}</td><td>{extra or '—'}</td><td>{mb(r.tamanho)}</td></tr>")
    rel.secao("Estudos")
    rel.bruto("<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>estudo</th><th class='t'>doença</th>"
              "<th class='t'>espacial</th><th class='t'>outras ômicas (artigo)</th><th>doadores pareados</th>"
              "<th>lâminas</th><th>células sc</th><th>expressão: mesmo × outros doadores</th><th>dados</th>"
              f"</tr></thead><tbody>{corpo}</tbody></table></div>")
    rel.texto("A coluna de expressão é o Spearman mediano entre o pseudo-bulk do single-cell e o do espacial do "
              "mesmo doador, comparado ao de outros doadores do estudo.")
    rel.salvar(os.path.join(raiz, "index.html"), nav="")
    with open(os.path.join(raiz, "downloads", "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1, default=str)
    log("fim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
