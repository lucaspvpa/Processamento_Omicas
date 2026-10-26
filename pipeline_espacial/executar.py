"""Levantamento do CELLxGENE Census: transcriptômica espacial (Visium, Slide-seqV2) e single-cell do mesmo doador.

Exemplos (rodar da pasta que contém pipeline_espacial e pipeline_tcga):
    python -m pipeline_espacial --saida "C:/Users/lucas/OneDrive/Desktop/Doutorado/Datasets/Espacial"
    python -m pipeline_espacial --versao 2025-11-10 --sem-cancer-sc

Só lê metadados (obs) do Census, sem baixar matrizes de expressão: cabe em poucos GB de memória e leva de 10 a
30 minutos conforme a internet. Gera <saida>/resultados_espacial/index.html com figuras e tabelas em CSV.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_tcga.graficos import AZUL, CORES, LARANJA, VERMELHO, barras_categoria, salvar, tabela_html, tema
from pipeline_tcga.relatorio import Relatorio, fmt, pct

COLS_OBS = ["dataset_id", "assay", "disease", "tissue_general", "tissue", "donor_id", "is_primary_data",
            "cell_type", "sex", "development_stage", "self_reported_ethnicity"]
COLS_ESPACIAL = COLS_OBS + ["in_tissue", "array_row", "array_col", "raw_sum", "nnz", "n_measured_vars"]
RX_CANCER = re.compile(r"carcinoma|cancer|tumou?r|oma\b|leuka?emia|lymphoma|neoplas|melanoma|myeloma|sarcoma|"
                       r"glioma|blastoma|adenoma", re.I)
# Doença (texto do Census) -> projeto TCGA, aproximado, para cruzar com o levantamento do TCGA
DOENCA_TCGA = [
    (r"breast", "BRCA"), (r"lung adenocarcinoma", "LUAD"), (r"(lung|pulmonary).*squamous|squamous.*lung", "LUSC"),
    (r"colorectal|colon|rectal", "COAD/READ"), (r"clear cell renal|renal cell|nonpapillary renal", "KIRC"),
    (r"papillary renal", "KIRP"), (r"uveal melanoma", "UVM"), (r"melanoma", "SKCM"),
    (r"glioblastoma", "GBM"), (r"astrocytoma|oligodendroglioma|glioma", "LGG"), (r"hepatocellular|liver cancer", "LIHC"),
    (r"cholangio", "CHOL"), (r"pancrea", "PAAD"), (r"gastric|stomach", "STAD"), (r"esophag", "ESCA"),
    (r"head and neck|oral|oropharynx|laryn|nasopharyn", "HNSC"), (r"ovar", "OV"), (r"endometri|uterine", "UCEC"),
    (r"cervi", "CESC"), (r"prostat", "PRAD"), (r"thyroid", "THCA"), (r"bladder|urothelial", "BLCA"),
    (r"acute myeloid", "LAML"), (r"diffuse large B", "DLBC"), (r"mesothelioma", "MESO"), (r"sarcoma", "SARC"),
    (r"testicular|germ cell", "TGCT"), (r"thymoma", "THYM"), (r"adrenocortical", "ACC"),
    (r"pheochromocytoma|paraganglioma", "PCPG"),
]


def tcga_de(doenca: str) -> str | None:
    for rx, sigla in DOENCA_TCGA:
        if re.search(rx, str(doenca), re.I):
            return sigla
    return None


def abrir_census(versao: str):
    """Abre o Census pela API oficial; se o servidor de espelhos estiver bloqueado, abre o S3 direto."""
    import cellxgene_census
    try:
        return cellxgene_census.open_soma(census_version=versao), versao
    except Exception:
        import tiledbsoma as soma
        import requests
        rel = requests.get("https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com/cell-census/"
                           "release.json", timeout=60).json()
        v = rel.get(versao, versao) if isinstance(rel.get(versao), str) else versao
        uri = rel[v]["soma"]["uri"]
        cfg = {"vfs.s3.region": "us-west-2", "vfs.s3.no_sign_request": "true"}
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            m = re.match(r"https?://([^:/]+):(\d+)", proxy)
            if m:
                cfg.update({"vfs.s3.proxy_host": m.group(1), "vfs.s3.proxy_port": m.group(2),
                            "vfs.s3.proxy_scheme": "http"})
        ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
        if ca:
            cfg["vfs.s3.ca_file"] = ca
        return soma.open(uri, context=soma.SOMATileDBContext(tiledb_config=cfg)), v


def ler_obs(df_soma, colunas, filtro=None) -> pd.DataFrame:
    """Lê o obs em blocos e mantém o texto como categoria (milhões de células cabem em poucos GB)."""
    from pandas.api.types import union_categoricals
    disp = set(df_soma.schema.names)
    cols = [c for c in colunas if c in disp]
    blocos = []
    for tbl in df_soma.read(column_names=cols, value_filter=filtro):
        b = tbl.to_pandas()
        for c in b.columns:
            if b[c].dtype == object:
                b[c] = b[c].astype("category")
        blocos.append(b)
    if not blocos:
        return pd.DataFrame(columns=cols)
    saida = {}
    for c in cols:
        partes = [b[c] for b in blocos]
        if isinstance(partes[0].dtype, pd.CategoricalDtype):
            saida[c] = pd.Series(union_categoricals(partes, ignore_order=True))
        else:
            saida[c] = pd.concat(partes, ignore_index=True)
    return pd.DataFrame(saida)


def por_doador(obs: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (coleção, doador): metadados do doador, mais frequentes."""
    def moda(s):
        v = s.replace({"unknown": np.nan, "na": np.nan}).dropna()
        return v.mode().iat[0] if len(v) else "unknown"
    return obs.groupby(["collection_name", "donor_id"], observed=True).agg(
        sexo=("sex", moda), estagio=("development_stage", moda), etnia=("self_reported_ethnicity", moda),
        doenca=("disease", moda)).reset_index()


def mapa_exemplo(esp: pd.DataFrame, pasta: str) -> tuple[str | None, str]:
    """Mapa de uma lâmina Visium de câncer: posição dos spots, colorida pela contagem total de UMIs."""
    vis = esp[esp["assay"].str.contains("Visium") & esp["disease"].str.contains(RX_CANCER)]
    if vis.empty or "array_row" not in vis:
        return None, ""
    dsid = vis.groupby("dataset_id", observed=True).size().sort_values(ascending=False).index[0]
    d = vis[vis["dataset_id"] == dsid]
    d = d[d["in_tissue"] == 1] if (d["in_tissue"] == 1).any() else d
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    sc = axs[0].scatter(d["array_col"], -d["array_row"], c=np.log10(d["raw_sum"] + 1), s=6, cmap="viridis", lw=0)
    fig.colorbar(sc, ax=axs[0], label="log10(UMIs por spot)")
    axs[0].set_title("UMIs por spot")
    ct = d["cell_type"].replace({"unknown": "(sem anotação)"})
    top = ct.value_counts().index[:8]
    for i, k in enumerate(top):
        m = ct == k
        axs[1].scatter(d.loc[m, "array_col"], -d.loc[m, "array_row"], s=6, lw=0, color=CORES[i % len(CORES)],
                       label=f"{k} ({m.sum()})")
    axs[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    axs[1].set_title("Anotação do spot")
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")
    titulo = f"{d['disease'].iat[0]} · {d['tissue'].iat[0]} · {len(d)} spots"
    fig.suptitle(titulo, fontsize=10)
    return salvar(fig, pasta, "mapa_espacial_exemplo"), titulo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Levantamento single-cell + espacial no CELLxGENE Census")
    ap.add_argument("--saida", default=os.getcwd())
    ap.add_argument("--versao", default="2025-11-10", help="versão do Census (precisa ter census_spatial_sequencing)")
    ap.add_argument("--sem-cancer-sc", action="store_true", help="pula o levantamento do single-cell de câncer "
                    "no Census inteiro (a parte mais demorada)")
    args = ap.parse_args(argv)
    tema()
    pasta = os.path.join(args.saida, "resultados_espacial")
    os.makedirs(pasta, exist_ok=True)

    def log(m):
        print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)

    census, versao = abrir_census(args.versao)
    log(f"Census {versao} aberto")
    if "census_spatial_sequencing" not in census.keys():
        raise SystemExit("Esta versão do Census não tem dados espaciais; use --versao 2025-01-30 ou mais nova.")
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    resumo_c = census["census_info"]["summary"].read().concat().to_pandas().set_index("label")["value"]

    log("lendo obs espacial (humano)")
    esp = ler_obs(census["census_spatial_sequencing"]["homo_sapiens"]["obs"], COLS_ESPACIAL)
    for c in esp.columns:
        if isinstance(esp[c].dtype, pd.CategoricalDtype):
            esp[c] = esp[c].astype(str)
    esp = esp.merge(ds[["dataset_id", "collection_id", "collection_name", "collection_doi", "dataset_title"]],
                    on="dataset_id", how="left")
    ds_esp = set(esp["dataset_id"])
    colecoes = set(esp["collection_id"])

    log("lendo obs single-cell das coleções que têm espacial")
    ds_sc = ds[ds["collection_id"].isin(colecoes) & ~ds["dataset_id"].isin(ds_esp)]
    sc = ler_obs(census["census_data"]["homo_sapiens"]["obs"], COLS_OBS,
                 f"dataset_id in {ds_sc['dataset_id'].tolist()}") if len(ds_sc) else pd.DataFrame(columns=COLS_OBS)
    sc["collection_id"] = sc["dataset_id"].map(ds.set_index("dataset_id")["collection_id"]).astype("category")
    sc["collection_name"] = sc["dataset_id"].map(ds.set_index("dataset_id")["collection_name"]).astype("category")

    rel = Relatorio("CELLxGENE Census: single-cell e transcriptômica espacial",
                    f"Census {versao} · schema {resumo_c.get('census_schema_version', '?')} · só metadados")
    rel.fonte(f"CELLxGENE Census {versao} (CZ CELLxGENE Discover), coleção census_spatial_sequencing e census_data")
    rel.kpi("células no Census (únicas)", int(resumo_c.get("unique_cell_count", 0)))
    rel.kpi("datasets no Census", len(ds))
    rel.kpi("datasets espaciais (humano)", len(ds_esp))
    rel.kpi("spots/beads espaciais", len(esp))
    rel.kpi("coleções com espacial", len(colecoes))
    rel.kpi("doadores com espacial", int(esp.groupby(["collection_id", "donor_id"], observed=True).ngroups))
    rel.nota("Por que o CELLxGENE Census: é o único banco aberto que coloca single-cell e transcriptômica espacial "
             "no mesmo esquema, com ontologias harmonizadas (tipo celular, tecido, doença, tecnologia), acesso "
             "programático sem cadastro e o mesmo identificador de doador dentro de cada estudo. O HTAN é mais "
             "rico em câncer (vários órgãos, imagem multiplexada, clínico), mas os níveis 1 e 2 são controlados e "
             "os dados abertos ficam espalhados entre Synapse, CDS e BigQuery; a parte do HTAN que é aberta e "
             "harmonizada (por exemplo HTAPP de mama) já está dentro do Census e aparece aqui.")

    # ---------------------------------------------------------------- espacial
    rel.secao("Transcriptômica espacial: o que existe")
    tab_ds = esp.groupby(["collection_name", "dataset_id", "dataset_title"], observed=True).agg(
        tecnologia=("assay", "first"), doenca=("disease", lambda s: "; ".join(sorted(set(s)))),
        tecido=("tissue_general", "first"), doadores=("donor_id", "nunique"), spots=("dataset_id", "size"),
        spots_no_tecido=("in_tissue", lambda s: int((s == 1).sum())), genes_medidos=("n_measured_vars", "median"),
        umis_mediana=("raw_sum", "median")).reset_index()
    tab_ds["h5ad"] = "https://datasets.cellxgene.cziscience.com/" + tab_ds["dataset_id"] + ".h5ad"
    tab_ds.to_csv(os.path.join(pasta, "datasets_espaciais.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    barras_categoria(axs[0], tab_ds["tecnologia"], "Datasets por tecnologia")
    barras_categoria(axs[1], tab_ds["doenca"].str.split("; ").explode(), "Datasets por doença", max_cat=14)
    barras_categoria(axs[2], tab_ds["tecido"], "Datasets por tecido", max_cat=14)
    for ax in axs:
        ax.set_xlabel("datasets")
    fig.tight_layout()
    rel.figura(salvar(fig, pasta, "espacial_panorama"), "Cada dataset espacial é, em geral, uma lâmina (Visium) ou "
               "um puck (Slide-seqV2). Um doador pode ter várias lâminas.", pasta)

    por_doenca = esp.groupby("disease", observed=True).agg(datasets=("dataset_id", "nunique"), coleções=("collection_id", "nunique"),
                                            doadores=("donor_id", "nunique"), spots=("dataset_id", "size"))
    por_doenca["câncer"] = por_doenca.index.str.contains(RX_CANCER)
    por_doenca["TCGA (aprox.)"] = [tcga_de(d) if c else None for d, c in zip(por_doenca.index, por_doenca["câncer"])]
    por_doenca = por_doenca.sort_values("spots", ascending=False)
    por_doenca.index.name = "doença"
    por_doenca.to_csv(os.path.join(pasta, "espacial_por_doenca.csv"))
    rel.tabela(tabela_html(por_doenca.reset_index()), "Espacial por doença")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (a, g) in enumerate(tab_ds.groupby("tecnologia", observed=True)):
        ax.hist(np.log10(g["spots"]), bins=30, alpha=0.7, color=CORES[i], label=f"{a} ({len(g)})")
    ax.set_xlabel("log10(spots ou beads por dataset)")
    ax.set_ylabel("datasets")
    ax.legend()
    rel.figura(salvar(fig, pasta, "espacial_spots_por_dataset"), "Tamanho das lâminas. Visium tem até ~5 mil "
               "spots de 55 µm (vários núcleos por spot); Slide-seqV2 tem dezenas de milhares de beads de 10 µm.",
               pasta)

    f, titulo = mapa_exemplo(esp, pasta)
    if f:
        rel.secao("Exemplo de lâmina de câncer")
        rel.figura(f, f"{titulo}. Posições dos spots na grade Visium (só metadados; as coordenadas em pixel e a "
                   "imagem H&E ficam no h5ad do dataset). Nos dados espaciais, a maioria dos spots Visium não tem "
                   "tipo celular, porque cada spot mistura células: é aí que entra a deconvolução com o "
                   "single-cell do mesmo doador.", pasta)

    # ---------------------------------------------------------------- pareamento sc + espacial
    rel.secao("Single-cell e espacial do mesmo doador",
              "Para integrar as duas modalidades (deconvolução, mapeamento de tipos celulares no tecido, modelos "
              "conjuntos), o ideal é ter as duas do mesmo doador. O identificador de doador só é comparável "
              "dentro do mesmo estudo, então o pareamento é feito por coleção.")
    linhas = []
    for (cid, nome), g in esp.groupby(["collection_id", "collection_name"], observed=True):
        s = sc[sc["collection_id"] == cid]
        d_esp, d_sc = set(g["donor_id"]), set(s["donor_id"])
        doencas = sorted(set(g["disease"]) | set(s["disease"]))
        linhas.append({"coleção": nome, "doenças": "; ".join(doencas),
                       "câncer": any(RX_CANCER.search(x) for x in doencas),
                       "tecnologia espacial": "; ".join(sorted(set(g["assay"]))),
                       "datasets espaciais": g["dataset_id"].nunique(), "datasets sc": s["dataset_id"].nunique(),
                       "células sc": len(s), "spots": len(g), "doadores espacial": len(d_esp),
                       "doadores sc": len(d_sc), "doadores com os dois": len(d_esp & d_sc),
                       "tipos celulares sc": s["cell_type"].nunique() if len(s) else 0,
                       "DOI": g["collection_doi"].iat[0]})
    par = pd.DataFrame(linhas).sort_values(["câncer", "doadores com os dois"], ascending=False)
    par.to_csv(os.path.join(pasta, "pareamento_por_colecao.csv"), index=False)
    p = par.set_index("coleção")
    so_esp = p["doadores espacial"] - p["doadores com os dois"]
    so_sc = p["doadores sc"] - p["doadores com os dois"]
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(p) + 1.4))
    y = np.arange(len(p))[::-1]
    ax.barh(y, p["doadores com os dois"], color=VERMELHO, label="sc + espacial")
    ax.barh(y, so_esp, left=p["doadores com os dois"], color=LARANJA, label="só espacial")
    ax.barh(y, so_sc, left=p["doadores com os dois"] + so_esp, color="#c9d3e0", label="só single-cell")
    ax.set_yticks(y)
    ax.set_yticklabels([("● " if c else "") + n[:62] for n, c in zip(p.index, p["câncer"])], fontsize=8)
    ax.set_xlabel("doadores")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Doadores por coleção (● = câncer)")
    rel.figura(salvar(fig, pasta, "pareamento_por_colecao"), "Vermelho: doadores com single-cell e espacial no "
               "mesmo estudo. Coleções sem barra vermelha têm doadores diferentes nas duas modalidades (ainda "
               "úteis como referência de tipos celulares, mas não para modelos pareados).", pasta)
    rel.tabela(tabela_html(par.drop(columns=["DOI"]), max_linhas=40), "Pareamento por coleção")

    cancer = par[par["câncer"]]
    res = {"versao": versao, "datasets_espaciais": len(ds_esp), "spots": len(esp), "colecoes_espaciais": len(colecoes),
           "doadores_espaciais": int(esp.groupby(["collection_id", "donor_id"], observed=True).ngroups),
           "por_tecnologia": tab_ds["tecnologia"].value_counts().to_dict(),
           "cancer": {"colecoes": len(cancer), "doadores_espacial": int(cancer["doadores espacial"].sum()),
                      "doadores_pareados": int(cancer["doadores com os dois"].sum()),
                      "celulas_sc": int(cancer["células sc"].sum())},
           "pareados_total": int(par["doadores com os dois"].sum())}
    rel.achado(f"{fmt(len(ds_esp), 0)} datasets espaciais humanos ({', '.join(f'{k}: {v}' for k, v in res['por_tecnologia'].items())}), "
               f"{fmt(len(esp), 0)} spots/beads, de {res['doadores_espaciais']} doadores em {len(colecoes)} estudos.")
    rel.achado(f"Câncer: {len(cancer)} estudos com espacial, {res['cancer']['doadores_espacial']} doadores com "
               f"espacial e só {res['cancer']['doadores_pareados']} com single-cell e espacial do mesmo doador.")

    # Tipos celulares nas coleções de câncer pareadas
    pareadas = cancer[cancer["doadores com os dois"] > 0]["coleção"]
    if len(pareadas):
        s = sc[sc["collection_name"].isin(pareadas)]
        ct = s.groupby(["collection_name", "cell_type"], observed=True).size().rename("células").reset_index()
        ct = ct.sort_values(["collection_name", "células"], ascending=[True, False])
        ct.to_csv(os.path.join(pasta, "tipos_celulares_sc_cancer_pareado.csv"), index=False)
        rel.secao("Tipos celulares do single-cell nas coleções de câncer pareadas")
        top = ct.groupby("collection_name", observed=True).head(12)
        rel.tabela(tabela_html(top, max_linhas=60))

    # Anotações do doador (equivalente ao clínico)
    alvo_col = set(cancer["coleção"]) if len(cancer) else set(par["coleção"])
    cols_d = ["collection_name", "donor_id", "sex", "development_stage", "self_reported_ethnicity", "disease"]
    todos = pd.concat([x.loc[x["collection_name"].isin(alvo_col), cols_d].astype(str) for x in (esp, sc)],
                      ignore_index=True)
    doad = por_doador(todos)
    doad.to_csv(os.path.join(pasta, "doadores_cancer.csv"), index=False)
    comp = (doad[["sexo", "estagio", "etnia", "doenca"]] != "unknown").mean()
    rel.secao("Anotações dos doadores (equivalente ao clínico)",
              "O CELLxGENE só padroniza sexo, estágio de desenvolvimento (idade), etnia autodeclarada, doença e "
              "tecido. Estádio, tratamento, sobrevida e subtipo molecular, quando existem, ficam em colunas "
              "próprias de cada dataset (no h5ad) ou no artigo, sem padrão entre estudos. Completude nos doadores "
              "das coleções de câncer: " + ", ".join(f"{k} {pct(v)}" for k, v in comp.items()) + ".")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, c, t in zip(axs, ["sexo", "estagio", "etnia"], ["Sexo", "Estágio de desenvolvimento", "Etnia"]):
        barras_categoria(ax, doad[c], t, max_cat=8)
        ax.set_xlabel("doadores")
    fig.tight_layout()
    rel.figura(salvar(fig, pasta, "doadores_anotacoes"), "Doadores das coleções de câncer com espacial.", pasta)

    # ---------------------------------------------------------------- single-cell de câncer no Census inteiro
    del todos, sc
    import gc
    gc.collect()
    if not args.sem_cancer_sc:
        log("lendo obs single-cell das doenças oncológicas no Census inteiro")
        cont = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
        dz = cont[(cont["category"] == "disease") & cont["organism"].str.contains("sapiens", case=False)]
        alvo = [d for d in dz["label"] if RX_CANCER.search(d)]
        sc_all = ler_obs(census["census_data"]["homo_sapiens"]["obs"],
                         ["dataset_id", "disease", "donor_id", "assay", "tissue_general", "is_primary_data"],
                         f"disease in {alvo} and is_primary_data == True")
        sc_all["collection_id"] = sc_all["dataset_id"].map(ds.set_index("dataset_id")["collection_id"]).astype("category")
        sc_all["doenca_principal"] = sc_all["disease"].map(
            lambda d: next((x for x in str(d).split(" || ") if RX_CANCER.search(x)), str(d))).astype("category")
        sc_all["TCGA (aprox.)"] = sc_all["doenca_principal"].map(tcga_de).astype("category")
        sc_all["doador"] = (sc_all["collection_id"].astype(str) + "|" + sc_all["donor_id"].astype(str)).astype("category")
        t = sc_all.groupby("doenca_principal", observed=True).agg(
            células=("dataset_id", "size"), datasets=("dataset_id", "nunique"), estudos=("collection_id", "nunique"),
            doadores=("doador", "nunique"),
            tecidos=("tissue_general", "nunique"), tcga=("TCGA (aprox.)", "first")).sort_values("células", ascending=False)
        t.index = t.index.astype(str)
        t.index.name = "doença"
        t.to_csv(os.path.join(pasta, "singlecell_cancer_por_doenca.csv"))
        tt = sc_all.groupby("TCGA (aprox.)", observed=True).agg(células=("dataset_id", "size"), estudos=("collection_id", "nunique"),
                                                  doadores=("doador", "nunique")).sort_values("células", ascending=False)
        tt.to_csv(os.path.join(pasta, "singlecell_cancer_por_tcga.csv"))
        rel.secao("Single-cell de câncer no Census inteiro",
                  "Mesmo sem espacial, o single-cell de câncer do Census serve de referência para deconvoluir "
                  "bulk (TCGA, DepMap) e espacial. Contagem só de células primárias (sem duplicatas entre datasets).")
        top = t.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(top) + 1.2))
        ax.barh([str(k)[:45] for k in top.index], top["células"] / 1e3, color=AZUL)
        for i, (v, d) in enumerate(zip(top["células"], top["doadores"])):
            ax.text(v / 1e3, i, f" {d} doadores", va="center", fontsize=7)
        ax.set_xlabel("mil células")
        ax.grid(axis="y", visible=False)
        ax.set_title("Single-cell por doença oncológica (25 maiores)")
        rel.figura(salvar(fig, pasta, "singlecell_cancer_por_doenca"), "Células e doadores por doença.", pasta)
        rel.tabela(tabela_html(tt.reset_index()), "Por projeto TCGA (mapeamento aproximado)")
        res["singlecell_cancer"] = {"celulas": int(len(sc_all)), "doencas": int(t.shape[0]),
                                    "estudos": int(sc_all["collection_id"].nunique()),
                                    "por_tcga": tt["células"].to_dict()}
        rel.achado(f"Single-cell de câncer no Census: {fmt(len(sc_all), 0)} células primárias, {t.shape[0]} doenças, "
                   f"{sc_all['collection_id'].nunique()} estudos.")

    rel.secao("Como baixar")
    rel.texto("A tabela datasets_espaciais.csv traz o link do h5ad de cada lâmina (com imagem e coordenadas). "
              "Para o single-cell, use cellxgene_census.get_anndata filtrando por dataset_id ou donor_id. "
              "As tabelas desta página estão em CSV na mesma pasta.")
    rel.salvar(os.path.join(pasta, "index.html"))
    with open(os.path.join(pasta, "resumo.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    log("fim")
    return 0
