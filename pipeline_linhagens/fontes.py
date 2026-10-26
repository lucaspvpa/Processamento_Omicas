"""Onde ficam os arquivos do DepMap e do Cell Model Passports e quais deles o pipeline usa.

Os nomes dos arquivos mudam entre versões. Por isso cada ômica é escolhida por expressão regular sobre o nome
do arquivo e, quando várias versões batem, fica a mais recente. Se nenhuma bater, o pipeline grava a lista
completa de arquivos disponíveis (arquivos_disponiveis.csv) para ajustar as expressões abaixo.
"""
from __future__ import annotations

import io
import json
import re

import pandas as pd

from .baixar import obter_texto

DEPMAP_LISTAS = [
    "https://depmap.org/portal/api/download/files",          # CSV: release, release_date, filename, url, ...
    "https://depmap.org/portal/download/api/downloads",      # JSON usado pela página de downloads
]
CMP_PAGINA = "https://cellmodelpassports.sanger.ac.uk/downloads"
CMP_BASE = "https://cog.sanger.ac.uk/cmp/download/"

# chave, rótulo, regex da versão (release), regex do nome do arquivo
DEPMAP_ALVOS = [
    ("modelo", "Anotação dos modelos", r"DepMap Public", r"^Model\.csv$"),
    ("perfis", "Perfis ômicos (mapa)", r"DepMap Public", r"^OmicsProfiles\.csv$"),
    ("condicao", "Condições de cultivo (mapa)", r"DepMap Public", r"^ModelCondition\.csv$"),
    ("expressao", "RNA-seq (TPM)", r"DepMap Public",
     r"^OmicsExpression.*Protein.*Coding.*TPM.*Logp1.*\.csv$|^OmicsExpressionTPMLogp1HumanProteinCodingGenes\.csv$"),
    ("cnv", "CNV por gene", r"DepMap Public", r"^OmicsCNGene(WGS)?\.csv$"),
    ("mutacao", "Mutações somáticas", r"DepMap Public", r"^OmicsSomaticMutations\.csv$"),
    ("fusao", "Fusões gênicas", r"DepMap Public", r"^OmicsFusionFiltered\.csv$"),
    ("assinaturas", "Assinaturas genômicas (MSI, ploidia, LOH)", r"DepMap Public", r"^OmicsSignatures\.csv$"),
    ("crispr", "CRISPR (Chronos)", r"DepMap Public", r"^CRISPRGeneEffect\.csv$"),
    ("rnai", "RNAi (DEMETER2)", r"DEMETER2", r"^D2_combined_gene_dep_scores\.csv$"),
    ("droga", "Fármacos (PRISM Repurposing)", r"PRISM",
     r"Repurposing.*Primary.*Data_Matrix\.csv$|primary-screen-replicate-collapsed-logfold-change\.csv$"),
    ("proteoma", "Proteoma MS (Nusinow 2020)", r"Proteomics|Harmonized", r"protein_quant_current_normalized\.csv$"),
    ("rppa", "RPPA (CCLE 2019)", r"CCLE 2019", r"^CCLE_RPPA_\d+\.csv$"),
    ("metilacao", "Metilação RRBS (CCLE 2019)", r"CCLE 2019", r"^CCLE_RRBS_TSS1kb_\d+\.txt"),
    ("mirna", "miRNA (CCLE 2019)", r"CCLE 2019", r"^CCLE_miRNA_\d+\.gct"),
    ("metaboloma", "Metaboloma (CCLE 2019)", r"CCLE 2019", r"^CCLE_metabolomics_\d+\.csv$"),
]
# Arquivos de mapa (não contam como ômica no UpSet)
MAPAS = {"modelo", "perfis", "condicao"}

# chave, rótulo, regex do nome do arquivo (o mais recente vence)
CMP_ALVOS = [
    ("modelo", "Anotação dos modelos", r"^model_list_(\d{8}|latest)\.csv(\.gz)?$"),
    ("expressao", "RNA-seq", r"^rnaseq_(all|tpm|merged)_?(\d{8}|latest)?\.(zip|csv(\.gz)?)$"),
    ("mutacao", "Mutações", r"^mutations_(all|summary)_?(\d{8}|latest)?\.(zip|csv(\.gz)?)$"),
    ("cnv", "CNV", r"^(WES_pureCN_CNV_genes|cnv_summary|cnv_.*genes?)_?(\d{8}|latest)?\.(zip|csv(\.gz)?)$"),
    ("proteoma", "Proteoma", r"^[Pp]roteomics.*\.(zip|csv(\.gz)?)$"),
    ("metilacao", "Metilação", r"^[Mm]ethylation.*\.(zip|csv(\.gz)?)$"),
    ("crispr", "CRISPR (Project Score)", r"(Project_?Score|[Cc][Rr][Ii][Ss][Pp][Rr]|gene_?effect).*\.(zip|csv(\.gz)?|tsv)$"),
    ("droga", "Fármacos (GDSC)", r"(GDSC|[Dd]rug).*\.(zip|csv(\.gz)?|xlsx)$"),
]

# Oncotree (DepMap) -> projeto TCGA. Aproximado: serve para cruzar com o levantamento do TCGA.
ONCOTREE_TCGA = {
    "ACC": "ACC", "BLCA": "BLCA", "UTUC": "BLCA", "IDC": "BRCA", "ILC": "BRCA", "BRCA": "BRCA", "BRCNOS": "BRCA",
    "MDLC": "BRCA", "CESC": "CESC", "CSCC": "CESC", "CEAD": "CESC", "CHOL": "CHOL", "IHCH": "CHOL", "EHCH": "CHOL",
    "COAD": "COAD", "COADREAD": "COAD", "READ": "READ", "DLBCLNOS": "DLBC", "DLBCL": "DLBC", "ESCC": "ESCA",
    "EAC": "ESCA", "GBM": "GBM", "HNSC": "HNSC", "OCSC": "HNSC", "OPHSC": "HNSC", "HPHSC": "HNSC", "LXSC": "HNSC",
    "CCRCC": "KIRC", "PRCC": "KIRP", "CHRCC": "KICH", "AML": "LAML", "AMLNOS": "LAML", "ASTR": "LGG", "ODG": "LGG",
    "DIFG": "LGG", "HCC": "LIHC", "LUAD": "LUAD", "LUSC": "LUSC", "PLMESO": "MESO", "MESO": "MESO", "HGSOC": "OV",
    "SOC": "OV", "OVT": "OV", "PAAD": "PAAD", "PHC": "PCPG", "PRAD": "PRAD", "SKCM": "SKCM", "MEL": "SKCM",
    "STAD": "STAD", "TGCT": "TGCT", "THPA": "THCA", "THFO": "THCA", "THCA": "THCA", "THYM": "THYM", "UCEC": "UCEC",
    "UEC": "UCEC", "UCS": "UCS", "UM": "UVM", "SARCNOS": "SARC", "LMS": "SARC", "DDLS": "SARC", "UPS": "SARC",
    "MFS": "SARC", "SYNS": "SARC",
}


def _ordem_versao(nome: str) -> tuple:
    """'DepMap Public 25Q2' -> (25, 2). Versões sem ano/trimestre vão para o fim."""
    m = re.search(r"(\d{2})Q(\d)", str(nome))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(20\d{2})", str(nome))
    return (int(m.group(1)) - 2000, 0) if m else (0, 0)


def lista_depmap(s) -> pd.DataFrame:
    """Tabela com colunas release, filename, url (uma linha por arquivo de todas as versões)."""
    erros = []
    for url in DEPMAP_LISTAS:
        try:
            txt = obter_texto(s, url)
            if txt.lstrip().startswith(("{", "[")):
                js = json.loads(txt)
                linhas = js.get("table", js) if isinstance(js, dict) else js
                df = pd.DataFrame(linhas)
                ren = {"releaseName": "release", "fileName": "filename", "downloadUrl": "url",
                       "date": "release_date", "releaseDate": "release_date"}
                df = df.rename(columns=ren)
            else:
                df = pd.read_csv(io.StringIO(txt))
            df.columns = [c.strip() for c in df.columns]
            if {"release", "filename", "url"} <= set(df.columns):
                return df
            erros.append(f"{url}: colunas {list(df.columns)}")
        except Exception as e:
            erros.append(f"{url}: {e}")
    raise RuntimeError("Não consegui a lista de arquivos do DepMap. " + " | ".join(erros))


def escolher_depmap(lista: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por alvo: a versão mais recente que tem o arquivo."""
    linhas = []
    for chave, rotulo, rx_rel, rx_arq in DEPMAP_ALVOS:
        cand = lista[lista["release"].astype(str).str.contains(rx_rel, regex=True, case=False)
                     & lista["filename"].astype(str).str.contains(rx_arq, regex=True, case=False)].copy()
        if cand.empty:
            linhas.append({"chave": chave, "rotulo": rotulo, "release": None, "filename": None, "url": None})
            continue
        cand["_ord"] = cand["release"].map(_ordem_versao)
        if "release_date" in cand.columns:
            cand["_data"] = pd.to_datetime(cand["release_date"], errors="coerce")
            cand = cand.sort_values(["_data", "_ord"], ascending=False, na_position="last")
        else:
            cand = cand.sort_values("_ord", ascending=False)
        r = cand.iloc[0]
        linhas.append({"chave": chave, "rotulo": rotulo, "release": r["release"], "filename": r["filename"],
                       "url": r["url"]})
    return pd.DataFrame(linhas)


def lista_cmp(s) -> pd.DataFrame:
    """Links de download da página do Cell Model Passports."""
    html = obter_texto(s, CMP_PAGINA)
    links = sorted(set(re.findall(r"https?://cog\.sanger\.ac\.uk/cmp/download/[^\"'\s<>)]+", html)))
    rel = re.findall(r"[\"'](/?cmp/download/[^\"'\s<>)]+)", html)
    links += [CMP_BASE + r.split("cmp/download/")[1] for r in rel]
    links = sorted(set(links))
    return pd.DataFrame({"filename": [l.rsplit("/", 1)[1] for l in links], "url": links})


def _data_nome(nome: str) -> str:
    if "latest" in nome:
        return "99999999"
    m = re.findall(r"(20\d{6})", nome)
    return max(m) if m else "00000000"


def escolher_cmp(lista: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for chave, rotulo, rx in CMP_ALVOS:
        cand = lista[lista["filename"].str.contains(rx, regex=True)].copy()
        if cand.empty:
            linhas.append({"chave": chave, "rotulo": rotulo, "filename": None, "url": None})
            continue
        cand["_d"] = cand["filename"].map(_data_nome)
        r = cand.sort_values("_d", ascending=False).iloc[0]
        linhas.append({"chave": chave, "rotulo": rotulo, "filename": r["filename"], "url": r["url"]})
    return pd.DataFrame(linhas)
