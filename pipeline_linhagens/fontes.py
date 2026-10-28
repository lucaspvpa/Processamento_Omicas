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
# O portal do DepMap passou a exigir verificação de navegador (Cloudflare Turnstile). Sem ele, a lista vem dos
# releases que o próprio DepMap publica no Figshare, da pasta antiga do CCLE (CCLE 2019) e do site do Gygi.
FIGSHARE_API = "https://api.figshare.com/v2"
FIGSHARE_AUTOR = "Broad DepMap"
FIGSHARE_FIXOS = [27993248, 9170975, 25917643]           # DepMap 24Q4 Public, DEMETER2, Repurposing Public 24Q2
CCLE_LEGADO = "https://data.broadinstitute.org/ccle/"
CCLE_LEGADO_DATA = "2019-05-08"
PROTEOMA_GYGI = [("Proteomics (Nusinow et al. 2020)", "2020-01-09",
                  "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz")]
GYGI_PAGINA = "https://gygi.hms.harvard.edu/publications/ccle.html"
DEPMAP_PAGINA = "https://depmap.org/portal/data_page/?tab=allData"
FIGSHARE_DEPMAP = "https://figshare.com/authors/Broad_DepMap/5514062"
CMP_API = "https://api.cellmodelpassports.sanger.ac.uk/download_files?include=groups.files"
CMP_PAGINA = "https://cellmodelpassports.sanger.ac.uk/downloads"
CMP_BASE = "https://cog.sanger.ac.uk/cmp/download/"

# No portal a versão se chama "DepMap Public 25Q2"; no Figshare, "DepMap 24Q4 Public".
DM_PUBLIC = r"^DepMap Public|^DepMap \d{2}Q\d Public"

# chave, rótulo, regex da versão (release), regex do nome do arquivo
DEPMAP_ALVOS = [
    ("modelo", "Anotação dos modelos", DM_PUBLIC, r"^Model\.csv$"),
    ("perfis", "Perfis ômicos (mapa)", DM_PUBLIC, r"^OmicsProfiles\.csv$"),
    ("condicao", "Condições de cultivo (mapa)", DM_PUBLIC, r"^ModelCondition\.csv$"),
    ("expressao", "RNA-seq (TPM)", DM_PUBLIC,
     r"^OmicsExpressionProteinCodingGenesTPMLogp1\.csv$|^OmicsExpressionTPMLogp1HumanProteinCodingGenes\.csv$"),
    ("cnv", "CNV por gene", DM_PUBLIC, r"^OmicsCNGene(WGS)?\.csv$"),
    ("mutacao", "Mutações somáticas", DM_PUBLIC, r"^OmicsSomaticMutations\.csv$"),
    ("fusao", "Fusões gênicas", DM_PUBLIC, r"^OmicsFusionFiltered\.csv$"),
    ("assinaturas", "Assinaturas genômicas (MSI, ploidia, LOH)", DM_PUBLIC, r"^OmicsSignatures\.csv$"),
    ("crispr", "CRISPR (Chronos)", DM_PUBLIC, r"^CRISPRGeneEffect\.csv$"),
    ("rnai", "RNAi (DEMETER2)", r"DEMETER\s?2", r"^D2_combined_gene_dep_scores\.csv$|^gene_effect\.csv$"),
    ("droga", "Fármacos (PRISM Repurposing)", r"PRISM|Repurposing",
     r"Repurposing.*Primary.*Data_Matrix\.csv$|primary-screen-replicate-collapsed-logfold-change\.csv$"),
    ("proteoma", "Proteoma MS (Nusinow 2020)", r"Proteomics|Harmonized",
     r"protein_quant_current_normalized\.csv(\.gz)?$"),
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
    ("expressao", "RNA-seq", r"^rnaseq_all_\d{8}\.zip$"),
    ("mutacao", "Mutações", r"^mutations_all_(\d{8}|latest)\.(zip|csv\.gz)$"),
    ("cnv", "CNV", r"^WES_pureCN_CNV_genes_\d{8}\.zip$"),
    ("fusao", "Fusões gênicas", r"^fusions_latest\.csv\.gz$|^Fusions_\d{8}\.zip$"),
    ("proteoma", "Proteoma", r"^Proteomics_\d{8}\.zip$"),
    ("metilacao", "Metilação", r"^[Mm]ethylation.*\.(zip|csv(\.gz)?)$"),
    ("crispr", "CRISPR (Project Score)", r"^Project_Score2?_fitness_scores_.*\.zip$"),
    ("droga", "Fármacos (GDSC2)", r"^GDSC2_fitted_dose_response_.*\.xlsx$"),
]
# Como ler cada arquivo do Passports (ver inventario.preparar). As matrizes do Sanger trazem várias linhas de
# cabeçalho e misturam dados gerados pelo Sanger e pelo Broad; "fonte" mantém só os do Sanger, para que a
# comparação com o DepMap seja entre medições independentes.
CMP_LEITURA = {
    "expressao": {"modo": "longo_matriz", "membro": r"^rnaseq_all", "id": "model_id", "gene": ["gene_symbol"],
                  "valor": "rsem_tpm", "fonte": "Sanger", "log2p1": True},
    "mutacao": {"modo": "longo_filtro", "id": "model_id", "gene": ["gene_symbol"], "fonte": "Sanger"},
    "cnv": {"modo": "colunas_meta", "membro": r"total_copy_number", "fonte": "Sanger"},
    "fusao": {"modo": "longo_filtro", "id": "model_id", "gene": ["gene_symbol_5prime", "gene_symbol_3prime"],
              "sep_gene": "--"},
    "proteoma": {"membro": r"Protein_matrix_averaged_\d"},
    "crispr": {"modo": "colunas_meta", "membro": r"fold_change", "fonte": "Sanger"},
    "droga": {"modo": "longo_matriz", "id": "SANGER_MODEL_ID", "gene": ["DRUG_NAME", "DRUG_ID"], "valor": "LN_IC50",
              "sep_gene": " #"},
}

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
            if txt.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
                raise RuntimeError("o portal devolveu uma página HTML (verificação de navegador)")
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
    try:
        return lista_depmap_alternativa(s)
    except Exception as e:
        erros.append(f"Figshare/CCLE: {e}")
    raise RuntimeError("Não consegui a lista de arquivos do DepMap. " + " | ".join(erros))


def lista_depmap_alternativa(s) -> pd.DataFrame:
    """Mesma tabela de lista_depmap, montada com o Figshare do DepMap, a pasta antiga do CCLE e o site do Gygi."""
    ids = list(FIGSHARE_FIXOS)
    try:
        for pagina in range(1, 4):
            r = s.post(f"{FIGSHARE_API}/articles/search", timeout=60,
                       json={"search_for": f':author: "{FIGSHARE_AUTOR}"', "page_size": 100, "page": pagina})
            r.raise_for_status()
            achados = [a["id"] for a in r.json()]
            ids += achados
            if len(achados) < 100:
                break
    except Exception:
        pass
    linhas = []
    for aid in dict.fromkeys(ids):
        art = json.loads(obter_texto(s, f"{FIGSHARE_API}/articles/{aid}"))
        arquivos = json.loads(obter_texto(s, f"{FIGSHARE_API}/articles/{aid}/files?page_size=1000"))
        for f in arquivos:
            linhas.append({"release": art["title"].strip(), "release_date": art.get("published_date"),
                           "filename": f["name"], "url": f["download_url"], "size": f.get("size"),
                           "origem": "figshare", "pagina": art.get("url_public_html") or art.get("figshare_url")})
    try:
        html = obter_texto(s, CCLE_LEGADO)
        for nome in sorted(set(re.findall(r'href="(CCLE_[^"/]+)"', html))):
            linhas.append({"release": "CCLE 2019", "release_date": CCLE_LEGADO_DATA, "filename": nome,
                           "url": CCLE_LEGADO + nome, "size": None, "origem": "data.broadinstitute.org/ccle",
                           "pagina": CCLE_LEGADO})
    except Exception:
        pass
    for rel, data, url in PROTEOMA_GYGI:
        linhas.append({"release": rel, "release_date": data, "filename": url.rsplit("/", 1)[1], "url": url,
                       "size": None, "origem": "gygi.hms.harvard.edu", "pagina": GYGI_PAGINA})
    if not linhas:
        raise RuntimeError("nenhum arquivo encontrado")
    return pd.DataFrame(linhas)


def escolher_depmap(lista: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por alvo: a versão mais recente que tem o arquivo."""
    linhas = []
    for chave, rotulo, rx_rel, rx_arq in DEPMAP_ALVOS:
        cand = lista[lista["release"].astype(str).str.contains(rx_rel, regex=True, case=False)
                     & lista["filename"].astype(str).str.contains(rx_arq, regex=True, case=False)].copy()
        if cand.empty:
            linhas.append({"chave": chave, "rotulo": rotulo, "release": None, "filename": None, "url": None,
                           "pagina": None})
            continue
        cand["_ord"] = cand["release"].map(_ordem_versao)
        cand["_arq"] = cand["filename"].map(_data_nome)  # desempate: data no nome (CCLE_RPPA_20181003.csv)
        if "release_date" in cand.columns:
            cand["_data"] = pd.to_datetime(cand["release_date"], errors="coerce", utc=True, format="mixed")
            cand = cand.sort_values(["_data", "_ord", "_arq"], ascending=False, na_position="last")
        else:
            cand = cand.sort_values(["_ord", "_arq"], ascending=False)
        r = cand.iloc[0]
        pagina = r.get("pagina") if isinstance(r.get("pagina"), str) else DEPMAP_PAGINA
        linhas.append({"chave": chave, "rotulo": rotulo, "release": r["release"], "filename": r["filename"],
                       "url": r["url"], "pagina": pagina})
    return pd.DataFrame(linhas)


def lista_cmp(s) -> pd.DataFrame:
    """Links de download do Cell Model Passports (API JSON da página de downloads; se falhar, o HTML)."""
    try:
        js = json.loads(obter_texto(s, CMP_API))
        links = set()
        for x in js.get("included", []):
            for k in ("url", "latest_link"):
                u = (x.get("attributes") or {}).get(k)
                if u and re.search(r"cog\.sanger\.ac\.uk/(cmp/)?download/[^/]+\.\w+$", u):
                    links.add(u)
        if links:
            links = sorted(links)
            return pd.DataFrame({"filename": [l.rsplit("/", 1)[1] for l in links], "url": links})
    except Exception:
        pass
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
        cand = lista[lista["filename"].astype(str).str.contains(rx, regex=True)].copy()
        if cand.empty:
            linhas.append({"chave": chave, "rotulo": rotulo, "filename": None, "url": None, "pagina": None})
            continue
        cand["_d"] = cand["filename"].map(_data_nome)
        r = cand.sort_values("_d", ascending=False).iloc[0]
        linhas.append({"chave": chave, "rotulo": rotulo, "filename": r["filename"], "url": r["url"],
                       "release": f"versão de {r['_d'][:4]}-{r['_d'][4:6]}-{r['_d'][6:]}" if r["_d"] not in
                       ("00000000", "99999999") else "mais recente", "pagina": CMP_PAGINA})
    return pd.DataFrame(linhas)
