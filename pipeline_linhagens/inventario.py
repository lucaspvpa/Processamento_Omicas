"""Descobre, em qualquer tabela, quais linhagens ela cobre e em que formato, e carrega matrizes numéricas.

Funciona sem conhecer o formato de antemão: procura a coluna (ou a linha de cabeçalho) cujos valores batem com
os identificadores conhecidos dos modelos (ModelID, nome da linhagem, RRID, ID do Sanger etc.).
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .baixar import abrir, separador

COLUNAS_GENE = ["HugoSymbol", "Hugo_Symbol", "gene_symbol", "GeneSymbol", "Gene", "gene", "symbol", "gene_id",
                "EntrezGeneID", "ensembl_gene_id"]


def norm(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


class Resolvedor:
    """Converte qualquer identificador conhecido de uma linhagem no ID principal do banco."""

    def __init__(self):
        self.mapa: dict[str, str] = {}

    def adicionar(self, chaves, valores):
        for k, v in zip(chaves, valores):
            if pd.isna(k) or pd.isna(v):
                continue
            for parte in str(k).split(";"):
                n = norm(parte)
                if len(n) >= 3 and n not in self.mapa:
                    self.mapa[n] = str(v)

    def __call__(self, x):
        v = self.mapa.get(norm(x))
        if v is None and "_" in str(x):
            # Nomes com sufixo de lote, como MDAMB468_BREAST_TenPx01 no proteoma do CCLE
            s = str(x)
            while v is None and "_" in s:
                s = s.rsplit("_", 1)[0]
                if len(norm(s)) >= 4:
                    v = self.mapa.get(norm(s))
        return v

    def serie(self, s: pd.Series) -> pd.Series:
        u = pd.Series(s.dropna().unique())
        m = dict(zip(u, u.map(self)))
        return s.map(m)

    def fracao(self, valores) -> float:
        v = [x for x in pd.Series(valores).dropna().unique()[:3000]]
        if not v:
            return 0.0
        return float(np.mean([self(x) is not None for x in v]))


def _pular(nome: str) -> int:
    return 2 if nome.lower().endswith((".gct", ".gct.gz")) else 0


def detectar(nome: str, fonte, res: Resolvedor) -> dict | None:
    """Formato da tabela e modelos cobertos. None quando a tabela não é indexada por modelo."""
    sep = "\t" if _pular(nome) else separador(fonte)
    with abrir(fonte) as f:
        amostra = pd.read_csv(f, sep=sep, nrows=3000, dtype=str, skiprows=_pular(nome), low_memory=False)
    cab = list(amostra.columns)
    frac_cab = res.fracao(cab)
    n_cab = sum(res(c) is not None for c in cab)

    melhor, frac_col = None, 0.0
    for c in cab[:20]:
        fr = res.fracao(amostra[c])
        if fr > frac_col:
            melhor, frac_col = c, fr

    if melhor is not None and frac_col >= 0.5 and not (n_cab >= 20 and frac_col < 0.9):
        col_gene = next((g for g in COLUNAS_GENE if g in cab and g != melhor), None)
        usar = [melhor] + ([col_gene] if col_gene else [])
        ids, genes, n = set(), set(), 0
        with abrir(fonte) as f:
            for bloco in pd.read_csv(f, sep=sep, usecols=usar, dtype=str, chunksize=500_000,
                                     skiprows=_pular(nome)):
                n += len(bloco)
                ids.update(bloco[melhor].dropna().unique())
                if col_gene:
                    genes.update(bloco[col_gene].dropna().unique())
        modelos = {m for m in (res(i) for i in ids) if m}
        numericas = [c for c in cab if c != melhor and pd.to_numeric(amostra[c], errors="coerce").notna().mean() > 0.5]
        longo = n > 1.2 * max(1, len(ids)) and len(numericas) < 100
        return {"formato": "longo (uma linha por evento)" if longo else "matriz (modelos nas linhas)",
                "orientacao": "longo" if longo else "linhas", "coluna_id": melhor, "coluna_gene": col_gene,
                "modelos": modelos, "linhas": n, "ids_brutos": len(ids),
                "variaveis": len(genes) if longo and col_gene else (None if longo else len(numericas)),
                "sep": sep}

    if n_cab >= 20 or (frac_cab >= 0.5 and n_cab >= 3):
        n = 0
        with abrir(fonte) as f:
            for linha in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                n += 1
        modelos = {m for m in (res(c) for c in cab) if m}
        return {"formato": "matriz (modelos nas colunas)", "orientacao": "colunas", "coluna_id": None,
                "coluna_gene": None, "modelos": modelos, "linhas": n - 1 - _pular(nome), "ids_brutos": n_cab,
                "variaveis": n - 1 - _pular(nome), "sep": sep}
    return None


def carregar_matriz(nome: str, fonte, info: dict, res: Resolvedor, max_variaveis: int = 60_000) -> pd.DataFrame | None:
    """Matriz modelos x variáveis (float32). Só para formatos de matriz."""
    sep, pular = info["sep"], _pular(nome)
    if info["orientacao"] == "linhas":
        partes = []
        with abrir(fonte) as f:
            for bloco in pd.read_csv(f, sep=sep, dtype=str, chunksize=200, skiprows=pular, low_memory=False):
                if "IsDefaultEntryForModel" in bloco.columns:
                    bloco = bloco[bloco["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true", "1"])]
                ids = bloco[info["coluna_id"]]
                num = bloco.drop(columns=[info["coluna_id"]]).apply(pd.to_numeric, errors="coerce").astype(np.float32)
                num.index = res.serie(ids).values
                partes.append(num[num.index.notna()])
        X = pd.concat(partes)
    elif info["orientacao"] == "colunas":
        partes = []
        with abrir(fonte) as f:
            for bloco in pd.read_csv(f, sep=sep, dtype=str, chunksize=2000, skiprows=pular, low_memory=False):
                idx = bloco.iloc[:, 0].astype(str)
                cols = [c for c in bloco.columns[1:] if res(c)]
                num = bloco[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
                num.index = idx
                partes.append(num.dropna(how="all"))
                if sum(len(p) for p in partes) >= max_variaveis * 3:
                    break
        X = pd.concat(partes).T
        X.index = [res(c) for c in X.index]
    else:
        return None
    X = X.dropna(axis=1, how="all").dropna(axis=0, how="all")
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.groupby(level=0).mean() if X.index.duplicated().any() else X
    if X.shape[1] > max_variaveis:
        X = X[X.var().sort_values(ascending=False).index[:max_variaveis]]
    return X.astype(np.float32)


def carregar_longo(fonte, info: dict, res: Resolvedor, colunas_extra=()) -> pd.DataFrame:
    """Tabela longa só com modelo, gene e as colunas extras que existirem."""
    sep = info["sep"]
    with abrir(fonte) as f:
        cab = pd.read_csv(f, sep=sep, nrows=0).columns
    usar = [info["coluna_id"]] + [c for c in [info.get("coluna_gene"), *colunas_extra] if c and c in cab]
    usar = list(dict.fromkeys(usar))
    partes = []
    with abrir(fonte) as f:
        for bloco in pd.read_csv(f, sep=sep, usecols=usar, dtype=str, chunksize=500_000):
            bloco["modelo"] = res.serie(bloco[info["coluna_id"]])
            partes.append(bloco[bloco["modelo"].notna()])
    df = pd.concat(partes, ignore_index=True)
    if info.get("coluna_gene"):
        df = df.rename(columns={info["coluna_gene"]: "gene"})
    return df
