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
                "EntrezGeneID", "ensembl_gene_id", "FusionName"]


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
    # As primeiras colunas e, depois delas, as que têm nome de identificador (no OmicsSomaticMutations do
    # DepMap 24Q4 o ModelID é a 68ª coluna)
    for c in cab[:20] + [c for c in cab[20:] if re.search(r"model|depmap|cell_?line|ccle|sample", c, re.I)]:
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


# ---------------------------------------------------------------- preparo dos arquivos do Passports

def preparar(tabelas: list[tuple[str, object]], cfg: dict, destino: str, log=print) -> list[tuple[str, object]]:
    """Aplica a receita de leitura (fontes.CMP_LEITURA) e devolve as tabelas no formato que detectar entende.

    membro: regex do nome da tabela a usar dentro do arquivo.
    modo: longo_matriz (tabela longa -> matriz modelos x variáveis), longo_filtro (tabela longa só com modelo e
    gene) ou colunas_meta (matriz com várias linhas de cabeçalho, modelos nas colunas -> modelos nas linhas).
    fonte: mantém só as linhas/colunas cuja coluna source/data_source tem esse valor.
    """
    if cfg.get("membro"):
        tabelas = [t for t in tabelas if re.search(cfg["membro"], t[0])]
    modo = cfg.get("modo")
    if not modo or not tabelas:
        return tabelas
    nome, fonte = tabelas[0]
    log(f"    preparando {nome} ({modo})")
    if modo == "colunas_meta":
        X = _matriz_com_meta(fonte, cfg)
    elif modo == "longo_matriz":
        X = _longo_para_matriz(nome, fonte, cfg)
    elif modo == "longo_filtro":
        n = 0
        for i, b in enumerate(_ler_longo(nome, fonte, cfg)):
            b.rename(columns={"id": "model_id"}).to_csv(destino, index=False, mode="w" if i == 0 else "a",
                                                         header=i == 0)
            n += len(b)
        log(f"    {n:,} linhas mantidas")
        return [(nome, destino)]
    else:
        raise ValueError(f"modo de leitura desconhecido: {modo}")
    X.index.name = "model_id"
    X.to_csv(destino, float_format="%.5g")
    log(f"    matriz preparada: {X.shape[0]} modelos x {X.shape[1]} variáveis")
    return [(nome, destino)]


def _ler_longo(nome: str, fonte, cfg: dict, bloco: int = 1_000_000):
    """Blocos com as colunas id, gene e valor (as que a receita pede), já filtrados pela fonte."""
    id_, genes, valor = cfg["id"], cfg.get("gene", []), cfg.get("valor")
    if nome.lower().endswith(".xlsx"):
        partes = [pd.read_excel(fonte, dtype=str)]
        cab = list(partes[0].columns)
    else:
        sep = separador(fonte)
        with abrir(fonte) as f:
            cab = list(pd.read_csv(f, sep=sep, nrows=0).columns)
    col_f = next((c for c in ["data_source", "source"] if c in cab), None) if cfg.get("fonte") else None
    usar = [c for c in dict.fromkeys([id_, *genes, valor, col_f]) if c]
    faltam = [c for c in usar if c not in cab]
    if faltam:
        raise RuntimeError(f"colunas ausentes em {nome}: {faltam}")
    if nome.lower().endswith(".xlsx"):
        it = partes
    else:
        f = abrir(fonte)
        it = pd.read_csv(f, sep=sep, usecols=usar, dtype=str, chunksize=bloco)
    for b in it:
        if col_f:
            b = b[b[col_f] == cfg["fonte"]]
        out = pd.DataFrame({"id": b[id_].values})
        if genes:
            g = b[genes[0]].fillna("")
            for c in genes[1:]:
                g = g.str.cat(b[c].fillna(""), sep=cfg.get("sep_gene", "_"))
            out["gene"] = g.values
        if valor:
            out["valor"] = pd.to_numeric(b[valor], errors="coerce").astype(np.float32).values
        yield out.dropna()


def _longo_para_matriz(nome: str, fonte, cfg: dict) -> pd.DataFrame:
    somas, contagens, largura = [], [], 0

    def consolidar(partes):
        return pd.concat(partes, axis=1).T.groupby(level=0).sum().T

    for b in _ler_longo(nome, fonte, cfg):
        if b.empty:
            continue
        a = b.groupby(["id", "gene"])["valor"].agg(["sum", "count"])
        somas.append(a["sum"].unstack().astype(np.float64))
        contagens.append(a["count"].unstack().astype(np.float32))
        largura += somas[-1].shape[1]
        if largura > 150_000:
            somas, contagens = [consolidar(somas)], [consolidar(contagens)]
            largura = somas[0].shape[1]
    if not somas:
        raise RuntimeError(f"{nome}: nenhuma linha depois do filtro")
    S, C = consolidar(somas), consolidar(contagens)
    X = (S / C.where(C > 0)).astype(np.float32)
    if cfg.get("log2p1"):
        X = np.log2(X.clip(lower=0) + 1).astype(np.float32)
    return X


def _matriz_com_meta(fonte, cfg: dict) -> pd.DataFrame:
    """Matriz do Sanger com linhas de cabeçalho (model_name, model_id, source, qc_pass...) e genes nas linhas."""
    sep = separador(fonte)
    with abrir(fonte) as f:
        meta = pd.read_csv(f, sep=sep, header=None, nrows=10, dtype=str)
    rot = meta.iloc[:, 0].fillna("").str.strip().str.lower()
    i_id = next((i for i, r in enumerate(rot) if r == "model_id"), None)
    if i_id is None:
        raise RuntimeError("matriz sem linha model_id no cabeçalho")
    linha_id = meta.iloc[i_id]
    nlab = next(j for j in range(1, meta.shape[1]) if pd.notna(linha_id.iloc[j]))
    ini = next(i for i in range(len(meta))
               if pd.to_numeric(meta.iloc[i, nlab:], errors="coerce").notna().mean() > 0.5)
    manter = pd.Series(True, index=range(nlab, meta.shape[1]))
    for nomes, alvo in [({"source", "data_source"}, cfg.get("fonte")), ({"qc_pass"}, "TRUE")]:
        for i in [i for i, r in enumerate(rot[:ini]) if r in nomes]:
            if alvo:
                manter &= meta.iloc[i, nlab:].fillna("").str.upper().values == alvo.upper()
    cols = [c for c in manter.index if manter[c]]
    if not cols:
        raise RuntimeError("nenhuma coluna depois do filtro de fonte")
    rot_gene = meta.iloc[ini - 1, :nlab].fillna("").str.lower().tolist() if ini > 0 else []
    col_gene = next((j for j, r in enumerate(rot_gene) if r in ("symbol", "gene_symbol")), 0)
    with abrir(fonte) as f:
        dados = pd.read_csv(f, sep=sep, header=None, skiprows=ini, usecols=[col_gene] + cols, index_col=0,
                            dtype={c: np.float32 for c in cols}, na_values=["NA", "NaN", ""])
    dados = dados[dados.index.notna()]
    dados = dados[~dados.index.duplicated()]
    X = dados.T
    X.index = linha_id.iloc[cols].values
    return X.groupby(level=0).mean().astype(np.float32)


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
