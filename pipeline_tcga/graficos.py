"""Estilo dos gráficos e funções estatísticas usadas por todas as análises."""
from __future__ import annotations

import itertools
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

AZUL = "#2f6690"
LARANJA = "#d9822b"
VERDE = "#3a7d44"
VERMELHO = "#b23a48"
CINZA = "#8d99ae"
ROXO = "#6c4f8c"
CORES = [AZUL, LARANJA, VERDE, VERMELHO, ROXO, "#2a9d8f", "#9c6644", "#e9c46a", "#577590", "#c06c84"]
COR_TIPO = {"Tumor": VERMELHO, "Normal": AZUL, "Outro": CINZA}
COR_ESTADIO = {"I": "#a8dadc", "II": "#457b9d", "III": "#e9c46a", "IV": "#b23a48", "NA": "#d9d9d9"}


def tema():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9, "axes.prop_cycle": matplotlib.cycler(color=CORES),
    })


def salvar(fig, pasta: str, nome: str) -> str:
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, nome + ".png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def fdr_bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    ordem = np.argsort(pv)
    n = len(pv)
    q = pv[ordem] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(n)
    res[ordem] = np.clip(q, 0, 1)
    out[ok] = res
    return out


def spearman_linhas(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Spearman entre linhas pareadas de A e B (mesmo índice, mesmas colunas)."""
    A, B = A.align(B, join="inner", axis=None)
    mask = np.isfinite(A.to_numpy(dtype=float)) & np.isfinite(B.to_numpy(dtype=float))
    rho = np.full(len(A), np.nan)
    pval = np.full(len(A), np.nan)
    nn = mask.sum(axis=1)
    for i in range(len(A)):
        m = mask[i]
        if m.sum() < 5:
            continue
        x, y = A.iloc[i, m].to_numpy(dtype=float), B.iloc[i, m].to_numpy(dtype=float)
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            continue
        r, p = stats.spearmanr(x, y)
        rho[i], pval[i] = r, p
    return pd.DataFrame({"rho": rho, "p": pval, "n": nn, "fdr": fdr_bh(pval)}, index=A.index)


def corr_spearman_matriz(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Matriz de Spearman entre linhas de X e linhas de Y (colunas = amostras em comum)."""
    comuns = X.columns.intersection(Y.columns)
    rx = X[comuns].rank(axis=1)
    ry = Y[comuns].rank(axis=1)
    rx = rx.sub(rx.mean(axis=1), axis=0).fillna(0)
    ry = ry.sub(ry.mean(axis=1), axis=0).fillna(0)
    nx = np.sqrt((rx ** 2).sum(axis=1)).replace(0, np.nan)
    ny = np.sqrt((ry ** 2).sum(axis=1)).replace(0, np.nan)
    c = (rx.to_numpy() @ ry.to_numpy().T) / np.outer(nx, ny)
    return pd.DataFrame(c, index=X.index, columns=Y.index)


def padronizar(X: pd.DataFrame) -> pd.DataFrame:
    """Amostras nas linhas. Centraliza, escala e imputa faltantes pela média (zero)."""
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    return Z.fillna(0)


def pca(X: pd.DataFrame, n: int = 10):
    """X: amostras x features. Devolve (scores, variância explicada, cargas)."""
    from sklearn.decomposition import PCA
    Z = padronizar(X)
    n = int(max(1, min(n, Z.shape[0] - 1, Z.shape[1])))
    m = PCA(n_components=n, random_state=0).fit(Z.to_numpy())
    sc = pd.DataFrame(m.transform(Z.to_numpy()), index=X.index, columns=[f"PC{i + 1}" for i in range(n)])
    cargas = pd.DataFrame(m.components_.T, index=X.columns, columns=sc.columns)
    return sc, m.explained_variance_ratio_, cargas


def embutir_2d(X: np.ndarray, semente: int = 0):
    """UMAP quando instalado e possível; senão t-SNE. Devolve (coords, nome do método)."""
    n = X.shape[0]
    if n < 6:
        return None, None
    try:
        from umap import UMAP
        viz = max(2, min(15, n - 1))
        return UMAP(n_neighbors=viz, min_dist=0.3, random_state=semente).fit_transform(X), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        perp = max(2, min(30, (n - 1) // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=semente, init="pca").fit_transform(X), "t-SNE"


def rv_coef(A: np.ndarray, B: np.ndarray) -> float:
    """Coeficiente RV entre duas matrizes de amostras (mesma ordem de linhas)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    Sa, Sb = A @ A.T, B @ B.T
    num = np.trace(Sa @ Sb)
    den = np.sqrt(np.trace(Sa @ Sa) * np.trace(Sb @ Sb))
    return float(num / den) if den > 0 else np.nan


def estadio_grupo(v) -> str:
    """Agrupa 'Stage IIIA' -> 'III'. Devolve 'NA' quando não há estádio."""
    if not isinstance(v, str):
        return "NA"
    s = v.upper().replace("STAGE", "").strip()
    for g in ("IV", "III", "II", "I"):
        if s.startswith(g):
            return g
    if s.startswith("0") or s.startswith("X"):
        return "NA"
    return "NA"


_N_TABELA = itertools.count(1)


def tabela_html(df: pd.DataFrame, max_linhas: int = 30, casas: int = 3) -> str:
    """Tabela HTML com números alinhados à direita e colunas de texto à esquerda (quebrando linha)."""
    d = df.head(max_linhas).copy()
    if not isinstance(d.index, pd.RangeIndex) or d.index.name is not None:
        d = d.reset_index().rename(columns={"index": ""})
    texto = [i for i, c in enumerate(d.columns) if not pd.api.types.is_numeric_dtype(d.iloc[:, i])]
    for i in range(d.shape[1]):
        if pd.api.types.is_float_dtype(d.iloc[:, i]):
            d.isetitem(i, d.iloc[:, i].map(
                lambda x: "" if pd.isna(x) else (f"{x:.{casas}g}" if abs(x) < 1e-3 or abs(x) >= 1e5
                                                 else f"{x:,.{casas}f}".rstrip("0").rstrip("."))))
    tid = f"tab{next(_N_TABELA)}"
    estilo = ""
    if texto:
        sel = ",".join(f"#{tid} td:nth-child({i + 1}),#{tid} th:nth-child({i + 1})" for i in texto)
        estilo = f"<style>{sel}{{text-align:left;white-space:normal}}</style>"
    return estilo + d.to_html(classes="tab", border=0, escape=True, index=False, table_id=tid)


def barras_categoria(ax, serie: pd.Series, titulo: str, max_cat: int = 12, cor=AZUL):
    vc = serie.fillna("(ausente)").astype(str).value_counts()
    if len(vc) > max_cat:
        vc = pd.concat([vc.iloc[:max_cat - 1], pd.Series({"(outros)": vc.iloc[max_cat - 1:].sum()})])
    vc = vc.iloc[::-1]
    cores = [CINZA if k in ("(ausente)", "(outros)") else cor for k in vc.index]
    ax.barh([k[:38] for k in vc.index], vc.values, color=cores)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("pacientes")
