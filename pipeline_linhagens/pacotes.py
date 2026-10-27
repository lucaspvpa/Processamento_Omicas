"""Arquivos para download: cada ômica já processada (IDs do banco) e pacotes por grupo de ômicas.

Grupo de ômicas = linhagens que têm TODAS as ômicas do grupo. Os grupos seguem o funil de interseção: o grupo k
reúne as k ômicas mais comuns, então cada grupo tem uma ômica a mais e menos linhagens que o anterior. Cada
pacote é um .zip com a anotação das linhagens e uma tabela por ômica, só com essas linhagens e na mesma ordem.
"""
from __future__ import annotations

import gzip
import html
import os
import zipfile

import pandas as pd

from pipeline_tcga.relatorio import fmt

# Colunas extras guardadas nas ômicas em formato longo (as que existirem no arquivo)
EXTRAS_LONGO = ["Chrom", "Pos", "Ref", "Alt", "VariantType", "VariantInfo", "ProteinChange", "HGVSp", "HGVSc",
                "LikelyLoF", "Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact", "AF", "DNAChange",
                "protein_mutation", "rna_mutation", "cdna_mutation", "effect", "classification", "vaf", "coding",
                "chromosome", "position", "LeftGene", "RightGene", "FusionName", "SpliceType", "FFPM",
                "gene_symbol_3prime", "gene_symbol_5prime"]


CSS = ("<style>.btn{display:inline-block;font-weight:600;padding:6px 12px;border:1px solid var(--acc);"
       "border-radius:6px;text-decoration:none;margin:2px 6px 2px 0;background:var(--panel);font-size:13px}"
       ".btn.cheio{background:var(--acc);color:var(--bg)}</style>")


def mb(n) -> str:
    if not n:
        return "—"
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{max(n / 1e6, 0.1):.1f} MB"


def salvar_omica(obj: pd.DataFrame, pasta_dados: str, chave: str, longo: bool) -> str:
    """Grava a ômica processada em <pasta_dados>/<chave>.csv.gz. Matriz: linhagens nas linhas."""
    os.makedirs(pasta_dados, exist_ok=True)
    cam = os.path.join(pasta_dados, f"{chave}.csv.gz")
    with gzip.open(cam, "wt", encoding="utf-8", compresslevel=6) as f:
        if longo:
            cols = ["modelo"] + [c for c in obj.columns if c != "modelo"]
            obj[cols].to_csv(f, index=False)
        else:
            obj.index.name = "modelo"
            obj.to_csv(f, float_format="%.4g")
    return cam


def _filtrar(cam: str, linhagens: list[str], longo: bool, destino) -> None:
    """Copia para destino (arquivo texto aberto) só as linhas das linhagens pedidas."""
    alvo = set(linhagens)
    primeiro = True
    partes = []
    for bloco in pd.read_csv(cam, chunksize=200 if not longo else 500_000, dtype={"modelo": str}, low_memory=False):
        b = bloco[bloco["modelo"].isin(alvo)]
        if longo:
            b.to_csv(destino, index=False, header=primeiro)
            primeiro = False
        else:
            partes.append(b)
    if not longo:
        m = pd.concat(partes).drop_duplicates("modelo").set_index("modelo")
        m.reindex([x for x in linhagens if x in m.index]).to_csv(destino, float_format="%.4g")


def grupos_funil(tab: pd.DataFrame) -> list[list[str]]:
    """Lista de grupos (listas de chaves) seguindo o funil; só entra quando o número de linhagens muda."""
    ordem = [c for c in tab.sum().sort_values(ascending=False).index if tab[c].any()]
    grupos, ant = [], None
    for k in range(2, len(ordem) + 1):
        g = ordem[:k]
        n = int(tab[g].all(axis=1).sum())
        if n == 0:
            break
        if ant is not None and n == ant:
            grupos[-1] = g  # mesma contagem: a ômica nova não custa linhagens, entra no grupo anterior
        else:
            grupos.append(g)
        ant = n
    return grupos


def montar(tab: pd.DataFrame, rotulos: dict, arquivos: dict, anot: pd.DataFrame, pasta_banco: str,
           longos: set, log) -> list[dict]:
    """Cria grupos/grupo_XX.zip. Devolve uma linha por grupo para a página."""
    pasta_g = os.path.join(pasta_banco, "dados", "grupos")
    os.makedirs(pasta_g, exist_ok=True)
    saida = []
    for i, g in enumerate(grupos_funil(tab[[c for c in tab.columns if c in arquivos]]), start=1):
        linhas = sorted(tab.index[tab[g].all(axis=1)])
        nome = f"grupo_{i:02d}_{len(g)}omicas_{len(linhas)}linhagens.zip"
        cam = os.path.join(pasta_g, nome)
        log(f"    pacote {nome}: {', '.join(g)}")
        with zipfile.ZipFile(cam, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("linhagens_anotacao.csv", anot.reindex(linhas).rename_axis("modelo").to_csv())
            for c in g:
                with z.open(f"{c}.csv", "w") as f:
                    t = _txt(f)
                    _filtrar(arquivos[c], linhas, c in longos, t)
                    t.flush()
            leia = ("Linhagens com todas estas ômicas: " + ", ".join(rotulos[c] for c in g) + ".\n"
                    "Matrizes: uma linha por linhagem (coluna modelo = ID do banco), mesma ordem em todos os "
                    "arquivos.\nFormato longo (mutações, fusões): uma linha por evento, coluna modelo.\n")
            z.writestr("LEIA-ME.txt", leia)
        saida.append({"grupo": i, "omicas": [rotulos[c] for c in g], "linhagens": len(linhas),
                      "arquivo": os.path.relpath(cam, pasta_banco).replace(os.sep, "/"), "bytes": os.path.getsize(cam)})
    return saida


def _txt(f):
    import io
    return io.TextIOWrapper(f, encoding="utf-8", newline="")


def secao_html(linhas_omicas: list[dict], grupos: list[dict]) -> str:
    """Botões de download: uma linha por ômica e uma linha por grupo de ômicas."""
    def btn(href, txt, cheio=False):
        return (f"<a class='btn{' cheio' if cheio else ''}' href='{html.escape(href)}' download>"
                f"{html.escape(txt)}</a>")
    css = CSS
    corpo_g = "".join(
        f"<tr><td>{g['grupo']}</td><td class='t'>{html.escape(' + '.join(g['omicas']))}</td>"
        f"<td>{fmt(g['linhagens'], 0)}</td><td>{mb(g['bytes'])}</td>"
        f"<td>{btn(g['arquivo'], 'baixar grupo', True)}</td></tr>" for g in grupos)
    corpo_o = ""
    for r in linhas_omicas:
        b = btn(r["dados"], "dados processados", True) if r.get("dados") else ""
        if r.get("url"):
            b += f"<a class='btn' href='{html.escape(r['url'])}'>arquivo original</a>"
        corpo_o += (f"<tr><td class='t'>{html.escape(r['rotulo'])}</td><td>{fmt(r.get('linhagens') or 0, 0)}</td>"
                    f"<td>{mb(r.get('bytes'))}</td><td>{b}</td></tr>")
    return (css + "<h2>Baixar os dados</h2>"
            "<h3>Por grupo de ômicas</h3><p>Cada pacote traz só as linhagens que têm todas as ômicas do grupo, "
            "com a anotação e uma tabela por ômica, prontas para cruzar (mesmo ID e mesma ordem). O grupo 1 tem "
            "as ômicas mais comuns; cada grupo seguinte acrescenta uma ômica e perde linhagens, como no funil.</p>"
            "<div class='tabwrap'><table class='tab'><thead><tr><th>grupo</th><th class='t'>ômicas</th>"
            f"<th>linhagens</th><th>tamanho</th><th></th></tr></thead><tbody>{corpo_g}</tbody></table></div>"
            "<h3>Por ômica</h3><p><i>Dados processados</i>: a tabela lida pelo pipeline, com o ID do banco "
            "(matriz com linhagens nas linhas, ou formato longo para mutações e fusões). <i>Arquivo original</i>: "
            "o arquivo publicado pelo banco, sem alteração.</p>"
            "<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>ômica</th><th>linhagens</th>"
            f"<th>tamanho</th><th></th></tr></thead><tbody>{corpo_o}</tbody></table></div>")
