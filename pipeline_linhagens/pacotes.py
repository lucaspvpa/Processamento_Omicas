"""Arquivos para download: cada ômica já processada (IDs do banco) e pacotes por grupo de ômicas.

Grupo de ômicas = linhagens que têm TODAS as ômicas do grupo. Os grupos seguem o funil de interseção: o grupo k
reúne as k ômicas mais comuns, então cada grupo tem uma ômica a mais e menos linhagens que o anterior. Cada
pacote é um .zip com a anotação das linhagens e uma tabela por ômica, só com essas linhagens e na mesma ordem.
Há pacotes para o banco inteiro e para cada tipo de linhagem (tecido), cada um com o seu próprio funil.

Os pacotes são escritos numa passada por ômica: o arquivo processado é lido em blocos uma vez só e cada bloco
vai para todos os pacotes que pedem aquela ômica. Assim a memória fica baixa mesmo com centenas de pacotes.
"""
from __future__ import annotations

import gzip
import html
import io
import os
import re
import unicodedata
import zipfile

import pandas as pd

from pipeline_tcga.relatorio import fmt

# Colunas extras guardadas nas ômicas em formato longo (as que existirem no arquivo)
EXTRAS_LONGO = ["Chrom", "Pos", "Ref", "Alt", "VariantType", "VariantInfo", "ProteinChange", "HGVSp", "HGVSc",
                "LikelyLoF", "Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact", "AF", "DNAChange",
                "protein_mutation", "rna_mutation", "cdna_mutation", "effect", "classification", "vaf", "coding",
                "chromosome", "position", "LeftGene", "RightGene", "FusionName", "SpliceType", "FFPM",
                "gene_symbol_3prime", "gene_symbol_5prime"]

LOTE_ZIPS = 100  # pacotes abertos ao mesmo tempo numa passada

CSS = ("<style>.btn{display:inline-block;font-weight:600;padding:6px 12px;border:1px solid var(--acc);"
       "border-radius:6px;text-decoration:none;margin:2px 6px 2px 0;background:var(--panel);font-size:13px}"
       ".btn.cheio{background:var(--acc);color:var(--bg)}.url{font-size:12px;color:var(--muted);word-break:break-all}"
       "</style>")


def mb(n) -> str:
    if not n:
        return "—"
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{max(n / 1e6, 0.1):.1f} MB"


def slug(txt: str) -> str:
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "sem_nome"


def btn(href: str, txt: str, cheio: bool = False, baixar: bool = True) -> str:
    return (f"<a class='btn{' cheio' if cheio else ''}' href='{html.escape(href)}'{' download' if baixar else ''}>"
            f"{html.escape(txt)}</a>")


def salvar_omica(obj: pd.DataFrame, pasta_dados: str, chave: str, longo: bool) -> str:
    """Grava a ômica processada em <pasta_dados>/<chave>.csv.gz, ordenada pelo ID da linhagem.

    Matriz: linhagens nas linhas. Formato longo: uma linha por evento. A ordenação garante que os pacotes saiam
    com as linhagens na mesma ordem em todas as ômicas.
    """
    os.makedirs(pasta_dados, exist_ok=True)
    cam = os.path.join(pasta_dados, f"{chave}.csv.gz")
    with gzip.open(cam, "wt", encoding="utf-8", compresslevel=6) as f:
        if longo:
            cols = ["modelo"] + [c for c in obj.columns if c != "modelo"]
            obj[cols].sort_values("modelo", kind="stable").to_csv(f, index=False)
        else:
            obj = obj[~obj.index.duplicated()].sort_index()
            obj.index.name = "modelo"
            obj.to_csv(f, float_format="%.4g")
    return cam


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


def planejar(tab: pd.DataFrame, arquivos: dict, pasta_g: str, base_rel: str) -> list[dict]:
    """Um dicionário por pacote: ômicas, linhagens, caminho do zip e caminho relativo a base_rel."""
    t = tab[[c for c in tab.columns if c in arquivos]]
    saida = []
    for i, g in enumerate(grupos_funil(t), start=1):
        linhas = sorted(t.index[t[g].all(axis=1)])
        cam = os.path.join(pasta_g, f"grupo_{i:02d}_{len(g)}omicas_{len(linhas)}linhagens.zip")
        saida.append({"grupo": i, "chaves": g, "linhagens_ids": linhas, "linhagens": len(linhas), "cam": cam,
                      "arquivo": os.path.relpath(cam, base_rel).replace(os.sep, "/")})
    return saida


def _distribuir(cam: str, chave: str, alvos: list[dict], longo: bool):
    """Uma passada pelo arquivo processado da ômica, escrevendo nos zips de todos os alvos."""
    abertos = []
    try:
        for p in alvos:
            z = zipfile.ZipFile(p["cam"], "a", zipfile.ZIP_DEFLATED)
            t = io.TextIOWrapper(z.open(f"{chave}.csv", "w", force_zip64=True), encoding="utf-8", newline="")
            abertos.append((z, t, set(p["linhagens_ids"])))
        primeiro = True
        for bloco in pd.read_csv(cam, chunksize=500_000 if longo else 200, dtype={"modelo": str}, low_memory=False):
            for _, t, alvo in abertos:
                b = bloco[bloco["modelo"].isin(alvo)]
                if longo:
                    b.to_csv(t, index=False, header=primeiro)
                else:
                    b.to_csv(t, index=False, header=primeiro, float_format="%.4g")
            primeiro = False
    finally:
        for z, t, _ in abertos:
            t.close()
            z.close()


def escrever(pacotes: list[dict], rotulos: dict, arquivos: dict, anot: pd.DataFrame, longos: set, log) -> None:
    """Cria todos os zips planejados (anotação + LEIA-ME) e preenche uma ômica por vez."""
    for p in pacotes:
        os.makedirs(os.path.dirname(p["cam"]), exist_ok=True)
        with zipfile.ZipFile(p["cam"], "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("linhagens_anotacao.csv", anot.reindex(p["linhagens_ids"]).rename_axis("modelo").to_csv())
            z.writestr("LEIA-ME.txt", "Linhagens com todas estas ômicas: " + ", ".join(rotulos[c] for c in p["chaves"])
                       + (f" (tipo: {p['tipo']})" if p.get("tipo") else "") + ".\n"
                       "Matrizes: uma linha por linhagem (coluna modelo = ID do banco), linhagens em ordem "
                       "alfabética do ID em todos os arquivos.\n"
                       "Formato longo (mutações, fusões): uma linha por evento, coluna modelo.\n")
    for chave in rotulos:
        alvos = [p for p in pacotes if chave in p["chaves"]]
        if not alvos or chave not in arquivos:
            continue
        log(f"    pacotes: {rotulos[chave]} em {len(alvos)} pacotes")
        for i in range(0, len(alvos), LOTE_ZIPS):
            _distribuir(arquivos[chave], chave, alvos[i:i + LOTE_ZIPS], chave in longos)
    for p in pacotes:
        p["bytes"] = os.path.getsize(p["cam"])
        p["omicas"] = [rotulos[c] for c in p["chaves"]]


def tabela_grupos(grupos: list[dict]) -> str:
    corpo = "".join(
        f"<tr><td>{g['grupo']}</td><td class='t'>{html.escape(' + '.join(g['omicas']))}</td>"
        f"<td>{fmt(g['linhagens'], 0)}</td><td>{mb(g.get('bytes'))}</td>"
        f"<td>{btn(g['arquivo'], 'baixar grupo', True)}</td></tr>" for g in grupos)
    return ("<div class='tabwrap'><table class='tab'><thead><tr><th>grupo</th><th class='t'>ômicas</th>"
            f"<th>linhagens</th><th>tamanho</th><th></th></tr></thead><tbody>{corpo}</tbody></table></div>")


TEXTO_GRUPOS = ("Cada pacote traz só as linhagens que têm todas as ômicas do grupo, com a anotação e uma tabela por "
                "ômica, prontas para cruzar (mesmo ID e mesma ordem). O grupo 1 tem as ômicas mais comuns; cada "
                "grupo seguinte acrescenta uma ômica e perde linhagens, como no funil.")


def secao_html(linhas_omicas: list[dict], grupos: list[dict], tipos_href: str | None = None) -> str:
    """Botões de download: uma linha por grupo de ômicas e uma linha por ômica."""
    corpo_o = ""
    for r in linhas_omicas:
        b = btn(r["dados"], "dados processados", True) if r.get("dados") else ""
        if r.get("url"):
            b += btn(r["url"], "arquivo original", baixar=False)
        corpo_o += (f"<tr><td class='t'>{html.escape(r['rotulo'])}</td><td>{fmt(r.get('linhagens') or 0, 0)}</td>"
                    f"<td>{mb(r.get('bytes'))}</td><td>{b}</td></tr>")
    tipos = (f"<p>{btn(tipos_href, 'Ver os grupos por tipo de linhagem', baixar=False)} mesmo funil e pacotes, "
             "separados por tecido.</p>" if tipos_href else "")
    return (CSS + "<h2>Baixar os dados</h2>" + tipos +
            f"<h3>Por grupo de ômicas (todas as linhagens)</h3><p>{TEXTO_GRUPOS}</p>" + tabela_grupos(grupos) +
            "<h3>Por ômica</h3><p><i>Dados processados</i>: a tabela lida pelo pipeline, com o ID do banco "
            "(matriz com linhagens nas linhas, ou formato longo para mutações e fusões). <i>Arquivo original</i>: "
            "o arquivo publicado pelo banco, sem alteração (os endereços estão em \"De onde vêm os dados\").</p>"
            "<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>ômica</th><th>linhagens</th>"
            f"<th>tamanho</th><th></th></tr></thead><tbody>{corpo_o}</tbody></table></div>")


def secao_fontes(esc: pd.DataFrame, nome: str, paginas_gerais: list[tuple[str, str]]) -> str:
    """Tabela com o endereço de onde cada arquivo foi baixado e a página da versão."""
    corpo = ""
    for _, r in esc.iterrows():
        url = r.get("url")
        if not isinstance(url, str) or not url:
            corpo += (f"<tr><td class='t'>{html.escape(str(r['rotulo']))}</td><td class='t' colspan='3'>"
                      "não encontrado nesta versão</td></tr>")
            continue
        pag = r.get("pagina")
        pag = (f"<a href='{html.escape(pag)}'>{html.escape(str(r.get('release') or 'página'))}</a>"
               if isinstance(pag, str) and pag else html.escape(str(r.get("release") or "—")))
        corpo += (f"<tr><td class='t'>{html.escape(str(r['rotulo']))}</td><td class='t'>{pag}</td>"
                  f"<td class='t'><a href='{html.escape(url)}'>{html.escape(str(r['filename']))}</a>"
                  f"<div class='url'>{html.escape(url)}</div></td></tr>")
    gerais = " · ".join(f"<a href='{html.escape(u)}'>{html.escape(t)}</a>" for t, u in paginas_gerais)
    return (CSS + "<h2>De onde vêm os dados</h2>"
            f"<p>Endereço exato de onde cada arquivo do {html.escape(nome)} foi baixado e a página da versão "
            f"correspondente. Páginas do banco: {gerais}.</p>"
            "<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>arquivo</th><th class='t'>versão</th>"
            f"<th class='t'>link do original</th></tr></thead><tbody>{corpo}</tbody></table></div>")
