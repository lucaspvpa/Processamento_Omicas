"""Uma página por tipo de linhagem (tecido de origem): UpSet, funil e pacotes por grupo de ômicas.

É o equivalente da página de cada câncer no levantamento do TCGA. Os pacotes de cada tipo seguem o funil das
linhagens daquele tipo, então os grupos (e a ordem das ômicas) podem mudar de um tecido para outro.
"""
from __future__ import annotations

import html
import os

import pandas as pd

from pipeline_tcga.graficos import salvar, tabela_html
from pipeline_tcga.relatorio import Relatorio, fmt

from . import analise, pacotes

MIN_LINHAGENS = 3  # tipos com menos linhagens que isso entram só na tabela geral


def _tipo(anot: pd.DataFrame, idx) -> pd.Series:
    return anot.reindex(idx)["linhagem"].fillna("(sem anotação)").astype(str)


def planejar(tab: pd.DataFrame, anot: pd.DataFrame, arquivos: dict, pasta_banco: str) -> list[dict]:
    """Um dicionário por tipo, com os pacotes planejados (ainda não escritos)."""
    tipo = _tipo(anot, tab.index)
    saida = []
    for nome, n in tipo.value_counts().items():
        s = pacotes.slug(nome)
        pasta_t = os.path.join(pasta_banco, "tipos", s)
        sub = tab.loc[tipo == nome]
        sub = sub.loc[:, sub.any()]
        pk = pacotes.planejar(sub, arquivos, os.path.join(pasta_t, "grupos"), pasta_t) if n >= MIN_LINHAGENS else []
        for p in pk:
            p["tipo"] = nome
        saida.append({"nome": nome, "slug": s, "pasta": pasta_t, "linhagens": int(n), "tab": sub, "pacotes": pk})
    return saida


def _nav(banco_sub: str) -> str:
    return (f'<nav><a href="../../../index.html">Início</a><a href="../../index.html">{html.escape(banco_sub)}</a>'
            '<a href="../index.html">Tipos de linhagem</a></nav>')


def pagina_tipo(t: dict, rotulos: dict, anot: pd.DataFrame, nome_banco: str, banco_sub: str):
    pasta, sub = t["pasta"], t["tab"]
    os.makedirs(pasta, exist_ok=True)
    tr = sub.rename(columns=rotulos)
    rel = Relatorio(f"{nome_banco}: {t['nome']}", "Combinações de ômicas e pacotes para download deste tipo de linhagem")
    ordem = [c for c in tr.sum().sort_values(ascending=False).index if tr[c].any()]
    rel.kpi("linhagens com ômica", t["linhagens"])
    rel.kpi("ômicas", len(ordem))
    rel.kpi("com todas", int(tr[ordem].all(axis=1).sum()) if ordem else 0)
    rel.kpi("pacotes", len(t["pacotes"]))

    a = anot.reindex(sub.index)
    a.rename_axis("modelo").join(tr).to_csv(os.path.join(pasta, "linhagens.csv"))
    rel.bruto(pacotes.CSS + "<p>" + pacotes.btn("linhagens.csv", "Lista de linhagens e ômicas (CSV)") + "</p>")

    if t["pacotes"]:
        rel.secao("Baixar os dados por grupo de ômicas", pacotes.TEXTO_GRUPOS)
        rel.bruto(pacotes.tabela_grupos(t["pacotes"]))
    else:
        rel.nota(f"Menos de {MIN_LINHAGENS} linhagens: sem pacotes próprios. Elas estão nos pacotes do banco inteiro.")

    if len(ordem) >= 2 and len(tr) >= 2:
        fig, _ = analise.desenhar_upset(tr[ordem], f"{t['nome']}: linhagens por combinação", unidade=analise.UNIDADE)
        rel.secao("Combinações de ômicas por linhagem")
        rel.figura(salvar(fig, pasta, "upset_combinacoes"), "<b>Como ler:</b> cada coluna é uma combinação exata de "
                   "ômicas e a barra de cima diz quantas linhagens têm exatamente aquela combinação. À esquerda, o "
                   "total de cada ômica.", pasta)
    acum, cur = [], pd.Series(True, index=tr.index)
    for c in ordem:
        cur = cur & tr[c]
        acum.append({"última adicionada": c, analise.UNIDADE: int(cur.sum())})
    if acum:
        rel.tabela(tabela_html(pd.DataFrame(acum)), "Funil de interseção")
    comb = tr.apply(lambda r: " + ".join(c for c in ordem if r[c]) or "nenhuma", axis=1).value_counts()
    rel.tabela(tabela_html(comb.rename_axis("combinação").reset_index(name=analise.UNIDADE), max_linhas=25),
               "Combinações mais comuns")
    if "doenca" in a:
        d = a["doenca"].fillna("(sem anotação)").value_counts().rename_axis("doença").reset_index(name=analise.UNIDADE)
        rel.tabela(tabela_html(d, max_linhas=25), "Doenças deste tipo")
    rel.salvar(os.path.join(pasta, "index.html"), nav=_nav(banco_sub))


def paginas(tipos: list[dict], rotulos: dict, anot: pd.DataFrame, nome_banco: str, pasta_banco: str,
            banco_sub: str) -> str:
    """Páginas de cada tipo e o índice tipos/index.html. Devolve o caminho do índice."""
    linhas = []
    for t in tipos:
        pagina_tipo(t, rotulos, anot, nome_banco, banco_sub)
        tr = t["tab"]
        todas = int(tr.all(axis=1).sum()) if tr.shape[1] else 0
        tam = sum(p.get("bytes") or 0 for p in t["pacotes"])
        linhas.append(f"<tr><td class='t'><a href='{t['slug']}/index.html'>{html.escape(t['nome'])}</a></td>"
                      f"<td>{fmt(t['linhagens'], 0)}</td><td>{tr.shape[1]}</td><td>{fmt(todas, 0)}</td>"
                      f"<td>{len(t['pacotes'])}</td><td>{pacotes.mb(tam)}</td></tr>")
    rel = Relatorio(f"{nome_banco}: tipos de linhagem", "Combinações de ômicas e pacotes por tecido de origem")
    rel.kpi("tipos", len(tipos))
    rel.kpi("com pacotes", sum(1 for t in tipos if t["pacotes"]))
    rel.texto("Clique no tipo para ver o UpSet, o funil e os pacotes para baixar. Cada tipo tem o seu próprio "
              "funil: o grupo 1 reúne as duas ômicas mais comuns naquele tecido e cada grupo seguinte acrescenta "
              f"uma. Tipos com menos de {MIN_LINHAGENS} linhagens não têm pacotes próprios.")
    rel.tabela("<table class='tab'><thead><tr><th class='t'>tipo</th><th>linhagens</th><th>ômicas</th>"
               "<th>com todas</th><th>pacotes</th><th>tamanho</th></tr></thead><tbody>" + "".join(linhas) +
               "</tbody></table>")
    cam = os.path.join(pasta_banco, "tipos", "index.html")
    rel.salvar(cam, nav=f'<nav><a href="../../index.html">Início</a><a href="../index.html">{html.escape(banco_sub)}</a></nav>')
    return cam
