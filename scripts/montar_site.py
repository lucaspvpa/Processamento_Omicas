"""Monta o site estático (pasta site/) a partir da saída do pipeline_tcga.

Uso, da raiz do repositório:
    python scripts/montar_site.py "C:\\Users\\lucas\\OneDrive\\Desktop\\Doutorado\\Datasets"

A pasta indicada deve conter resultados/ e exemplos/ gerados pelo pipeline. O script copia os relatórios
(HTML e PNG), compacta os exemplos e as tabelas de cada câncer em site/downloads/ e escreve a página inicial.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile

EXT = (".html", ".png", ".csv")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(RAIZ, "site")
MARCA = "<!-- botoes-download -->"

CSS = """
:root{--fg:#1d2433;--muted:#5b6475;--line:#e3e6ec;--bg:#fbfbfd;--acc:#2f6690;--panel:#ffffff;--th:#f3f5f9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--fg:#e6e9ef;--muted:#a3abba;--line:#2e3442;
--bg:#14171d;--acc:#8cb8e0;--panel:#1b1f27;--th:#232835}}
:root[data-theme="dark"]{--fg:#e6e9ef;--muted:#a3abba;--line:#2e3442;--bg:#14171d;--acc:#8cb8e0;--panel:#1b1f27;
--th:#232835}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 6px} h2{font-size:19px;margin:36px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
p{max-width:72ch}.sub{color:var(--muted);margin:0 0 18px}
a{color:var(--acc)}a:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.kpi span{color:var(--muted);font-size:12px}
.btn{display:inline-block;font-weight:600;padding:8px 14px;border:1px solid var(--acc);border-radius:6px;
text-decoration:none;margin:4px 8px 4px 0;background:var(--panel)}
.btn.cheio{background:var(--acc);color:var(--bg)}
.tabwrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px}
table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{background:var(--th)}
td.l{text-align:left}.tag{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.tag.ok{border-color:var(--acc);color:var(--acc)}
figure{margin:16px 0;max-width:760px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
.barra{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:10px 12px;margin:12px 0}
.barra span{color:var(--muted);font-size:13px}
"""


def mb(caminho: str) -> str:
    return f"{os.path.getsize(caminho) / 1e6:.1f} MB".replace(".", ",")


def copiar_resultados(res: str):
    for pasta, _, arquivos in os.walk(res):
        rel = os.path.relpath(pasta, res)
        for a in arquivos:
            if a.endswith(EXT) and not (rel == "." and a == "index.html"):
                destino = os.path.join(SITE, rel, a)
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                shutil.copy2(os.path.join(pasta, a), destino)


def zipar_exemplos(exe: str) -> dict:
    """Um zip por câncer com os dados do paciente de exemplo de cada combinação de ômicas."""
    saida = {}
    if not os.path.isdir(exe):
        return saida
    os.makedirs(os.path.join(SITE, "downloads"), exist_ok=True)
    for c in sorted(os.listdir(exe)):
        base = os.path.join(exe, c)
        if not os.path.isdir(base):
            continue
        z = os.path.join(SITE, "downloads", f"exemplos_{c}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            for pasta, _, arquivos in os.walk(base):
                for a in arquivos:
                    p = os.path.join(pasta, a)
                    f.write(p, os.path.join(f"exemplos_{c}", os.path.relpath(p, base)))
        saida[c] = z
    return saida


def zipar_tabelas(c: str) -> str | None:
    """Todas as tabelas CSV de um câncer num zip só."""
    base = os.path.join(SITE, c)
    csvs = [os.path.join(p, a) for p, _, arqs in os.walk(base) for a in arqs if a.endswith(".csv")]
    if not csvs:
        return None
    z = os.path.join(SITE, "downloads", f"tabelas_{c}.zip")
    os.makedirs(os.path.dirname(z), exist_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in csvs:
            f.write(p, os.path.join(f"tabelas_{c}", os.path.relpath(p, base)))
    return z


def botoes(c: str, ex: dict, tab: dict, prefixo: str) -> str:
    partes = []
    if c in ex:
        partes.append(f'<a class="btn cheio" href="{prefixo}downloads/exemplos_{c}.zip" download>'
                      f'Baixar dados de exemplo ({mb(ex[c])})</a>')
    if tab.get(c):
        partes.append(f'<a class="btn" href="{prefixo}downloads/tabelas_{c}.zip" download>'
                      f'Baixar tabelas ({mb(tab[c])})</a>')
    return "".join(partes)


def injetar_botoes(c: str, ex: dict, tab: dict):
    """Põe a barra de download logo abaixo do título da página do câncer."""
    p = os.path.join(SITE, c, "index.html")
    b = botoes(c, ex, tab, "../")
    if not os.path.exists(p) or not b:
        return
    s = open(p, encoding="utf-8").read()
    if MARCA in s:
        return
    barra = (f'{MARCA}<style>{CSS_BARRA}</style><div class="barra-dl">{b}'
             f'<span>Exemplos: um paciente por combinação de ômicas, com todos os dados brutos dele.</span></div>')
    i = s.find("</h1>")
    s = s[:i + 5] + barra + s[i + 5:] if i >= 0 else s.replace("<main>", "<main>" + barra, 1)
    open(p, "w", encoding="utf-8").write(s)


CSS_BARRA = (".barra-dl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}"
             ".barra-dl .btn{display:inline-block;font-weight:600;padding:8px 14px;border:1px solid var(--acc);"
             "border-radius:6px;text-decoration:none;background:var(--panel)}"
             ".barra-dl .btn.cheio{background:var(--acc);color:var(--bg)}"
             ".barra-dl span{color:var(--muted);font-size:13px}")


def pagina_inicial(ex: dict, tab: dict):
    canceres = sorted(c for c in os.listdir(SITE)
                      if c not in ("PANCAN", "downloads") and os.path.exists(os.path.join(SITE, c, "index.html")))
    linhas, n_pac, n_comp = [], 0, 0
    for c in canceres:
        r = {}
        rj = os.path.join(RES, c, "resumo.json")
        if os.path.exists(rj):
            r = json.load(open(rj, encoding="utf-8"))
        d = r.get("disponibilidade") or {}
        completo = r.get("modo") == "completo"
        n_pac += d.get("pacientes") or 0
        n_comp += completo
        tag = '<span class="tag ok">completa</span>' if completo else '<span class="tag">disponibilidade</span>'
        dl = []
        if c in ex:
            dl.append(f'<a href="downloads/exemplos_{c}.zip" download>exemplos</a>')
        if tab.get(c):
            dl.append(f'<a href="downloads/tabelas_{c}.zip" download>tabelas</a>')
        linhas.append(f'<tr><td><a href="{c}/index.html"><b>{c}</b></a></td><td>{d.get("pacientes", "")}</td>'
                      f'<td>{len(d.get("omicas_disponiveis") or [])}</td><td>{d.get("todas_disponiveis", "")}</td>'
                      f'<td class="l">{tag}</td><td class="l">{" · ".join(dl)}</td></tr>')
    destaque = [c for c in canceres if c in ex]
    upset = ""
    if destaque:
        c = destaque[0]
        upset = (f'<figure><img src="{c}/00_disponibilidade/upset_combinacoes.png" alt="UpSet de {c}"></figure>')
    btns = "".join(f'<a class="btn cheio" href="downloads/exemplos_{c}.zip" download>'
                   f'Baixar exemplos de {c} ({mb(ex[c])})</a>' for c in destaque)
    pagina = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Exploração multiômica TCGA</title><style>{CSS}</style></head><body><main>
<h1>Exploração multiômica TCGA</h1>
<p class="sub">PanCanAtlas 2018, contagens brutas do GDC, ATAC-seq e proteoma CPTAC, organizados por tipo de câncer.</p>
<div class="kpis"><div class="kpi"><b>{len(linhas)}</b><span>cânceres</span></div>
<div class="kpi"><b>{f"{n_pac:,}".replace(",", ".")}</b><span>pacientes (TCGA-CDR)</span></div>
<div class="kpi"><b>{n_comp}</b><span>com análise completa por ômica</span></div></div>
<p><a class="btn" href="PANCAN/index.html">Comparação pan-câncer</a>{btns}</p>
{upset}
<h2>Cânceres</h2>
<p>Análise <b>completa</b> tem um relatório por ômica e o comparativo entre ômicas. Análise de
<b>disponibilidade</b> tem UpSet, quantidade de dados por ômica e dados clínicos.</p>
<div class="tabwrap"><table><thead><tr><th>câncer</th><th>pacientes</th><th>ômicas</th><th>com todas</th>
<th>análise</th><th>downloads</th></tr></thead><tbody>{"".join(linhas)}</tbody></table></div>
<h2>Sobre os dados</h2>
<p>Os dados de exemplo trazem, para cada combinação de ômicas de um câncer, um paciente com todos os seus dados
(mutações, segmentos de CNV, expressão, metilação, proteínas e clínico) e um LEIA-ME com as fontes. Os demais
dados foram apagados depois da análise; as tabelas trazem os resumos usados nas figuras.</p>
<p class="sub">Gerado pelo pipeline_tcga.</p>
</main></body></html>
"""
    open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(pagina)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    origem = sys.argv[1]
    RES = os.path.join(origem, "resultados")
    if not os.path.isdir(RES):
        sys.exit(f"Não achei {RES}")
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE)
    copiar_resultados(RES)
    ex = zipar_exemplos(os.path.join(origem, "exemplos"))
    tab = {c: zipar_tabelas(c) for c in os.listdir(SITE) if os.path.isdir(os.path.join(SITE, c))
           and c not in ("downloads",)}
    for c in tab:
        injetar_botoes(c, ex, tab)
    # as tabelas soltas já estão nos zips; tirá-las deixa o site com metade do tamanho
    for p, _, arqs in os.walk(SITE):
        for a in arqs:
            if a.endswith(".csv"):
                os.remove(os.path.join(p, a))
    pagina_inicial(ex, tab)
    total = sum(os.path.getsize(os.path.join(p, a)) for p, _, arqs in os.walk(SITE) for a in arqs)
    print(f"Site montado em {SITE} ({total / 1e6:.0f} MB, {len(ex)} câncer(es) com exemplos)")
