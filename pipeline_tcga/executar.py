"""Ponto de entrada do pipeline.

Exemplos (rodar da pasta que contém a pasta pipeline_tcga):
    python -m pipeline_tcga --canceres CHOL
    python -m pipeline_tcga --canceres CHOL BRCA --saida "C:/Users/lucas/OneDrive/Desktop/Doutorado/Datasets"
    python -m pipeline_tcga --todos --pular-existentes
    python -m pipeline_tcga --todos --so-disponibilidade

Para cada câncer: baixa os dados para uma pasta temporária, gera figuras, tabelas e relatórios em
<saida>/resultados/<CANCER>, guarda um paciente de exemplo por combinação de ômicas em
<saida>/exemplos/<CANCER> e apaga os dados brutos. No fim, refaz a comparação pan-câncer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from . import config, exemplos, pancan
from .analises import Contexto, clinico, cnv, comparativo, contagens, continuas, disponibilidade, mutacao
from .dados import Coletor
from .fontes import Fontes
from .graficos import tabela_html, tema
from .relatorio import Relatorio, fmt

DESCRICAO = {
    "clinico": "Completude, estádio, metadados e sobrevida",
    "mutacao": "Carga mutacional, genes mais mutados, espectro e oncoplot",
    "cnv": "Ganhos e perdas no genoma, fração alterada e genes recorrentes",
    "rna": "Distribuição, PCA/UMAP, agrupamento, tumor x normal e clínica",
    "rna_contagem": "Bibliotecas, superdispersão (binomial negativa), zeros e exploração em log-CPM",
    "mirna": "Distribuição, PCA/UMAP, agrupamento, tumor x normal e clínica",
    "mirna_contagem": "Bibliotecas, superdispersão, zeros e exploração em log-CPM",
    "metilacao": "Distribuição de beta, metilação global e exploração dos CpGs variáveis",
    "rppa": "Proteínas e fosfoproteínas: PCA, agrupamento e clínica",
    "atac": "Picos de cromatina acessível: PCA, agrupamento e clínica",
    "cptac": "Proteoma por espectrometria de massa: faltantes, PCA e clínica",
}
ORDEM_CARGA = ["mutacao", "rna", "rna_contagem", "mirna", "mirna_contagem", "cnv", "metilacao", "rppa", "atac", "cptac"]


class Log:
    def __init__(self, arquivo: str):
        os.makedirs(os.path.dirname(arquivo), exist_ok=True)
        self.f = open(arquivo, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str):
        linha = f"[{dt.datetime.now():%H:%M:%S} +{(time.time() - self.t0) / 60:5.1f}min] {msg}"
        print(linha, flush=True)
        self.f.write(linha + "\n")
        self.f.flush()


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, pd.Series):
        return o.to_dict()
    return str(o)


def _tamanho(pasta: str) -> int:
    total = 0
    for raiz, _, arquivos in os.walk(pasta):
        for a in arquivos:
            try:
                total += os.path.getsize(os.path.join(raiz, a))
            except OSError:
                pass
    return total


def relatorio_indisponivel(ctx, chave: str, om, motivo: str):
    pasta = ctx.pasta_omica(chave)
    rel = Relatorio(f"{ctx.cancer} · {config.ROTULO[chave]}", "Sem dados para analisar nesta execução")
    rel.nota(html.escape(motivo))
    for n_ in (om.notas if om is not None else []):
        rel.nota(n_)
    rel.salvar(os.path.join(pasta, "relatorio.html"), ctx.nav())


def analisar_omica(ctx, chave: str, om, col):
    if chave == "mutacao":
        return mutacao.analisar(ctx, om, col.amostras("mutacao", tumor_e_normal=False))
    if chave == "cnv":
        return cnv.analisar(ctx, om)
    if chave in ("rna_contagem", "mirna_contagem"):
        return contagens.analisar(ctx, om, "rna" if chave == "rna_contagem" else "mirna")
    if chave == "metilacao":
        return continuas.analisar_metilacao(ctx, om)
    return continuas.analisar(ctx, om)


def processar_cancer(cancer: str, args, fontes: Fontes, log, dir_comp: str) -> dict:
    dir_res = os.path.join(args.saida, "resultados", cancer)
    dir_tmp = os.path.join(args.tmp, cancer)
    os.makedirs(dir_res, exist_ok=True)
    t0 = time.time()
    log(f"===== {cancer} =====")
    col = Coletor(fontes, cancer, dir_comp, dir_tmp, log)
    ctx = Contexto(cancer, dir_res, col.clinico_cdr(), log)
    status = {}

    log("Disponibilidade das ômicas")
    col.calcular_disponibilidade()
    disp = disponibilidade.analisar(ctx, col)
    ctx.resumo["disponibilidade"] = disp["res"]
    for _, r in disp["qt"].iterrows():
        log(f"  {r['ômica']:<24} {r['pacientes (tumor)']:>5} pacientes  ({r['fonte']})")

    escolha = exemplos.escolher(disp["tab"], ctx.clin, args.max_exemplos) if not args.so_disponibilidade else pd.DataFrame()
    ex = exemplos.Exemplos(os.path.join(args.saida, "exemplos"), cancer, escolha, log) if len(escolha) else None
    if ex is not None:
        col.exemplos = ex.colunas_amostra(col)
        log(f"  {len(escolha)} combinações de ômicas terão um paciente de exemplo")

    log("Clínico")
    gdc, fonte_gdc = col.clinico_gdc()
    try:
        clinico.analisar(ctx, gdc, fonte_gdc)
        status["clinico"] = "ok"
    except Exception as e:
        status["clinico"] = f"erro: {e}"
        log(traceback.format_exc())
    if ex is not None:
        ex.clinico(ctx.clin, gdc)

    omicas = {}
    if not args.so_disponibilidade:
        genes_rna = None
        for chave in ORDEM_CARGA:
            if args.omicas and chave not in args.omicas:
                continue
            log(f"{config.ROTULO[chave]}: carregando")
            t1 = time.time()
            om = None
            try:
                om = col.cnv(genes_filtro=genes_rna) if chave == "cnv" else col.carregar(chave)
            except Exception as e:
                log(f"  falhou ao carregar: {e}")
                log(traceback.format_exc())
                status[chave] = f"erro ao baixar: {e}"
                relatorio_indisponivel(ctx, chave, om, f"Falha ao obter os dados: {e}")
                continue
            if om.vazia:
                motivo = col.fonte_disp.get(chave, "indisponível")
                status[chave] = "sem dados" if motivo != "não existe para este câncer" else motivo
                relatorio_indisponivel(ctx, chave, om, f"Sem amostras deste câncer ({motivo}).")
                log(f"  sem dados ({motivo})")
                continue
            if chave == "rna":
                genes_rna = set(om.matriz.index)
            forma = "" if om.matriz is None else f" {om.matriz.shape[0]} x {om.matriz.shape[1]}"
            log(f"  carregado{forma} em {time.time() - t1:.0f}s; analisando")
            try:
                analisar_omica(ctx, chave, om, col)
                status[chave] = "ok"
            except Exception as e:
                status[chave] = f"erro na análise: {e}"
                log(traceback.format_exc())
            if ex is not None:
                try:
                    ex.omica(om)
                except Exception as e:
                    log(f"  exemplos de {chave} falharam: {e}")
            if chave in ("metilacao", "atac"):
                om.extra.pop("exemplos", None)
            omicas[chave] = om

        log("Comparações entre ômicas")
        try:
            comparativo.analisar(ctx, omicas, ctx.resumo)
            status["comparativo"] = "ok"
        except Exception as e:
            status["comparativo"] = f"erro: {e}"
            log(traceback.format_exc())

    if ex is not None:
        ex.leia_me({**col.fonte_disp})
        log(f"  exemplos gravados: {len(ex.salvos)} arquivos, {fmt(_tamanho(ex.raiz) / 1e6)} MB")

    # métricas por amostra (alimentam o pan-câncer)
    if ctx.metricas:
        met = pd.DataFrame({k: pd.Series(v) for k, v in ctx.metricas.items()})
        met.index.name = "amostra"
        met.to_csv(os.path.join(dir_res, "metricas_por_amostra.csv"))

    resumo = {"cancer": cancer, "data": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
              "modo": "disponibilidade" if args.so_disponibilidade else "completo", "status": status,
              "fontes": col.fonte_disp, "notas": col.notas, "minutos": round((time.time() - t0) / 60, 1),
              "exemplos": [] if ex is None else escolha.assign(omicas=escolha["omicas"].map("+".join)).to_dict("records"),
              **ctx.resumo}
    with open(os.path.join(dir_res, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1, default=_json)
    indice_cancer(ctx, disp, status, col, escolha, args)

    tam = _tamanho(dir_tmp)
    if not args.manter_dados:
        shutil.rmtree(dir_tmp, ignore_errors=True)
        log(f"  dados temporários de {cancer} apagados ({fmt(tam / 1e6)} MB)")
    log(f"{cancer} concluído em {(time.time() - t0) / 60:.1f} min; resultados em {dir_res}")
    return resumo


def indice_cancer(ctx, disp, status, col, escolha, args):
    r = disp["res"]
    rel = Relatorio(f"{ctx.cancer} · panorama multiômico", "TCGA PanCanAtlas 2018, contagens brutas do GDC, ATAC-seq e CPTAC")
    rel.kpi("pacientes (TCGA-CDR)", r["pacientes"])
    rel.kpi("ômicas com dados", len(r["omicas_disponiveis"]) - 1)
    rel.kpi("pacientes com todas", r["todas_disponiveis"])
    rel.kpi("combinações distintas", len(r["combinacoes"]))

    linhas = []
    for o in config.OMICAS:
        k = o["chave"]
        st = status.get(k, "não rodado" if args.so_disponibilidade and k != "clinico" else "—")
        pac = r["por_omica"].get(k, 0)
        link = (f'<a href="{o["pasta"]}/relatorio.html">{html.escape(o["rotulo"])}</a>'
                if os.path.exists(os.path.join(ctx.pasta, o["pasta"], "relatorio.html")) else html.escape(o["rotulo"]))
        linhas.append(f"<tr><td>{link}</td><td>{pac}</td><td class='t'>{html.escape(DESCRICAO[k])}</td>"
                      f"<td class='t'>{html.escape(st)}</td></tr>")
    if os.path.exists(os.path.join(ctx.pasta, "20_comparativo", "relatorio.html")):
        linhas.append('<tr><td><a href="20_comparativo/relatorio.html"><b>Comparações entre ômicas</b></a></td><td></td>'
                      "<td class='t'>Concordância entre fontes, efeitos em cis, mRNA x proteína, RV, agrupamento integrado</td>"
                      f"<td class='t'>{html.escape(status.get('comparativo', ''))}</td></tr>")
    rel.secao("Relatórios")
    rel.bruto('<div class="tabwrap"><table class="tab"><thead><tr><th>Relatório</th><th>pacientes</th>'
              "<th class='t'>conteúdo</th><th class='t'>situação</th></tr></thead><tbody>" + "".join(linhas) +
              "</tbody></table></div>")
    rel.bruto('<p><a href="../PANCAN/index.html">→ Comparação pan-câncer</a> · '
              '<a href="../index.html">todos os cânceres</a></p>')

    base = ctx.pasta
    f_upset, f_funil, f_n = disp["figs"]
    rel.secao("Combinações de ômicas por paciente")
    rel.figura(f_upset, "<b>Como ler:</b> cada coluna é uma combinação exata de ômicas (bolinhas pretas ligadas) e a "
               "barra de cima diz quantos pacientes têm exatamente aquela combinação. A barra vermelha é a dos "
               "pacientes com todas as ômicas que existem para este câncer. À esquerda, o total de pacientes de cada "
               "ômica. Conta só amostras tumorais.", base)
    qt = disp["qt"].drop(columns=["chave"]).set_index("ômica")
    rel.tabela(tabela_html(qt, max_linhas=20), "Quantidade de dados por ômica")
    rel.figura(f_funil, "<b>Como ler:</b> quantos pacientes sobram quando você exige cada ômica a mais, da mais comum "
               "para a menos comum. Ajuda a decidir quais ômicas entram no modelo.", base)
    rel.figura(f_n, "<b>Como ler:</b> distribuição do número de ômicas (incluindo o clínico) por paciente.", base)
    rel.tabela(tabela_html(disp["comb"].set_index("combinação"), max_linhas=30), "Combinações (todas as ômicas)")

    if len(escolha):
        e = escolha.assign(omicas=escolha["omicas"].map(lambda x: " + ".join(config.ROTULO[c] for c in x)))
        e = e[["pasta", "paciente", "pacientes", "omicas"]].rename(
            columns={"pasta": "pasta em exemplos/" + ctx.cancer, "pacientes": "pacientes com esta combinação",
                     "omicas": "ômicas"})
        rel.secao("Pacientes de exemplo guardados")
        rel.texto("Os dados brutos foram apagados depois das análises. Para cada combinação abaixo ficou guardado um "
                  "paciente completo (todas as features de cada ômica) em <code>exemplos/" + ctx.cancer + "</code>.")
        rel.tabela(tabela_html(e.set_index("paciente"), max_linhas=40))

    for chave, notas in col.notas.items():
        for n_ in notas:
            rel.nota(f"<b>{html.escape(config.ROTULO.get(chave, chave))}:</b> {n_}")
    rel.salvar(os.path.join(base, "index.html"), '<nav><a href="../index.html">← Todos os cânceres</a></nav>')


def main(argv=None):
    p = argparse.ArgumentParser(prog="pipeline_tcga", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canceres", nargs="+", default=["CHOL"], help="siglas TCGA (padrão: CHOL)")
    p.add_argument("--todos", action="store_true", help="os 33 cânceres do TCGA")
    p.add_argument("--saida", default=os.getcwd(), help="pasta onde ficam resultados/ e exemplos/ (padrão: pasta atual)")
    p.add_argument("--tmp", default=os.path.join(tempfile.gettempdir(), "tcga_pipeline_tmp"),
                   help="pasta temporária para os dados brutos (padrão: temporário do sistema, fora do OneDrive)")
    p.add_argument("--modo", choices=["auto", "api", "arquivo"], default="auto",
                   help="auto: usa a API do Xena/GDC quando responde e baixa arquivos inteiros quando não")
    p.add_argument("--omicas", nargs="+", choices=ORDEM_CARGA, help="rodar só estas ômicas (padrão: todas)")
    p.add_argument("--so-disponibilidade", action="store_true",
                   help="só disponibilidade, UpSet e clínico (rápido; bom para os 33 cânceres)")
    p.add_argument("--max-exemplos", type=int, default=12, help="combinações de ômicas com paciente de exemplo")
    p.add_argument("--pular-existentes", action="store_true", help="pula cânceres que já têm resumo.json completo")
    p.add_argument("--manter-dados", action="store_true", help="não apaga os dados baixados")
    args = p.parse_args(argv)
    for fluxo in (sys.stdout, sys.stderr):  # console do Windows pode não ser UTF-8
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    canceres = config.CANCERES if args.todos else [c.upper() for c in args.canceres]
    invalidos = [c for c in canceres if c not in config.CANCERES]
    if invalidos:
        p.error(f"siglas desconhecidas: {invalidos}. Válidas: {' '.join(config.CANCERES)}")
    args.saida = os.path.abspath(args.saida)
    args.tmp = os.path.abspath(args.tmp)
    dir_comp = os.path.join(args.tmp, "_compartilhado")

    tema()
    log = Log(os.path.join(args.saida, "logs", f"pipeline_{dt.datetime.now():%Y%m%d_%H%M%S}.log"))
    log(f"Cânceres: {' '.join(canceres)}")
    log(f"Resultados em {args.saida}; dados temporários em {args.tmp}")
    fontes = Fontes(log, args.modo)
    for nome in ("pancan", "gdc_api", "atac", "cbio"):
        fontes.acessivel(nome)

    falhas = []
    for c in canceres:
        rj = os.path.join(args.saida, "resultados", c, "resumo.json")
        if args.pular_existentes and os.path.exists(rj):
            with open(rj, encoding="utf-8") as f:
                if json.load(f).get("modo") == "completo" or args.so_disponibilidade:
                    log(f"{c}: já processado, pulando")
                    continue
        try:
            processar_cancer(c, args, fontes, log, dir_comp)
        except KeyboardInterrupt:
            log("Interrompido pelo usuário")
            break
        except Exception as e:
            falhas.append(c)
            log(f"{c} falhou: {e}")
            log(traceback.format_exc())

    log("Pan-câncer")
    try:
        dir_res = os.path.join(args.saida, "resultados")
        pancan.gerar(dir_res, log)
        pancan.indice_geral(dir_res)
    except Exception:
        log(traceback.format_exc())
    if not args.manter_dados:
        tam = _tamanho(args.tmp)
        shutil.rmtree(args.tmp, ignore_errors=True)
        log(f"Pasta temporária apagada ({fmt(tam / 1e6)} MB)")
    log(f"Fim. Abra {os.path.join(args.saida, 'resultados', 'index.html')}")
    if falhas:
        log(f"Cânceres com falha: {' '.join(falhas)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
