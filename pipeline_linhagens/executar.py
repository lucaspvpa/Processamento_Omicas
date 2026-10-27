"""Ponto de entrada do levantamento de linhagens (DepMap e Cell Model Passports).

Exemplos (rodar da pasta que contém pipeline_linhagens e pipeline_tcga):
    python -m pipeline_linhagens --saida "C:/Users/lucas/OneDrive/Desktop/Doutorado/Datasets"
    python -m pipeline_linhagens --bancos depmap --omicas expressao crispr
    python -m pipeline_linhagens --so-listar        # só grava a lista de arquivos de cada banco

Para cada banco: baixa a anotação dos modelos e um arquivo por ômica para uma pasta temporária, descobre o
formato de cada tabela, gera figuras, tabelas e um relatório HTML por ômica em
<saida>/resultados_linhagens/<BANCO> e apaga os dados brutos no fim (a menos que --manter-dados).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from pipeline_tcga.graficos import tabela_html, tema
from pipeline_tcga.relatorio import Relatorio, fmt

from . import analise, fontes
from .baixar import baixar, nome_seguro, sessao, tabelas_do_arquivo
from .inventario import Resolvedor, carregar_longo, carregar_matriz, detectar, preparar

COLUNAS_DEPMAP = ["OncotreeLineage", "OncotreePrimaryDisease", "PrimaryOrMetastasis", "Sex", "AgeCategory",
                  "PatientRace", "GrowthPattern", "SampleCollectionSite", "ModelType", "PatientTreatmentStatus"]
COLUNAS_CMP = ["tissue", "cancer_type", "model_type", "tissue_status", "sample_site", "gender", "ethnicity",
               "growth_properties", "msi_status", "smoking_status"]


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


def ler_csv(caminho: str, **kw) -> pd.DataFrame:
    sep = "\t" if re.search(r"\.(tsv|txt)(\.gz)?$", caminho) else ","
    return pd.read_csv(caminho, sep=sep, low_memory=False, **kw)


# ---------------------------------------------------------------- anotação de cada banco

def anot_depmap(arqs: dict) -> tuple[pd.DataFrame, pd.DataFrame, Resolvedor]:
    bruto = ler_csv(arqs["modelo"])
    idc = "ModelID" if "ModelID" in bruto.columns else "DepMap_ID"
    bruto = bruto.drop_duplicates(idc).set_index(idc)
    res = Resolvedor()
    res.adicionar(bruto.index, bruto.index)
    for c in ["CCLEName", "CCLE_Name", "StrippedCellLineName", "CellLineName", "RRID", "SangerModelID",
              "COSMICID", "PublicComments_Alias", "cell_line_name"]:
        if c in bruto.columns:
            res.adicionar(bruto[c], bruto.index)
    # Mapas de condição e de perfil (ModelConditionID / ProfileID -> ModelID)
    if arqs.get("condicao"):
        mc = ler_csv(arqs["condicao"])
        if {"ModelConditionID", "ModelID"} <= set(mc.columns):
            res.adicionar(mc["ModelConditionID"], mc["ModelID"])
    if arqs.get("perfis"):
        pr = ler_csv(arqs["perfis"])
        alvo = "ModelID" if "ModelID" in pr.columns else ("ModelConditionID" if "ModelConditionID" in pr.columns else None)
        if alvo:
            dest = pr[alvo] if alvo == "ModelID" else pr[alvo].map(res)
            for c in ["ProfileID", "SequencingID"]:
                if c in pr.columns:
                    res.adicionar(pr[c], dest)
    anot = pd.DataFrame(index=bruto.index)
    anot["linhagem"] = bruto.get("OncotreeLineage", bruto.get("lineage"))
    anot["doenca"] = bruto.get("OncotreePrimaryDisease", bruto.get("primary_disease"))
    anot["subtipo"] = bruto.get("OncotreeSubtype")
    cod = bruto.get("OncotreeCode", pd.Series(index=bruto.index, dtype=object))
    anot["tcga"] = cod.map(fontes.ONCOTREE_TCGA)
    anot["sanger"] = bruto.get("SangerModelID")
    return anot, bruto, res


def anot_cmp(arqs: dict, anot_dm: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, Resolvedor]:
    bruto = ler_csv(arqs["modelo"])
    bruto = bruto.drop_duplicates("model_id").set_index("model_id")
    res = Resolvedor()
    res.adicionar(bruto.index, bruto.index)
    for c in ["model_name", "synonyms", "BROAD_ID", "RRID", "COSMIC_ID", "sample_id", "CCLE_ID"]:
        if c in bruto.columns:
            res.adicionar(bruto[c], bruto.index)
    anot = pd.DataFrame(index=bruto.index)
    anot["linhagem"] = bruto.get("tissue")
    anot["doenca"] = bruto.get("cancer_type")
    anot["subtipo"] = bruto.get("cancer_type_detail")
    anot["broad"] = bruto.get("BROAD_ID")
    anot["tcga"] = np.nan
    col_tcga = next((c for c in bruto.columns if "tcga" in c.lower()), None)
    if col_tcga:
        anot["tcga"] = bruto[col_tcga]
    elif anot_dm is not None and anot["broad"].notna().any():
        anot["tcga"] = anot["broad"].map(anot_dm["tcga"])
    return anot, bruto, res


# ---------------------------------------------------------------- um banco

def processar_banco(banco: str, args, s, log, anot_dm=None) -> dict:
    nome = {"depmap": "DepMap", "cmp": "Cell Model Passports"}[banco]
    pasta = os.path.join(args.saida, "resultados_linhagens", banco.upper() if banco == "depmap" else "PASSPORTS")
    brutos = os.path.join(args.tmp, banco)
    os.makedirs(pasta, exist_ok=True)
    os.makedirs(brutos, exist_ok=True)

    log(f"[{nome}] lista de arquivos")
    if banco == "depmap":
        lista = fontes.lista_depmap(s)
        lista.drop(columns=[c for c in lista.columns if c == "url"]).to_csv(
            os.path.join(pasta, "arquivos_disponiveis.csv"), index=False)
        esc = fontes.escolher_depmap(lista)
    else:
        lista = fontes.lista_cmp(s)
        lista.to_csv(os.path.join(pasta, "arquivos_disponiveis.csv"), index=False)
        esc = fontes.escolher_cmp(lista)
    esc.drop(columns=["url"]).to_csv(os.path.join(pasta, "arquivos_escolhidos.csv"), index=False)
    log(f"[{nome}] {esc['filename'].notna().sum()} de {len(esc)} alvos encontrados")
    for _, r in esc[esc["filename"].isna()].iterrows():
        log(f"    sem arquivo para {r['chave']} ({r['rotulo']})")
    if args.so_listar:
        return {}

    arqs = {}
    mapas = fontes.MAPAS if banco == "depmap" else {"modelo"}
    for _, r in esc[esc["chave"].isin(mapas) & esc["url"].notna()].iterrows():
        log(f"[{nome}] baixando {r['filename']}")
        arqs[r["chave"]] = baixar(s, r["url"], os.path.join(brutos, nome_seguro(r["filename"])), log)
    if "modelo" not in arqs:
        raise RuntimeError(f"{nome}: não achei a tabela de modelos; veja arquivos_disponiveis.csv")
    anot, bruto, res = anot_depmap(arqs) if banco == "depmap" else anot_cmp(arqs, anot_dm)
    anot.to_csv(os.path.join(pasta, "anotacao_linhagens.csv"))
    log(f"[{nome}] {len(anot)} modelos anotados, {len(res.mapa)} identificadores conhecidos")

    omicas = esc[~esc["chave"].isin(mapas) & esc["url"].notna()]
    if args.omicas:
        omicas = omicas[omicas["chave"].isin(args.omicas)]

    disp, rotulos, inventario, linhas_idx, resumo, matrizes = {}, {}, [], [], {}, {}
    for i, (_, r) in enumerate(omicas.iterrows(), start=1):
        chave, rotulo = r["chave"], r["rotulo"]
        sub = f"{i:02d}_{chave}"
        pasta_o = os.path.join(pasta, sub)
        os.makedirs(pasta_o, exist_ok=True)
        rel = analise.Relatorio(f"{nome}: {rotulo}", f"arquivo {r['filename']}" +
                                (f" · versão {r['release']}" if r.get("release") else ""))
        linha = {"chave": chave, "rotulo": rotulo, "arquivo": r["filename"], "versao": r.get("release"),
                 "pagina": f"{sub}/index.html"}
        leitura = fontes.CMP_LEITURA.get(chave) if banco == "cmp" else None
        preparado = os.path.join(brutos, f"{chave}_preparado.csv")
        try:
            if str(r["filename"]).lower().endswith(".xlsx") and not (leitura and leitura.get("modo")):
                raise RuntimeError("planilha Excel: fora do escopo da leitura automática")
            log(f"[{nome}] {rotulo}: baixando {r['filename']}")
            cam = baixar(s, r["url"], os.path.join(brutos, nome_seguro(r["filename"])), log)
            arquivos = tabelas_do_arquivo(cam)
            if leitura:
                arquivos = preparar(arquivos, leitura, preparado, log)
                if leitura.get("fonte"):
                    rel.nota(f"Só dados gerados pelo {leitura['fonte']} (o arquivo também traz dados de outras "
                             "fontes, como o Broad).")
            tabelas = []
            for nm, fonte in arquivos:
                try:
                    info = detectar(nm, fonte, res)
                except Exception as e:
                    log(f"    {nm}: não consegui ler ({e})")
                    continue
                if info is None:
                    inventario.append({"ômica": rotulo, "tabela": nm, "formato": "sem identificador de linhagem"})
                    continue
                inventario.append({"ômica": rotulo, "tabela": nm, "formato": info["formato"],
                                   "linhas do arquivo": info["linhas"], "identificadores": info["ids_brutos"],
                                   UNID: len(info["modelos"]), "variáveis": info["variaveis"]})
                tabelas.append((nm, fonte, info))
                log(f"    {nm}: {info['formato']}, {len(info['modelos'])} linhagens, {info['variaveis']} variáveis")
            if not tabelas:
                raise RuntimeError("nenhuma tabela do arquivo tem identificadores de linhagem reconhecíveis")
            # Tabela principal: mais linhagens; em empate, TPM antes de contagens
            nm, fonte, info = max(tabelas, key=lambda t: (len(t[2]["modelos"]), "tpm" in t[0].lower()))
            rel.fonte(f"{nome}, {r['filename']}" + (f" / {nm}" if nm != r["filename"] else ""))
            rel.nota(f"Formato detectado: <b>{info['formato']}</b>. {len(info['modelos'])} de "
                     f"{info['ids_brutos']} identificadores casaram com a tabela de modelos do {nome}.")
            disp[chave] = pd.Series(True, index=sorted(info["modelos"]))
            rotulos[chave] = rotulo
            if info["orientacao"] == "longo":
                df = carregar_longo(fonte, info, res)
                out = analise.analisar_longo(df, anot, rotulo, pasta_o, rel)
            else:
                X = carregar_matriz(nm, fonte, info, res)
                out = analise.analisar_matriz(X, anot, chave, rotulo, pasta_o, rel)
                if chave == "expressao":
                    matrizes[chave] = X
            resumo[chave] = out
            linha.update({"formato": info["formato"], "linhagens": len(info["modelos"]),
                          "variaveis": out.get("variaveis", info["variaveis"])})
        except Exception as e:
            log(f"[{nome}] {rotulo}: ERRO {e}")
            log(traceback.format_exc())
            rel.nota(f"Falhou: {e}")
            linha.update({"formato": f"erro: {e}"})
            resumo[chave] = {"erro": str(e)}
        rel.salvar(os.path.join(pasta_o, "index.html"), nav=analise.nav(2))
        linhas_idx.append(linha)
        if not args.manter_dados:
            for f in os.listdir(brutos):
                if f in (nome_seguro(str(r["filename"])), os.path.basename(preparado)):
                    try:
                        os.remove(os.path.join(brutos, f))
                    except OSError:
                        pass

    inv = pd.DataFrame(inventario)
    inv.to_csv(os.path.join(pasta, "inventario_tabelas.csv"), index=False)

    # Página do banco
    rel = Relatorio(f"{nome}: panorama das linhagens", "Inventário das ômicas, sobreposição e anotações")
    tab = pd.DataFrame(disp).reindex(anot.index).fillna(False).astype(bool) if disp else pd.DataFrame(index=anot.index)
    tab = tab.loc[tab.any(axis=1)] if tab.shape[1] else tab
    rel.kpi("modelos anotados", len(anot))
    rel.kpi("com pelo menos uma ômica", len(tab))
    rel.kpi("ômicas", len(disp))
    if "linhagem" in anot:
        rel.kpi("tecidos", int(anot["linhagem"].nunique()))
    rel.secao("Ômicas disponíveis")
    rel.tabela(analise.tabela_links(linhas_idx))
    if len(inv):
        rel.tabela(tabela_html(inv, max_linhas=80), "Todas as tabelas lidas")
    res_disp = analise.disponibilidade(tab, rotulos, anot, nome, pasta, rel) if tab.shape[1] else {}
    res_anot = analise.anotacoes(anot, bruto, pasta, rel, COLUNAS_DEPMAP if banco == "depmap" else COLUNAS_CMP)
    rel.salvar(os.path.join(pasta, "index.html"), nav=analise.nav(1))
    resumo_banco = {"banco": nome, "gerado_em": dt.datetime.now().isoformat(timespec="seconds"),
                    "arquivos": esc.drop(columns=["url"]).to_dict(orient="records"),
                    "disponibilidade": res_disp, "omicas": resumo,
                    "anotacoes": {"modelos": len(anot), **res_anot}}
    with open(os.path.join(pasta, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo_banco, f, ensure_ascii=False, indent=1, default=str)
    return {"anot": anot, "disp": tab, "rotulos": rotulos, "matrizes": matrizes, "resumo": resumo_banco,
            "linhas": linhas_idx}


UNID = "linhagens"


# ---------------------------------------------------------------- comparação entre os bancos

def comparar(dm: dict, cmp: dict, saida: str, log):
    pasta = os.path.join(saida, "resultados_linhagens", "COMPARACAO")
    os.makedirs(pasta, exist_ok=True)
    rel = Relatorio("DepMap × Cell Model Passports", "Linhagens em comum e ômicas disponíveis nos dois bancos")
    pares = set()
    a_dm, a_cm = dm["anot"], cmp["anot"]
    if "sanger" in a_dm:
        pares |= {(d, s) for d, s in a_dm["sanger"].dropna().items() if s in a_cm.index}
    if "broad" in a_cm:
        pares |= {(b, s) for s, b in a_cm["broad"].dropna().items() if b in a_dm.index}
    p = pd.DataFrame(sorted(pares), columns=["DepMap", "Passports"]).drop_duplicates("DepMap")
    p.to_csv(os.path.join(pasta, "pares_de_linhagens.csv"), index=False)
    n_dm, n_cm, n_ambos = len(a_dm), len(a_cm), len(p)
    rel.kpi("modelos no DepMap", n_dm)
    rel.kpi("modelos no Passports", n_cm)
    rel.kpi("em comum", n_ambos)
    rel.texto("A correspondência usa o SangerModelID do DepMap e o BROAD_ID do Passports. Linhagens em comum "
              "permitem validar um banco com o outro e somar ômicas que só um deles tem (por exemplo RNAi e "
              "metilação no DepMap, CRISPR do Project Score e farmacologia GDSC no Sanger).")

    linhas = []
    for chave in sorted(set(dm["rotulos"]) | set(cmp["rotulos"])):
        tem_dm = set(dm["disp"].index[dm["disp"][chave]]) if chave in dm["disp"] else set()
        tem_cm = set(cmp["disp"].index[cmp["disp"][chave]]) if chave in cmp["disp"] else set()
        pd_ = p[p["DepMap"].isin(tem_dm) & p["Passports"].isin(tem_cm)]
        linhas.append({"ômica": dm["rotulos"].get(chave) or cmp["rotulos"].get(chave), "DepMap": len(tem_dm),
                       "Passports": len(tem_cm), "nos dois (mesma linhagem)": len(pd_)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(os.path.join(pasta, "omicas_nos_dois_bancos.csv"), index=False)
    rel.secao("Ômicas nos dois bancos")
    rel.tabela(tabela_html(tab))

    res = {"depmap": n_dm, "passports": n_cm, "em_comum": n_ambos, "por_omica": linhas}
    Xa, Xb = dm["matrizes"].get("expressao"), cmp["matrizes"].get("expressao")
    if Xa is not None and Xb is not None and n_ambos:
        import matplotlib.pyplot as plt
        from pipeline_tcga.graficos import AZUL, salvar
        Xa = Xa.copy()
        Xa.columns = [re.sub(r"\s*\(.*\)$", "", str(c)) for c in Xa.columns]
        Xa = Xa.loc[:, ~Xa.columns.duplicated()]
        Xb = Xb.loc[:, ~Xb.columns.duplicated()]
        if Xb.to_numpy()[np.isfinite(Xb.to_numpy())].max() > 50:  # TPM em escala linear
            Xb = np.log2(Xb.clip(lower=0) + 1)
        genes = Xa.columns.intersection(Xb.columns)
        pp = p[p["DepMap"].isin(Xa.index) & p["Passports"].isin(Xb.index)]
        if len(genes) > 500 and len(pp) > 5:
            A = Xa.loc[pp["DepMap"], genes].to_numpy()
            B = Xb.loc[pp["Passports"], genes].to_numpy()
            ra = pd.DataFrame(A).rank(axis=1).to_numpy()
            rb = pd.DataFrame(B).rank(axis=1).to_numpy()
            ra = ra - np.nanmean(ra, axis=1, keepdims=True)
            rb = rb - np.nanmean(rb, axis=1, keepdims=True)
            rho = np.nansum(ra * rb, axis=1) / np.sqrt(np.nansum(ra ** 2, axis=1) * np.nansum(rb ** 2, axis=1))
            # Linha de base: pares embaralhados
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(rb))
            rho0 = np.nansum(ra * rb[perm], axis=1) / np.sqrt(np.nansum(ra ** 2, axis=1) * np.nansum(rb[perm] ** 2, axis=1))
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.hist(rho0, bins=50, color="#d0d4dc", label="pares embaralhados")
            ax.hist(rho, bins=50, color=AZUL, alpha=0.85, label="mesma linhagem")
            ax.set_xlabel("Spearman entre perfis de expressão (DepMap × Passports)")
            ax.legend()
            rel.secao("Reprodutibilidade da expressão entre os bancos")
            rel.figura(salvar(fig, pasta, "expressao_concordancia"),
                       f"Correlação de Spearman, em {len(genes)} genes, entre o RNA-seq da mesma linhagem nos dois "
                       f"bancos ({len(pp)} linhagens). Mediana {np.nanmedian(rho):.3f}; pares embaralhados, "
                       f"{np.nanmedian(rho0):.3f}. Linhagens com correlação baixa podem ser trocas de identidade "
                       "ou derivas de cultivo.", pasta)
            out = pd.DataFrame({"DepMap": pp["DepMap"].values, "Passports": pp["Passports"].values, "rho": rho})
            out.sort_values("rho").to_csv(os.path.join(pasta, "expressao_concordancia.csv"), index=False)
            res["expressao_rho_mediana"] = float(np.nanmedian(rho))
            res["expressao_rho_embaralhado"] = float(np.nanmedian(rho0))
    rel.salvar(os.path.join(pasta, "index.html"), nav=analise.nav(1))
    with open(os.path.join(pasta, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    log(f"[comparação] {n_ambos} linhagens em comum")
    return res


def indice(saida: str, resultados: dict):
    rel = Relatorio("Levantamento de linhagens celulares", "DepMap (Broad) e Cell Model Passports (Sanger)")
    for banco, sub in [("depmap", "DEPMAP"), ("cmp", "PASSPORTS")]:
        r = resultados.get(banco)
        if not r:
            continue
        rs = r["resumo"]
        disp = rs.get("disponibilidade", {})
        rel.secao(rs["banco"])
        rel.texto(f'{fmt(rs["anotacoes"]["modelos"], 0)} modelos anotados; {fmt(disp.get("linhagens", 0), 0)} com '
                  f'alguma ômica; {fmt(disp.get("todas", 0), 0)} com todas as ômicas lidas. '
                  f'<a href="{sub}/index.html">Abrir panorama</a>.')
        rel.tabela(analise.tabela_links([{**l, "pagina": f"{sub}/{l['pagina']}"} for l in r["linhas"]]))
    if "comparacao" in resultados:
        c = resultados["comparacao"]
        rel.secao("Comparação")
        rel.texto(f'{fmt(c["em_comum"], 0)} linhagens estão nos dois bancos. '
                  '<a href="COMPARACAO/index.html">Abrir comparação</a>.')
    rel.salvar(os.path.join(saida, "resultados_linhagens", "index.html"), nav="")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Levantamento descritivo do DepMap e do Cell Model Passports")
    ap.add_argument("--saida", default=os.getcwd(), help="pasta onde criar resultados_linhagens/")
    ap.add_argument("--bancos", nargs="+", default=["depmap", "cmp"], choices=["depmap", "cmp"])
    ap.add_argument("--omicas", nargs="+", help="só estas chaves (expressao, cnv, mutacao, crispr, ...)")
    ap.add_argument("--tmp", default=os.path.join(tempfile.gettempdir(), "linhagens_brutos"))
    ap.add_argument("--manter-dados", action="store_true", help="não apaga os arquivos baixados")
    ap.add_argument("--so-listar", action="store_true", help="só grava a lista de arquivos de cada banco")
    args = ap.parse_args(argv)

    tema()
    log = Log(os.path.join(args.saida, "logs", f"linhagens_{dt.datetime.now():%Y%m%d_%H%M%S}.log"))
    s = sessao()
    resultados = {}
    for banco in args.bancos:
        try:
            anot_dm = resultados.get("depmap", {}).get("anot")
            resultados[banco] = processar_banco(banco, args, s, log, anot_dm=anot_dm)
        except Exception as e:
            log(f"[{banco}] falhou: {e}")
            log(traceback.format_exc())
    if args.so_listar:
        return 0
    if resultados.get("depmap") and resultados.get("cmp"):
        try:
            resultados["comparacao"] = comparar(resultados["depmap"], resultados["cmp"], args.saida, log)
        except Exception as e:
            log(f"[comparação] falhou: {e}")
            log(traceback.format_exc())
    indice(args.saida, resultados)
    if not args.manter_dados:
        shutil.rmtree(args.tmp, ignore_errors=True)
    log("fim")
    return 0
