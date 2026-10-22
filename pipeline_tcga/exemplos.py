"""Guarda os dados de um paciente de cada combinação de ômicas, para servir de exemplo depois que o resto é apagado.

Estrutura: exemplos/<CANCER>/<NN>_<combinação>/<paciente>/<ômica>.csv(.gz)
"""
from __future__ import annotations

import os

import pandas as pd

from . import config
from .dados import paciente

ABREV = {"clinico": "clin", "mutacao": "mut", "cnv": "cnv", "rna": "rna", "rna_contagem": "rnaCont",
         "mirna": "mir", "mirna_contagem": "mirCont", "metilacao": "met", "rppa": "rppa", "atac": "atac",
         "cptac": "cptac"}
CAMPOS_CLINICOS = ["age_at_initial_pathologic_diagnosis", "gender", "race", "ajcc_pathologic_tumor_stage",
                   "histological_type", "histological_grade", "vital_status", "OS", "OS.time", "PFI", "PFI.time"]


def escolher(tab: pd.DataFrame, clin: pd.DataFrame, max_comb: int = 12) -> pd.DataFrame:
    """Um paciente por combinação de ômicas (as max_comb combinações mais comuns e sempre a mais completa).

    Dentro da combinação, escolhe o paciente com menos campos clínicos faltando.
    """
    cols = [c for c in tab.columns if c != "clinico"]
    chave = tab[cols].apply(lambda r: tuple(c for c in cols if r[c]), axis=1)
    chave = chave[chave.map(len) > 0]
    if chave.empty:
        return pd.DataFrame(columns=["combinacao", "omicas", "pacientes", "paciente"])
    cont = chave.value_counts()
    sel = list(cont.index[:max_comb])
    mais_completa = max(cont.index, key=len)
    if mais_completa not in sel:
        sel.append(mais_completa)
    campos = [c for c in CAMPOS_CLINICOS if c in clin.columns]
    falt = clin.reindex(tab.index)[campos].isna().sum(axis=1) if campos else pd.Series(0, index=tab.index)
    linhas = []
    for i, k in enumerate(sorted(sel, key=lambda k: (-len(k), -cont[k])), 1):
        pacs = chave.index[chave == k]
        p = falt.reindex(pacs).sort_values(kind="stable").index[0]
        linhas.append({"pasta": f"{i:02d}_" + "-".join(ABREV[c] for c in k), "combinacao":
                       " + ".join(config.ROTULO[c] for c in k), "omicas": list(k), "pacientes": int(cont[k]),
                       "paciente": p})
    return pd.DataFrame(linhas)


class Exemplos:
    def __init__(self, raiz: str, cancer: str, escolha: pd.DataFrame, log):
        self.raiz = os.path.join(raiz, cancer)
        self.escolha = escolha
        self.log = log
        self.pacientes = set(escolha["paciente"]) if len(escolha) else set()
        self.salvos = []
        if len(escolha):
            os.makedirs(self.raiz, exist_ok=True)
            escolha.assign(omicas=escolha["omicas"].map("+".join)).to_csv(
                os.path.join(self.raiz, "exemplos.csv"), index=False)

    def _pastas(self, p: str) -> list[str]:
        out = []
        for _, r in self.escolha[self.escolha["paciente"] == p].iterrows():
            d = os.path.join(self.raiz, r["pasta"], p)
            os.makedirs(d, exist_ok=True)
            out.append(d)
        return out

    def _gravar(self, p: str, nome: str, df: pd.DataFrame | None, comprimir: bool = False, index: bool = True):
        if df is None or len(df) == 0:
            return
        nome = nome + (".csv.gz" if comprimir else ".csv")
        for d in self._pastas(p):
            df.to_csv(os.path.join(d, nome), index=index, float_format="%.6g")
            self.salvos.append(os.path.join(d, nome))

    def colunas_amostra(self, col) -> list[str]:
        """Amostras (15 caracteres) dos pacientes de exemplo em todas as ômicas, para o coletor guardar."""
        out = set()
        for df in col.disp.values():
            if len(df):
                out |= set(df.loc[df["paciente"].isin(self.pacientes), "amostra"])
        return sorted(out)

    def clinico(self, clin: pd.DataFrame, gdc: pd.DataFrame):
        for p in self.pacientes:
            partes = []
            if p in clin.index:
                partes.append(clin.loc[[p]].T.rename(columns={p: "TCGA-CDR"}))
            if gdc is not None and len(gdc):
                g = gdc.loc[gdc.index == p]
                if len(g):
                    partes.append(g.iloc[[0]].T.set_axis(["GDC"], axis=1))
            if partes:
                self._gravar(p, "clinico", pd.concat(partes), index=True)

    def omica(self, om):
        if om is None or om.vazia or not self.pacientes:
            return
        nome = config.PASTA[om.chave][3:]
        for p in self.pacientes:
            if om.chave == "mutacao":
                m = om.extra["mutacoes"]
                self._gravar(p, nome, m[m["sample"].str[:12] == p], index=False)
                continue
            if om.chave == "cnv":
                s = om.extra["segmentos"]
                self._gravar(p, "cnv_segmentos", s[s["sampleID"].str[:12] == p], index=False)
            if om.chave in ("metilacao", "atac"):
                ex = om.extra.get("exemplos")
                if ex is not None and len(ex):
                    cols = [c for c in ex.columns if paciente(c) == p]
                    self._gravar(p, nome, ex[cols] if cols else None, comprimir=True)
                continue
            M = om.matriz
            cols = [c for c in M.columns if paciente(c) == p]
            if not cols:
                continue
            X = M[cols]
            if om.chave in ("rna_contagem",) and om.extra.get("genes") is not None:
                X = om.extra["genes"].reindex(X.index).join(X)
            self._gravar(p, nome, X, comprimir=len(X) > 50000)

    def leia_me(self, fontes: dict):
        for _, r in self.escolha.iterrows():
            d = os.path.join(self.raiz, r["pasta"])
            if not os.path.isdir(d):
                continue
            linhas = [f"Combinação: {r['combinacao']}",
                      f"Pacientes do câncer com exatamente esta combinação: {r['pacientes']}",
                      f"Paciente de exemplo: {r['paciente']} (o de menos campos clínicos faltando)", "",
                      "Arquivos e fontes:"]
            for c in ["clinico"] + r["omicas"]:
                linhas.append(f"  {config.PASTA[c][3:]}: {fontes.get(c, '')}")
            linhas += ["", "Colunas = amostras (barcode de 15 caracteres; código 01-09 tumor, 10-19 normal).",
                       "Metilação e ATAC: todos os CpGs/picos da amostra, não só os mais variáveis."]
            with open(os.path.join(d, "LEIA-ME.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(linhas) + "\n")
