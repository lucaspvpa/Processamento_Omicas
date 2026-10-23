"""Coleta dos dados de um câncer: disponibilidade por ômica e carregamento das matrizes.

Cada ômica tenta primeiro a fonte que baixa só as amostras do câncer (API do Xena ou do GDC)
e, se ela não responder, usa a alternativa por arquivo. A fonte usada fica registrada em
`Omica.fonte` e aparece nos relatórios.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .fontes import ErroFonte, Fontes

UUID_RE = re.compile(r"([0-9a-fA-F]{8})[-_]([0-9a-fA-F]{4})[-_]([0-9a-fA-F]{4})[-_]([0-9a-fA-F]{4})[-_]([0-9a-fA-F]{12})")


# ----------------------------------------------------------------- barcodes
def paciente(s: str) -> str:
    return s[:12]


def amostra15(s: str) -> str:
    return s[:15]


def codigo_tipo(s: str) -> str:
    return s[13:15] if len(s) >= 15 else ""


def eh_tumor(s: str) -> bool:
    c = codigo_tipo(s)
    return c.isdigit() and 1 <= int(c) <= 9


def eh_normal(s: str) -> bool:
    c = codigo_tipo(s)
    return c.isdigit() and 10 <= int(c) <= 19


@dataclass
class Omica:
    chave: str
    matriz: pd.DataFrame | None = None      # features x amostras (colunas = barcode de 15 caracteres)
    extra: dict = field(default_factory=dict)
    fonte: str = ""
    notas: list = field(default_factory=list)

    @property
    def vazia(self) -> bool:
        if self.chave == "mutacao":
            return self.extra.get("mutacoes") is None or len(self.extra["mutacoes"]) == 0
        return self.matriz is None or self.matriz.shape[1] == 0


class AcumuladorTopVar:
    """Percorre uma matriz grande em blocos guardando só as features mais variáveis,
    estatísticas de todas as features, histograma por amostra e as colunas de exemplo."""

    def __init__(self, n_top: int, exemplos: list[str], bins: np.ndarray | None = None):
        self.n_top, self.exemplos, self.bins = n_top, set(exemplos), bins
        self.top = None
        self.stats = []
        self.hist = None
        self.cols_ex = []

    def add(self, df: pd.DataFrame):
        v = df.to_numpy(dtype="float32")
        with np.errstate(all="ignore"):
            media = np.nanmean(v, axis=1)
            dp = np.nanstd(v, axis=1)
        falt = np.isnan(v).mean(axis=1)
        self.stats.append(pd.DataFrame({"media": media, "dp": dp, "faltante": falt}, index=df.index))
        if self.bins is not None:
            h = np.stack([np.histogram(v[:, j][~np.isnan(v[:, j])], bins=self.bins)[0]
                          for j in range(v.shape[1])], axis=1)
            self.hist = h if self.hist is None else self.hist + h
            self._cols = list(df.columns)
        ex = [c for c in df.columns if c in self.exemplos]
        if ex:
            self.cols_ex.append(df[ex])
        ok = df[(falt < 0.5) & np.isfinite(dp)]
        cand = ok if self.top is None else pd.concat([self.top, ok])
        if len(cand) > self.n_top:
            cand = cand.loc[cand.std(axis=1).nlargest(self.n_top).index]
        self.top = cand

    def resultado(self):
        stats = pd.concat(self.stats) if self.stats else pd.DataFrame()
        hist = None
        if self.hist is not None:
            centros = (self.bins[:-1] + self.bins[1:]) / 2
            hist = pd.DataFrame(self.hist, index=np.round(centros, 4), columns=self._cols)
        ex = pd.concat(self.cols_ex) if self.cols_ex else pd.DataFrame()
        return self.top, stats, hist, ex


def _ler_tabela_filtrada(caminho: str, col_amostra: str, pacientes: set, **kw) -> pd.DataFrame:
    partes = []
    for bloco in pd.read_csv(caminho, sep="\t", dtype=str, chunksize=400_000, **kw):
        partes.append(bloco[bloco[col_amostra].str[:12].isin(pacientes)])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def _colunas_unicas(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas para barcode de 15 caracteres e remove duplicatas (fica a primeira)."""
    df = df.copy()
    df.columns = [amostra15(c) for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


class Coletor:
    def __init__(self, fontes: Fontes, cancer: str, dir_comp: str, dir_cancer: str, log):
        self.f, self.cancer, self.dir_comp, self.dir_cancer, self.log = fontes, cancer, dir_comp, dir_cancer, log
        os.makedirs(dir_comp, exist_ok=True)
        os.makedirs(dir_cancer, exist_ok=True)
        self._cdr = None
        self.disp: dict[str, pd.DataFrame] = {}
        self.fonte_disp: dict[str, str] = {}
        self.notas: dict[str, list] = {}
        self.gdc_listas: dict[str, pd.DataFrame] = {}
        self.atac_ids: dict[str, str] = {}
        self.exemplos: list[str] = []

    # ------------------------------------------------------------- arquivos
    def arquivo(self, hub: str, chave_ou_nome: str) -> str:
        nome = config.DS.get(chave_ou_nome, chave_ou_nome)
        destino = os.path.join(self.dir_comp, re.sub(r"[^\w.+-]", "_", nome))
        return self.f.baixar(hub, nome, destino)

    # -------------------------------------------------------------- clínico
    def cdr(self) -> pd.DataFrame:
        if self._cdr is None:
            df = pd.read_csv(self.arquivo("pancan", "cdr"), sep="\t", dtype=str)
            self._cdr = df
        return self._cdr

    def clinico_cdr(self) -> pd.DataFrame:
        df = self.cdr()
        df = df[df["cancer type abbreviation"] == self.cancer].copy()
        df["_tumor"] = df["sample"].map(eh_tumor)
        df = df.sort_values("_tumor", ascending=False).drop_duplicates("_PATIENT")
        df = df.drop(columns=["_tumor"]).set_index("_PATIENT")
        df.index.name = "paciente"
        return df

    def pacientes(self) -> set:
        return set(self.clinico_cdr().index)

    def clinico_gdc(self) -> tuple[pd.DataFrame, str]:
        projeto = f"TCGA-{self.cancer}"
        if self.f.acessivel("gdc_api"):
            try:
                casos = self.f.gdc_casos(projeto)
                return _achatar_casos_gdc(casos), "API do GDC (/cases)"
            except Exception as e:
                self.log(f"  clínico GDC pela API falhou: {e}")
        try:
            caminho = self.arquivo("gdc", f"TCGA-{self.cancer}.clinical.tsv")
            df = pd.read_csv(caminho, sep="\t", dtype=str)
            df = df.drop_duplicates("submitter_id").set_index("submitter_id")
            df.index.name = "paciente"
            return df, "GDC hub do UCSC Xena (TCGA-*.clinical.tsv)"
        except Exception as e:
            self.log(f"  clínico GDC indisponível: {e}")
            return pd.DataFrame(), "indisponível"

    # ------------------------------------------------------- disponibilidade
    def _registrar(self, chave: str, ids: list[str], fonte: str, mapa_paciente: dict | None = None):
        pac = self.pacientes()
        linhas = []
        for i in ids:
            p = (mapa_paciente or {}).get(i) or (paciente(i) if i.startswith("TCGA-") else None)
            if p is None or p not in pac:
                continue
            a = amostra15(i) if i.startswith("TCGA-") else (mapa_paciente or {}).get("_amostra_" + i, p + "-01")
            linhas.append({"id_original": i, "amostra": a, "paciente": p, "tipo": codigo_tipo(a),
                           "tumor": eh_tumor(a), "normal": eh_normal(a)})
        df = pd.DataFrame(linhas, columns=["id_original", "amostra", "paciente", "tipo", "tumor", "normal"])
        df = df.astype({"tumor": bool, "normal": bool})  # vazio viria como object e quebraria a indexação
        self.disp[chave] = df.drop_duplicates("amostra")
        self.fonte_disp[chave] = fonte

    def calcular_disponibilidade(self):
        c, log = self.cancer, self.log
        pac = self.pacientes()
        log(f"  {len(pac)} pacientes de {c} no TCGA-CDR")
        self._registrar("clinico", list(self.clinico_cdr()["sample"]), "TCGA-CDR (PanCanAtlas)")

        # Mutação e CNV: arquivos pan-câncer de linhas (pequenos), filtrados por paciente
        mut = _ler_tabela_filtrada(self.arquivo("pancan", "mutacao"), "sample", pac)
        self._mut = mut
        self._registrar("mutacao", sorted(mut["sample"].unique()) if len(mut) else [], "MC3 v0.2.8 (PanCanAtlas)")
        seg = _ler_tabela_filtrada(self.arquivo("pancan", "cnv_seg"), "sampleID", pac)
        self._seg = seg
        self._registrar("cnv", sorted(seg["sampleID"].unique()) if len(seg) else [], "SNP6 whitelisted (PanCanAtlas)")

        # Matrizes pequenas: miRNA e RPPA
        for chave in ("mirna", "rppa"):
            with self.f.abrir(self.arquivo("pancan", chave)) as fh:
                cab = fh.readline().rstrip("\n").split("\t")[1:]
            self._registrar(chave, cab, "PanCanAtlas (" + config.DS[chave] + ")")

        # RNA-seq PanCan
        try:
            if self.f.acessivel("pancan"):
                ids = self.f.xena_amostras("pancan", config.DS["rna"])
                fonte = "PanCanAtlas EB++ via API do Xena"
            else:
                raise ErroFonte("API do Xena indisponível")
        except Exception:
            ids = self.f.cabecalho_remoto("pancan", config.DS["rna"])[1:]
            fonte = "PanCanAtlas EB++ (arquivo inteiro lido em fluxo)"
        self._registrar("rna", ids, fonte)

        # Metilação
        self._met_fonte = None
        try:
            if self.f.acessivel("pancan"):
                ids = self.f.xena_amostras("pancan", config.DS["metilacao"])
                self._registrar("metilacao", ids, "PanCanAtlas HM450 via API do Xena")
                self._met_fonte = "api"
        except Exception as e:
            log(f"  metilação pela API falhou: {e}")
        if self._met_fonte is None:
            try:
                ids = self.f.cabecalho_remoto("gdc", f"TCGA-{c}.methylation450.tsv")[1:]
                self._registrar("metilacao", ids, "GDC HM450 (SeSAMe, hg38) via arquivo do GDC hub")
                self._met_fonte = "gdc_hub"
                self.notas.setdefault("metilacao", []).append(
                    "A API do Xena não respondeu; usei a metilação do GDC (processamento SeSAMe/hg38), "
                    "não a versão PanCanAtlas.")
            except Exception as e:
                log(f"  metilação indisponível: {e}")
                self._registrar("metilacao", [], "indisponível")

        # Contagens brutas do GDC
        projeto = f"TCGA-{c}"
        for chave, dtype, wf in (("rna_contagem", "Gene Expression Quantification", "STAR - Counts"),
                                 ("mirna_contagem", "miRNA Expression Quantification", "BCGSC miRNA Profiling")):
            feito = False
            if self.f.acessivel("gdc_api"):
                try:
                    lista = self.f.gdc_arquivos(projeto, dtype, wf)
                    lista["amostra"] = lista["amostra_gdc"].str[:15]
                    self.gdc_listas[chave] = lista
                    self._registrar(chave, list(lista["amostra_gdc"]), f"API do GDC ({wf})")
                    feito = True
                except Exception as e:
                    log(f"  listagem GDC de {chave} falhou: {e}")
            if not feito and chave == "rna_contagem":
                try:
                    ids = self.f.cabecalho_remoto("gdc", f"TCGA-{c}.star_counts.tsv")[1:]
                    self._registrar(chave, ids, "GDC hub (STAR counts em log2(x+1), revertido para contagem)")
                    self.notas.setdefault(chave, []).append(
                        "A API do GDC não respondeu. As contagens vieram do GDC hub do Xena, que guarda "
                        "log2(contagem+1); revertemos com 2^x−1 e arredondamento, o que pode diferir em poucas "
                        "leituras nos genes muito expressos.")
                    feito = True
                except Exception as e:
                    log(f"  contagens alternativas falharam: {e}")
            if not feito:
                self._registrar(chave, [], "indisponível")
                self.notas.setdefault(chave, []).append(
                    "A API do GDC não respondeu nesta execução. Para miRNA não há alternativa com contagem bruta: "
                    "o GDC hub do Xena só guarda log2(RPM+1), e sem o tamanho da biblioteca não dá para voltar à "
                    "contagem. Rode de novo quando a API do GDC estiver acessível."
                    if chave == "mirna_contagem" else "Fonte indisponível nesta execução (API do GDC).")

        # ATAC-seq
        self._atac_ok = False
        if self.f.acessivel("atac"):
            try:
                ids = self.f.xena_amostras("atac", config.DS["atac"])
                mapa = self._mapear_atac(ids)
                self._registrar("atac", [k for k in mapa if not k.startswith("_amostra_")],
                               "ATAC-seq TCGA (Corces 2018) via API do Xena", mapa)
                self._atac_ok = True
            except Exception as e:
                log(f"  ATAC-seq falhou: {e}")
        if not self._atac_ok:
            self._registrar("atac", [], "indisponível")
            self.notas.setdefault("atac", []).append("O hub de ATAC-seq do Xena não respondeu nesta execução.")

        # CPTAC
        estudo = config.CPTAC_ESTUDOS.get(c)
        if estudo and self.f.acessivel("cbio"):
            try:
                ids = self.f.cbio_get(f"sample-lists/{estudo}_protein_quantification/sample-ids")
                self._registrar("cptac", ids, f"CPTAC via cBioPortal ({estudo})")
            except Exception as e:
                log(f"  CPTAC falhou: {e}")
                self._registrar("cptac", [], "indisponível")
        else:
            self._registrar("cptac", [], "não existe para este câncer" if not estudo else "indisponível")
            if not estudo:
                self.notas.setdefault("cptac", []).append(
                    "O CPTAC só mediu proteoma por espectrometria de massa em amostras TCGA de BRCA, "
                    "COAD/READ e OV. As coortes prospectivas do CPTAC são outros pacientes.")

    def _mapear_atac(self, ids: list[str]) -> dict:
        """Devolve {id_original: paciente} só para as amostras deste câncer."""
        mapa = {}
        uuids = {}
        for i in ids:
            if i.startswith("TCGA-"):
                mapa[i] = paciente(i)
                continue
            prefixo = re.split(r"[x_\-]", i, maxsplit=1)[0].upper()
            if prefixo != self.cancer:
                continue
            m = UUID_RE.search(i)
            if m:
                uuids[i] = "-".join(m.groups()).lower()
        if uuids and self.f.acessivel("gdc_api"):
            bar = self.f.gdc_ids_para_barcode(list(set(uuids.values())))
            for i, u in uuids.items():
                b = bar.get(u)
                if b:
                    mapa[i] = paciente(b)
                    if len(b) >= 15:
                        mapa["_amostra_" + i] = amostra15(b)
        elif uuids:
            self.notas.setdefault("atac", []).append(
                "Os IDs do ATAC-seq são UUIDs e a API do GDC não respondeu para convertê-los em pacientes.")
        self.atac_ids = {k: v for k, v in mapa.items() if not k.startswith("_amostra_")}
        return mapa

    def tabela_disponibilidade(self) -> pd.DataFrame:
        """Pacientes x ômicas (True se o paciente tem amostra tumoral naquela ômica)."""
        pac = sorted(self.pacientes())
        out = pd.DataFrame(False, index=pac, columns=[o["chave"] for o in config.OMICAS])
        for chave, df in self.disp.items():
            if chave == "clinico":
                out.loc[df["paciente"].unique(), chave] = True
                continue
            t = df[df["tumor"]]
            out.loc[t["paciente"].unique(), chave] = True
        out.index.name = "paciente"
        return out

    # ------------------------------------------------------------ carregar
    def amostras(self, chave: str, tumor_e_normal: bool = True) -> list[str]:
        df = self.disp.get(chave)
        if df is None or df.empty:
            return []
        if not tumor_e_normal:
            df = df[df["tumor"]]
        return list(df["id_original"])

    def mutacao(self) -> Omica:
        mut = self._mut.copy()
        mut = mut[mut["sample"].map(eh_tumor)]
        mut["sample"] = mut["sample"].str[:15]
        return Omica("mutacao", extra={"mutacoes": mut}, fonte=self.fonte_disp["mutacao"])

    def cnv(self, genes_filtro: set | None = None) -> Omica:
        seg = self._seg.copy()
        seg["sampleID"] = seg["sampleID"].str[:15]
        for c in ("start", "end", "value"):
            seg[c] = pd.to_numeric(seg[c], errors="coerce")
        seg["chr"] = seg["chr"].astype(str).str.replace("chr", "", regex=False).replace({"23": "X", "24": "Y"})
        genes = pd.read_csv(self.arquivo("pancan", "genes_hg19"), sep="\t", dtype=str)
        genes = genes.rename(columns={"#id": "id"})
        genes["chrom"] = genes["chrom"].str.replace("chr", "", regex=False)
        ok_chr = [str(i) for i in range(1, 23)] + ["X", "Y"]
        genes = genes[genes["chrom"].isin(ok_chr)].copy()
        genes["ini"] = pd.to_numeric(genes["chromStart"])
        genes["fim"] = pd.to_numeric(genes["chromEnd"])
        genes["meio"] = (genes["ini"] + genes["fim"]) // 2
        genes = genes.drop_duplicates("gene")
        if genes_filtro:
            sel = genes[genes["gene"].isin(genes_filtro)]
            if len(sel) > 5000:
                genes = sel
        genes = genes.set_index("gene")
        mat = cnv_por_gene(seg, genes)
        return Omica("cnv", matriz=mat, extra={"segmentos": seg, "posicoes": genes[["chrom", "ini", "fim", "meio"]]},
                     fonte=self.fonte_disp["cnv"] + "; nível de gene calculado dos segmentos (hg19)")

    def _matriz_arquivo(self, chave: str) -> pd.DataFrame:
        caminho = self.arquivo("pancan", chave)
        ids = set(self.amostras(chave))
        with self.f.abrir(caminho) as fh:
            cab = fh.readline().rstrip("\n").split("\t")
        usar = [cab[0]] + [c for c in cab[1:] if c in ids]
        df = pd.read_csv(caminho, sep="\t", usecols=usar, index_col=0)
        return _colunas_unicas(df.apply(pd.to_numeric, errors="coerce").astype("float32"))

    def mirna(self) -> Omica:
        return Omica("mirna", matriz=self._matriz_arquivo("mirna"), fonte=self.fonte_disp["mirna"])

    def rppa(self) -> Omica:
        return Omica("rppa", matriz=self._matriz_arquivo("rppa"), fonte=self.fonte_disp["rppa"])

    def rna(self) -> Omica:
        ids = self.amostras("rna")
        if not ids:
            return Omica("rna", fonte="indisponível")
        if "API" in self.fonte_disp["rna"]:
            blocos = list(self.f.xena_blocos("pancan", config.DS["rna"], ids))
            mat = pd.concat(blocos)
        else:
            alvo = set(ids)
            mat = self.f.matriz_remota("pancan", config.DS["rna"], lambda c: c in alvo)
        return Omica("rna", matriz=_colunas_unicas(mat), fonte=self.fonte_disp["rna"])

    def metilacao(self) -> Omica:
        ids = self.amostras("metilacao")
        if not ids:
            return Omica("metilacao", fonte="indisponível", notas=self.notas.get("metilacao", []))
        exemplos = [i for i in ids if amostra15(i) in set(self.exemplos)]
        acc = AcumuladorTopVar(config.TOP_CPGS, exemplos, bins=np.linspace(0, 1, 51))
        if self._met_fonte == "api":
            blocos = self.f.xena_blocos("pancan", config.DS["metilacao"], ids, valores_por_bloco=2_000_000)
        else:
            alvo = set(ids)
            blocos = _blocos_de_linhas(self.f.linhas_remotas("gdc", f"TCGA-{self.cancer}.methylation450.tsv"),
                                       lambda c: c in alvo, 20000, self.log)
        for b in blocos:
            # as 65 sondas rs* do 450k medem genótipo de SNP (controle de identidade), não metilação
            acc.add(b[~b.index.astype(str).str.startswith("rs")])
        top, stats, hist, ex = acc.resultado()
        anot = None
        try:
            mapa = pd.read_csv(self.arquivo("pancan", "met450_map"), sep="\t", dtype=str)
            mapa = mapa.rename(columns={"#id": "id"}).set_index("id")
            anot = mapa.reindex(stats.index)
        except Exception as e:
            self.log(f"  anotação 450k indisponível: {e}")
        hist.columns = [amostra15(c) for c in hist.columns]
        ex = _colunas_unicas(ex) if len(ex) else ex
        return Omica("metilacao", matriz=_colunas_unicas(top),
                     extra={"stats": stats, "hist": hist, "exemplos": ex, "anot": anot},
                     fonte=self.fonte_disp["metilacao"], notas=self.notas.get("metilacao", []))

    def rna_contagem(self) -> Omica:
        return self._contagem("rna_contagem")

    def mirna_contagem(self) -> Omica:
        return self._contagem("mirna_contagem")

    def _contagem(self, chave: str) -> Omica:
        notas = self.notas.get(chave, [])
        if not self.amostras(chave):
            return Omica(chave, fonte="indisponível", notas=notas)
        lista = self.gdc_listas.get(chave)
        if lista is not None:
            lista = lista[lista["amostra_gdc"].isin(self.amostras(chave))]
            lista = lista.sort_values("amostra_gdc").drop_duplicates("amostra")
            if chave == "rna_contagem":
                lidos = self.f.gdc_baixar_tabelas(list(lista["file_id"]), _ler_star)
                genes = next(iter(lidos.values()))[1]
                mat = pd.DataFrame({fid: v[0] for fid, v in lidos.items()})
            else:
                lidos = self.f.gdc_baixar_tabelas(list(lista["file_id"]), _ler_mirna)
                genes = None
                mat = pd.DataFrame(lidos)
            mat = mat.rename(columns=dict(zip(lista["file_id"], lista["amostra"])))
            mat = mat.loc[:, ~mat.columns.duplicated()].astype("float32")
            return Omica(chave, matriz=mat, extra={"genes": genes}, fonte=self.fonte_disp[chave], notas=notas)
        # alternativa: GDC hub em log2(x+1)
        alvo = set(self.amostras(chave))
        mat = self.f.matriz_remota("gdc", f"TCGA-{self.cancer}.star_counts.tsv", lambda c: c in alvo)
        mat = np.round(np.power(2.0, mat.astype("float64")) - 1).clip(lower=0).astype("float32")
        genes = None
        try:
            g = pd.read_csv(self.arquivo("gdc", "gdc_genes"), sep="\t", dtype=str).set_index("id")
            genes = pd.DataFrame({"gene_name": g["gene"], "gene_type": np.nan}).reindex(mat.index)
        except Exception as e:
            self.log(f"  mapa de genes do GDC indisponível: {e}")
        return Omica(chave, matriz=_colunas_unicas(mat), extra={"genes": genes}, fonte=self.fonte_disp[chave],
                     notas=notas)

    def atac(self) -> Omica:
        notas = self.notas.get("atac", [])
        if not self._atac_ok or self.disp["atac"].empty:
            return Omica("atac", fonte="indisponível", notas=notas)
        d = self.disp["atac"]
        ids = list(d["id_original"])
        renome = dict(zip(d["id_original"], d["amostra"]))
        exemplos = [i for i in ids if renome[i] in set(self.exemplos)]
        acc = AcumuladorTopVar(config.TOP_PICOS_ATAC, exemplos)
        for b in self.f.xena_blocos("atac", config.DS["atac"], ids, valores_por_bloco=2_000_000):
            acc.add(b)
        top, stats, _, ex = acc.resultado()
        top = top.rename(columns=renome)
        prom, mapa = None, None
        try:
            prom = pd.concat(list(self.f.xena_blocos("atac", config.DS["atac_promotor"], ids))).rename(columns=renome)
            mapa = pd.read_csv(self.arquivo("atac", "atac_promotor_map"), sep="\t", dtype=str)
            mapa = mapa.rename(columns={"#id": "id"}).set_index(mapa.columns[0] if "id" not in mapa.columns else "id")
        except Exception as e:
            self.log(f"  ATAC promotor/anotação indisponível: {e}")
        return Omica("atac", matriz=top, extra={"stats": stats, "exemplos": ex.rename(columns=renome),
                                                "promotor": prom, "promotor_map": mapa},
                     fonte=self.fonte_disp["atac"], notas=notas)

    def cptac(self) -> Omica:
        estudo = config.CPTAC_ESTUDOS.get(self.cancer)
        notas = self.notas.get("cptac", [])
        ids = self.amostras("cptac")
        if not estudo or not ids:
            return Omica("cptac", fonte=self.fonte_disp.get("cptac", "indisponível"), notas=notas)
        genes = self.f.cbio_get("genes", pageSize=100000, projection="SUMMARY")
        entrez = [g["entrezGeneId"] for g in genes if g.get("entrezGeneId", 0) > 0]
        simb = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in genes}
        linhas = []
        for i in range(0, len(entrez), 3000):
            r = self.f.cbio_post(f"molecular-profiles/{estudo}_protein_quantification/molecular-data/fetch",
                                 {"sampleListId": f"{estudo}_protein_quantification",
                                  "entrezGeneIds": entrez[i:i + 3000]}, projection="SUMMARY")
            linhas += [(simb.get(x["entrezGeneId"]), x["sampleId"], x.get("value")) for x in r]
        df = pd.DataFrame(linhas, columns=["gene", "amostra", "valor"]).dropna(subset=["gene"])
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        mat = df.pivot_table(index="gene", columns="amostra", values="valor", aggfunc="first")
        alvo = set(self.disp["cptac"]["id_original"])
        mat = mat[[c for c in mat.columns if c in alvo]]
        return Omica("cptac", matriz=_colunas_unicas(mat.astype("float32")), fonte=self.fonte_disp["cptac"],
                     notas=notas)

    def carregar(self, chave: str, **kw) -> Omica:
        return getattr(self, chave)(**kw)


# ----------------------------------------------------------------- leitores
def _ler_star(buf: io.BytesIO):
    df = pd.read_csv(buf, sep="\t", comment="#")
    df = df[~df["gene_id"].str.startswith("N_")]
    return df.set_index("gene_id")["unstranded"].astype("float32"), df.set_index("gene_id")[["gene_name", "gene_type"]]


def _ler_mirna(buf: io.BytesIO):
    df = pd.read_csv(buf, sep="\t")
    return df.set_index("miRNA_ID")["read_count"].astype("float32")


def _blocos_de_linhas(linhas, manter, tamanho, log):
    cab = next(linhas).rstrip("\n").split("\t")
    idx = [i for i, c in enumerate(cab) if i > 0 and manter(c)]
    cols = [cab[i] for i in idx]
    nomes, buf, n = [], [], 0
    for linha in linhas:
        p = linha.rstrip("\n").split("\t")
        nomes.append(p[0])
        buf.append([p[i] if i < len(p) else "" for i in idx])
        if len(buf) >= tamanho:
            n += len(buf)
            yield pd.DataFrame(buf, index=nomes, columns=cols).apply(pd.to_numeric, errors="coerce").astype("float32")
            nomes, buf = [], []
            if (n // tamanho) % 5 == 0:
                log(f"    {n:,} linhas lidas")
    if buf:
        yield pd.DataFrame(buf, index=nomes, columns=cols).apply(pd.to_numeric, errors="coerce").astype("float32")


def cnv_por_gene(seg: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    """Valor do segmento que cobre o meio de cada gene, por amostra."""
    amostras = sorted(seg["sampleID"].unique())
    out = np.full((len(genes), len(amostras)), np.nan, dtype="float32")
    gpos = {c: (np.where(genes["chrom"].values == c)[0], genes["meio"].values[genes["chrom"].values == c])
            for c in genes["chrom"].unique()}
    for j, (am, s) in enumerate(seg.groupby("sampleID", sort=True)):
        for c, sc in s.groupby("chr"):
            if c not in gpos:
                continue
            sc = sc.sort_values("start")
            ini, fim, val = sc["start"].values, sc["end"].values, sc["value"].values
            linhas, meios = gpos[c]
            k = np.searchsorted(ini, meios, side="right") - 1
            ok = (k >= 0) & (meios <= fim[np.clip(k, 0, None)])
            out[linhas[ok], j] = val[k[ok]]
    return pd.DataFrame(out, index=genes.index, columns=amostras)


def _achatar_casos_gdc(casos: list[dict]) -> pd.DataFrame:
    linhas = []
    for c in casos:
        r = {"paciente": c.get("submitter_id"), "disease_type": c.get("disease_type"),
             "primary_site": c.get("primary_site")}
        for k, v in (c.get("demographic") or {}).items():
            if not isinstance(v, (dict, list)):
                r[f"{k}.demographic"] = v
        diags = c.get("diagnoses") or []
        prim = next((d for d in diags if str(d.get("diagnosis_is_primary_disease")).lower() == "true"),
                    diags[0] if diags else {})
        for k, v in prim.items():
            if not isinstance(v, (dict, list)):
                r[f"{k}.diagnoses"] = v
        trat = [t for d in diags for t in (d.get("treatments") or [])]
        if trat:
            r["treatment_type.treatments.diagnoses"] = ";".join(sorted({str(t.get("treatment_type")) for t in trat}))
            r["treatment_or_therapy.treatments.diagnoses"] = ";".join(
                sorted({str(t.get("treatment_or_therapy")) for t in trat}))
        exp = (c.get("exposures") or [{}])[0]
        for k, v in exp.items():
            if not isinstance(v, (dict, list)):
                r[f"{k}.exposures"] = v
        fu = [f.get("days_to_follow_up") for f in (c.get("follow_ups") or []) if f.get("days_to_follow_up") is not None]
        r["days_to_follow_up.follow_ups"] = max(fu) if fu else None
        linhas.append(r)
    df = pd.DataFrame(linhas).set_index("paciente")
    drop = [c for c in df.columns if re.search(r"(_id|datetime|state|submitter_id)\.", c)]
    return df.drop(columns=drop).astype(str).replace({"None": np.nan, "nan": np.nan})
