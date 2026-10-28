"""Pacotes para download por grupo de ômicas em cada câncer, com todos os pacientes do grupo.

Grupo de ômicas = pacientes que têm amostra tumoral em TODAS as ômicas do grupo. Os grupos seguem o funil da
página do câncer: o grupo k reúne as k ômicas mais comuns, então cada grupo tem uma ômica a mais e menos
pacientes que o anterior. Cada pacote é um .zip em resultados/<CANCER>/dados/grupos com o clínico e uma tabela
por ômica, só com os pacientes do grupo (amostras tumorais e normais deles).

Enquanto o câncer roda, cada ômica é gravada inteira em resultados/<CANCER>/dados/<ômica>.csv.gz (features nas
linhas, amostras nas colunas; mutação e segmentos de CNV em formato longo). A metilação e o ATAC são gravados
bloco a bloco durante a leitura, sem passar pela memória inteiros. No fim, cada arquivo é lido uma vez e
distribuído para todos os pacotes que pedem aquela ômica.
"""
from __future__ import annotations

import gzip
import html
import io
import os
import re
import shutil
import zipfile

import pandas as pd

from . import config
from .relatorio import fmt

MARCA_INI, MARCA_FIM = "<!-- pacotes -->", "<!-- /pacotes -->"
COLUNAS_FIXAS = {"gene_name", "gene_type"}  # colunas de anotação que vão em todos os pacotes (contagens do GDC)

CSS = ("<style>.btn{display:inline-block;font-weight:600;padding:6px 12px;border:1px solid var(--acc);"
       "border-radius:6px;text-decoration:none;margin:2px 6px 2px 0;background:var(--panel);font-size:13px}"
       ".btn.cheio{background:var(--acc);color:var(--bg)}</style>")

# arquivo -> (ômica da disponibilidade, formato, coluna da amostra no formato longo)
ARQUIVOS = {
    "mutacao": ("mutacao", "longo", "sample"),
    "cnv_segmentos": ("cnv", "longo", "sampleID"),
    "cnv_genes": ("cnv", "matriz", None),
    "rna": ("rna", "matriz", None),
    "rna_contagem": ("rna_contagem", "matriz", None),
    "mirna": ("mirna", "matriz", None),
    "mirna_contagem": ("mirna_contagem", "matriz", None),
    "metilacao_top": ("metilacao", "matriz", None),
    "metilacao_completa": ("metilacao", "matriz", None),
    "rppa": ("rppa", "matriz", None),
    "atac": ("atac", "matriz", None),
    "cptac": ("cptac", "matriz", None),
}
DESCRICAO = {
    "mutacao": "mutações somáticas MC3 (uma linha por mutação)",
    "cnv_segmentos": "segmentos de CNV SNP6 (log2 tumor/normal)",
    "cnv_genes": "CNV por gene, calculado dos segmentos (hg19)",
    "rna": "RNA-seq PanCan (EB++ normalizado)",
    "rna_contagem": "RNA-seq, contagens brutas do GDC (STAR)",
    "mirna": "miRNA PanCan normalizado",
    "mirna_contagem": "miRNA, contagens brutas do GDC",
    "metilacao_top": f"metilação 450k, os {fmt(config.TOP_CPGS, 0)} CpGs mais variáveis do câncer",
    "metilacao_completa": "metilação 450k, todos os CpGs",
    "rppa": "RPPA",
    "atac": "ATAC-seq, todos os picos (log2 contagens)",
    "cptac": "proteoma CPTAC",
}


def mb(n) -> str:
    if not n:
        return "—"
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{max(n / 1e6, 0.1):.1f} MB"


class Gravador:
    """Grava as ômicas inteiras de um câncer em <pasta>/dados e monta os pacotes no fim."""

    def __init__(self, pasta_cancer: str, metilacao: str = "top"):
        self.pasta = os.path.join(pasta_cancer, "dados")
        shutil.rmtree(self.pasta, ignore_errors=True)  # dados de uma execução anterior
        os.makedirs(self.pasta, exist_ok=True)
        self.metilacao = metilacao  # "top" ou "completa": qual metilação entra nos pacotes
        self.arquivos: dict[str, str] = {}

    def _cam(self, nome: str) -> str:
        return os.path.join(self.pasta, f"{nome}.csv.gz")

    def matriz(self, nome: str, M: pd.DataFrame, anot: pd.DataFrame | None = None):
        if M is None or M.shape[1] == 0:
            return
        X = M if anot is None else anot.reindex(M.index).join(M)
        X.index.name = "feature"
        with gzip.open(self._cam(nome), "wt", encoding="utf-8", compresslevel=6) as f:
            X.to_csv(f, float_format="%.6g")
        self.arquivos[nome] = self._cam(nome)

    def longo(self, nome: str, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        with gzip.open(self._cam(nome), "wt", encoding="utf-8", compresslevel=6) as f:
            df.to_csv(f, index=False)
        self.arquivos[nome] = self._cam(nome)

    def em_blocos(self, nome: str):
        """Escritor incremental (metilação e ATAC): chame com cada bloco features x amostras e depois .fechar()."""
        return _EscritorBlocos(self, nome)

    def omica(self, om):
        """Grava o que está na memória depois de carregar a ômica."""
        k = om.chave
        if k == "mutacao":
            self.longo("mutacao", om.extra.get("mutacoes"))
        elif k == "cnv":
            self.longo("cnv_segmentos", om.extra.get("segmentos"))
            self.matriz("cnv_genes", om.matriz)
        elif k == "metilacao":
            self.matriz("metilacao_top", om.matriz)
        elif k == "rna_contagem":
            g = om.extra.get("genes")
            self.matriz(k, om.matriz, g if isinstance(g, pd.DataFrame) else None)
        elif k != "atac":  # o ATAC completo já foi gravado em blocos
            self.matriz(k, om.matriz)

    # ------------------------------------------------------------------ pacotes
    def nos_pacotes(self) -> dict[str, str]:
        """Arquivos que entram nos pacotes (só uma das duas metilações)."""
        fora = "metilacao_completa" if self.metilacao == "top" else "metilacao_top"
        if fora == "metilacao_top" and "metilacao_completa" not in self.arquivos:
            fora = "metilacao_completa"
        return {k: v for k, v in self.arquivos.items() if k != fora}

    def montar(self, tab: pd.DataFrame, clin: pd.DataFrame, gdc: pd.DataFrame | None, cancer: str, log) -> list[dict]:
        """tab: pacientes x ômicas (bool, da disponibilidade). Devolve uma linha por grupo para a página."""
        usar = self.nos_pacotes()
        omicas = sorted({ARQUIVOS[a][0] for a in usar})
        t = tab[[c for c in tab.columns if c in omicas]]
        pasta_g = os.path.join(self.pasta, "grupos")
        os.makedirs(pasta_g, exist_ok=True)
        pacotes = []
        for i, g in enumerate(grupos_funil(t), start=1):
            pac = sorted(t.index[t[g].all(axis=1)])
            abrev = "-".join(_ABREV.get(c, c) for c in g)
            cam = os.path.join(pasta_g, f"{cancer}_grupo_{i:02d}_{abrev}_{len(pac)}pacientes.zip")
            arqs = [a for a in usar if ARQUIVOS[a][0] in g]
            pacotes.append({"grupo": i, "chaves": g, "omicas": [config.ROTULO[c] for c in g], "pacientes": len(pac),
                            "ids": set(pac), "lista": pac, "cam": cam, "arquivos": arqs,
                            "arquivo": os.path.relpath(cam, os.path.dirname(self.pasta)).replace(os.sep, "/")})
        for p in pacotes:
            with zipfile.ZipFile(p["cam"], "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("clinico_cdr.csv", clin.reindex(p["lista"]).to_csv())
                if gdc is not None and len(gdc):
                    z.writestr("clinico_gdc.csv", gdc[gdc.index.isin(p["ids"])].to_csv())
                z.writestr("LEIA-ME.txt", _leia_me(p, cancer))
        for a in usar:
            alvos = [p for p in pacotes if a in p["arquivos"]]
            if alvos:
                log(f"    pacotes: {a} em {len(alvos)} pacotes")
                _distribuir(usar[a], a, alvos)
        for p in pacotes:
            p["bytes"] = os.path.getsize(p["cam"])
            log(f"    {os.path.basename(p['cam'])}: {mb(p['bytes'])}")
        return pacotes

    def secao_html(self, grupos: list[dict]) -> str:
        """Seção 'Baixar os dados' da página do câncer (entre marcadores, para poder ser trocada)."""
        corpo_g = "".join(
            f"<tr><td>{g['grupo']}</td><td class='t'>{html.escape(' + '.join(g['omicas']))}</td>"
            f"<td>{fmt(g['pacientes'], 0)}</td><td>{mb(g['bytes'])}</td>"
            f"<td>{_btn(g['arquivo'], 'baixar grupo', True)}</td></tr>" for g in grupos)
        corpo_o = "".join(
            f"<tr><td class='t'>{html.escape(DESCRICAO.get(a, a))}</td><td>{mb(os.path.getsize(c))}</td>"
            f"<td>{_btn('dados/' + os.path.basename(c), 'baixar', True)}</td></tr>"
            for a, c in self.arquivos.items())
        met = ("Nos pacotes, a metilação traz os CpGs mais variáveis; a metilação completa está na tabela por ômica."
               if self.metilacao == "top" and "metilacao_completa" in self.arquivos else "")
        return (MARCA_INI + CSS + "<h2>Baixar os dados</h2>"
                "<h3>Por grupo de ômicas</h3><p>Cada pacote traz <b>todos</b> os pacientes que têm amostra tumoral "
                "em todas as ômicas do grupo: o clínico (TCGA-CDR e GDC) e uma tabela por ômica com as amostras "
                "desses pacientes (tumor e, quando existe, normal). O grupo 1 tem as ômicas mais comuns; cada grupo "
                f"seguinte acrescenta uma ômica e perde pacientes, como no funil. {met}</p>"
                "<div class='tabwrap'><table class='tab'><thead><tr><th>grupo</th><th class='t'>ômicas</th>"
                f"<th>pacientes</th><th>tamanho</th><th></th></tr></thead><tbody>{corpo_g}</tbody></table></div>"
                "<h3>Por ômica (todos os pacientes do câncer)</h3>"
                "<div class='tabwrap'><table class='tab'><thead><tr><th class='t'>arquivo</th><th>tamanho</th>"
                f"<th></th></tr></thead><tbody>{corpo_o}</tbody></table></div>" + MARCA_FIM)


class _EscritorBlocos:
    def __init__(self, g: Gravador, nome: str):
        self.g, self.nome = g, nome
        self.f = gzip.open(g._cam(nome), "wt", encoding="utf-8", compresslevel=6)
        self.cab = True

    def __call__(self, bloco: pd.DataFrame):
        bloco.index.name = "feature"
        bloco.to_csv(self.f, header=self.cab, float_format="%.6g")
        self.cab = False

    def fechar(self):
        self.f.close()
        if self.cab:  # nada escrito
            os.remove(self.g._cam(self.nome))
        else:
            self.g.arquivos[self.nome] = self.g._cam(self.nome)


_ABREV = {"mutacao": "mut", "cnv": "cnv", "rna": "rna", "rna_contagem": "rnaCont", "mirna": "mir",
          "mirna_contagem": "mirCont", "metilacao": "met", "rppa": "rppa", "atac": "atac", "cptac": "cptac"}


def _btn(href: str, txt: str, cheio: bool = False) -> str:
    return (f"<a class='btn{' cheio' if cheio else ''}' href='{html.escape(href)}' download>"
            f"{html.escape(txt)}</a>")


def grupos_funil(tab: pd.DataFrame) -> list[list[str]]:
    """Grupos (listas de chaves) seguindo o funil; só entra quando o número de pacientes muda."""
    ordem = [c for c in tab.sum().sort_values(ascending=False).index if tab[c].any()]
    grupos, ant = [], None
    for k in range(1, len(ordem) + 1):
        g = ordem[:k]
        n = int(tab[g].all(axis=1).sum())
        if n == 0:
            break
        if ant is not None and n == ant:
            grupos[-1] = g  # mesma contagem: a ômica nova não custa pacientes, entra no grupo anterior
        else:
            grupos.append(g)
        ant = n
    return grupos


def _leia_me(p: dict, cancer: str) -> str:
    linhas = [f"TCGA-{cancer}, grupo {p['grupo']}: {' + '.join(p['omicas'])}",
              f"{p['pacientes']} pacientes com amostra tumoral em todas estas ômicas.", "",
              "Arquivos:", "  clinico_cdr.csv: TCGA-CDR (Liu et al. 2018), uma linha por paciente",
              "  clinico_gdc.csv: clínico do GDC (quando disponível)"]
    linhas += [f"  {a}.csv: {DESCRICAO.get(a, a)}" for a in p["arquivos"]]
    linhas += ["", "Matrizes: features nas linhas (coluna feature), amostras nas colunas (barcode de 15 caracteres;",
               "código 01-09 tumor, 10-19 normal). Os 12 primeiros caracteres são o paciente.",
               "Formato longo (mutação, segmentos de CNV): uma linha por evento, coluna sample ou sampleID."]
    return "\n".join(linhas) + "\n"


def _distribuir(cam: str, nome: str, alvos: list[dict]):
    """Uma passada pelo arquivo da ômica, escrevendo as colunas (ou linhas) de cada pacote no seu zip."""
    _, forma, col_am = ARQUIVOS[nome]
    abertos = []
    try:
        for p in alvos:
            z = zipfile.ZipFile(p["cam"], "a", zipfile.ZIP_DEFLATED)
            t = io.TextIOWrapper(z.open(f"{nome}.csv", "w", force_zip64=True), encoding="utf-8", newline="")
            abertos.append((z, t, p["ids"]))
        cols = {}
        primeiro = True
        for bloco in pd.read_csv(cam, chunksize=300_000 if forma == "longo" else 5000, dtype=str,
                                 keep_default_na=False):
            for i, (_, t, ids) in enumerate(abertos):
                if forma == "longo":
                    b = bloco[bloco[col_am].str[:12].isin(ids)]
                else:
                    if i not in cols:
                        cols[i] = [c for c in bloco.columns if c == "feature" or c in COLUNAS_FIXAS
                                   or (re.match(r"TCGA-", c) and c[:12] in ids)]
                    b = bloco[cols[i]]
                b.to_csv(t, index=False, header=primeiro)
            primeiro = False
    finally:
        for z, t, _ in abertos:
            t.close()
            z.close()


def injetar(pagina: str, secao: str):
    """Põe (ou troca) a seção de downloads na página do câncer, logo depois dos relatórios."""
    if not os.path.exists(pagina):
        return
    s = open(pagina, encoding="utf-8").read()
    s = re.sub(re.escape(MARCA_INI) + ".*?" + re.escape(MARCA_FIM), "", s, flags=re.S)
    i = s.find("<h2>Combinações de ômicas por paciente</h2>")
    s = s[:i] + secao + s[i:] if i >= 0 else s.replace("</main>", secao + "</main>", 1)
    open(pagina, "w", encoding="utf-8").write(s)
