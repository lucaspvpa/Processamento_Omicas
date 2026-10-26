"""Downloads com cache, novas tentativas e leitura de arquivos .zip/.gz."""
from __future__ import annotations

import io
import os
import re
import time
import zipfile

import requests

CABECALHO = {"User-Agent": "pipeline-linhagens/1.0 (pesquisa academica)"}


def sessao() -> requests.Session:
    s = requests.Session()
    s.headers.update(CABECALHO)
    return s


def obter_texto(s: requests.Session, url: str, tentativas: int = 4) -> str:
    for i in range(tentativas):
        try:
            r = s.get(url, timeout=120)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == tentativas - 1:
                raise
            time.sleep(2 ** (i + 1))
    return ""


def baixar(s: requests.Session, url: str, destino: str, log=print, tentativas: int = 4) -> str:
    """Baixa url para destino (pula se já existe com tamanho > 0). Devolve o caminho."""
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return destino
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    tmp = destino + ".parcial"
    for i in range(tentativas):
        try:
            with s.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                feito, t0 = 0, time.time()
                with open(tmp, "wb") as f:
                    for bloco in r.iter_content(chunk_size=1 << 20):
                        f.write(bloco)
                        feito += len(bloco)
                        if time.time() - t0 > 30:
                            t0 = time.time()
                            log(f"    {os.path.basename(destino)}: {feito / 1e6:,.0f} MB"
                                + (f" de {total / 1e6:,.0f} MB" if total else ""))
            os.replace(tmp, destino)
            return destino
        except Exception as e:
            log(f"    falha ao baixar {os.path.basename(destino)} ({e}); tentativa {i + 1}/{tentativas}")
            if i == tentativas - 1:
                raise
            time.sleep(2 ** (i + 1))
    return destino


def nome_seguro(nome: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", nome)


def tabelas_do_arquivo(caminho: str) -> list[tuple[str, object]]:
    """Lista (nome, fonte) de cada tabela de texto do arquivo. Em um .zip, uma por membro .csv/.tsv/.txt."""
    if caminho.lower().endswith(".zip"):
        z = zipfile.ZipFile(caminho)
        saida = []
        for m in z.namelist():
            base = os.path.basename(m)
            if base.startswith(".") or "__MACOSX" in m:
                continue
            if re.search(r"\.(csv|tsv|txt)(\.gz)?$", m, re.I):
                saida.append((base, (z, m)))
        return saida
    return [(os.path.basename(caminho), caminho)]


def abrir(fonte):
    """Abre a fonte devolvida por tabelas_do_arquivo como objeto de arquivo binário."""
    if isinstance(fonte, tuple):
        z, m = fonte
        f = z.open(m)
        if m.lower().endswith(".gz"):
            import gzip
            return gzip.open(f)
        return f
    if str(fonte).lower().endswith(".gz"):
        import gzip
        return gzip.open(fonte)
    return open(fonte, "rb")


def separador(fonte) -> str:
    with abrir(fonte) as f:
        linha = io.TextIOWrapper(f, encoding="utf-8", errors="replace").readline()
    return "\t" if linha.count("\t") > linha.count(",") else ","
