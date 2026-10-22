"""Acesso às fontes: API do UCSC Xena, downloads de arquivo, API do GDC e API do cBioPortal.

Tudo que vem da rede passa por aqui. Os downloads retomam de onde pararam quando a
conexão cai, e a API de cada hub é testada uma vez para decidir se usamos a consulta
por amostras (menos dados baixados) ou o arquivo inteiro.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import time
from typing import Callable, Iterator
from urllib.parse import quote

import pandas as pd
import requests

from . import config


class ErroFonte(RuntimeError):
    pass


def _q(x) -> str:
    """Converte um valor Python em literal da linguagem de consulta do Xena."""
    if x is None:
        return "nil"
    if isinstance(x, str):
        return '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(x, (list, tuple)):
        return "[" + " ".join(_q(i) for i in x) + "]"
    return str(x)


Q_AMOSTRAS = ('(fn [dataset limit] (map :value (query {:select [:value] :from [:dataset] '
              ':join [:field [:= :dataset.id :dataset_id] :code [:= :field.id :field_id]] '
              ':limit limit :where [:and [:= :dataset.name dataset] [:= :field.name "sampleID"]]})))')
Q_CAMPOS = ('(fn [dataset] (map :name (query {:select [:field.name] :from [:dataset] '
            ':join [:field [:= :dataset.id :dataset_id]] :where [:= :dataset.name dataset]})))')
Q_FETCH = '(fn [dataset samples probes] (fetch [{:table dataset :columns probes :samples samples}]))'


class LeitorRetomavel(io.RawIOBase):
    """Lê uma URL como fluxo de bytes e reconecta com Range se a conexão cair."""

    def __init__(self, sessao: requests.Session, url: str, log, tentativas: int = 8):
        self.s, self.url, self.log, self.tentativas = sessao, url, log, tentativas
        self.pos = 0
        self.total = None
        self._resp = None
        self._it = None
        self._buf = b""
        self._abrir()

    def _abrir(self):
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        r = self.s.get(self.url, stream=True, timeout=(30, 300), headers=headers)
        if r.status_code not in (200, 206):
            raise ErroFonte(f"HTTP {r.status_code} em {self.url}")
        if self.pos and r.status_code == 200:
            raise ErroFonte("servidor não aceita retomada (Range)")
        if self.total is None:
            cl = r.headers.get("Content-Length")
            self.total = int(cl) if cl else None
        self._resp = r
        self._it = r.raw.stream(1 << 20, decode_content=False)

    def readable(self):
        return True

    def readinto(self, b):
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                if self.total is not None and self.pos < self.total:
                    self._reconectar(ErroFonte("fluxo terminou antes do fim"))
                    continue
                return 0
            except Exception as e:  # conexão caiu no meio
                self._reconectar(e)
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.pos += n
        return n

    def _reconectar(self, erro):
        for i in range(self.tentativas):
            espera = min(60, 2 ** (i + 1))
            self.log(f"    conexão interrompida ({type(erro).__name__}); retomando do byte {self.pos:,} em {espera}s")
            time.sleep(espera)
            try:
                self._resp.close()
            except Exception:
                pass
            try:
                self._abrir()
                return
            except Exception as e:
                erro = e
        raise ErroFonte(f"não consegui retomar o download de {self.url}: {erro}")

    def close(self):
        try:
            if self._resp is not None:
                self._resp.close()
        finally:
            super().close()


class Fontes:
    def __init__(self, log, modo: str = "auto"):
        self.log = log
        self.modo = modo  # auto | api | arquivo
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "pipeline_tcga/1.0"
        self._status: dict[str, bool] = {}

    # ------------------------------------------------------------------ util
    def _requisitar(self, metodo: str, url: str, tentativas: int = 5, **kw) -> requests.Response:
        ultimo = None
        tempo = kw.pop("timeout", (30, 600))
        for i in range(tentativas):
            try:
                r = self.s.request(metodo, url, timeout=tempo, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise ErroFonte(f"HTTP {r.status_code}")
                return r
            except (requests.exceptions.ProxyError, requests.exceptions.SSLError) as e:
                raise ErroFonte(f"bloqueado: {e}") from e
            except Exception as e:
                ultimo = e
                time.sleep(min(60, 2 ** (i + 1)))
        raise ErroFonte(f"falhou {metodo} {url}: {ultimo}")

    def acessivel(self, nome: str) -> bool:
        """Testa uma vez se um serviço responde (hub Xena, GDC ou cBioPortal)."""
        if nome in self._status:
            return self._status[nome]
        ok = False
        try:
            if nome in config.HOSTS:
                if self.modo == "arquivo":
                    ok = False
                else:
                    r = self.s.post(config.HOSTS[nome] + "/data/", data=b"(+ 1 2)",
                                    headers={"Content-Type": "text/plain"}, timeout=20)
                    ok = r.ok and r.text.strip().startswith("3")
            elif nome == "gdc_api":
                ok = self.s.get(config.GDC_API + "/status", timeout=20).ok
            elif nome == "cbio":
                ok = self.s.get(config.CBIO_API + "/health", timeout=20).ok
        except Exception:
            ok = False
        self._status[nome] = ok
        self.log(f"  [rede] {nome}: {'acessível' if ok else 'indisponível'}")
        return ok

    # --------------------------------------------------------------- Xena API
    def xena(self, hub: str, fn: str, *params, timeout=(30, 900)):
        consulta = "(%s %s)" % (fn, " ".join(_q(p) for p in params))
        r = self._requisitar("POST", config.HOSTS[hub] + "/data/", data=consulta.encode(),
                             headers={"Content-Type": "text/plain"}, timeout=timeout)
        if not r.ok:
            raise ErroFonte(f"Xena {hub} respondeu {r.status_code}: {r.text[:200]}")
        return json.loads(r.text)

    def xena_amostras(self, hub: str, dataset: str) -> list[str]:
        return [str(x) for x in self.xena(hub, Q_AMOSTRAS, dataset, None)]

    def xena_campos(self, hub: str, dataset: str) -> list[str]:
        return [str(x) for x in self.xena(hub, Q_CAMPOS, dataset) if x != "sampleID"]

    def xena_blocos(self, hub: str, dataset: str, amostras: list[str], campos: list[str] | None = None,
                    valores_por_bloco: int = 3_000_000) -> Iterator[pd.DataFrame]:
        """Gera blocos (features x amostras) de um dataset denso, só com as amostras pedidas."""
        if campos is None:
            campos = self.xena_campos(dataset=dataset, hub=hub)
        passo = max(200, int(valores_por_bloco / max(1, len(amostras))))
        total = len(campos)
        for i in range(0, total, passo):
            bloco = campos[i:i + passo]
            vals = self.xena(hub, Q_FETCH, dataset, amostras, bloco)
            # o fetch devolve uma lista por requisição; aceitamos a matriz embrulhada ou não
            while len(vals) == 1 and isinstance(vals[0], list) and vals[0] and isinstance(vals[0][0], list):
                vals = vals[0]
            df = pd.DataFrame(vals, index=bloco, columns=amostras)
            df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
            if (i // passo) % 10 == 0:
                self.log(f"    {dataset[:40]}: {min(i + passo, total):,}/{total:,} features")
            yield df

    # ------------------------------------------------------------- arquivos
    def urls(self, hub: str, dataset: str) -> list[str]:
        nome = quote(dataset, safe="/")
        out = []
        for base in config.DOWNLOAD_BASES[hub]:
            out += [base + nome + ".gz", base + nome]
        return out

    def _primeira_url(self, hub: str, dataset: str) -> str:
        for u in self.urls(hub, dataset):
            try:
                r = self.s.head(u, timeout=30, allow_redirects=True)
                if r.ok:
                    return r.url
            except Exception:
                continue
        raise ErroFonte(f"nenhuma URL de download respondeu para {dataset}")

    def baixar(self, hub: str, dataset: str, destino: str) -> str:
        """Baixa o arquivo para `destino` (retomando quedas). Devolve o caminho."""
        for cand in (destino, destino + ".gz"):
            if os.path.exists(cand) and os.path.getsize(cand) > 0:
                return cand
        url = self._primeira_url(hub, dataset)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tmp = destino + ".parcial"
        self.log(f"    baixando {os.path.basename(dataset)}")
        with LeitorRetomavel(self.s, url, self.log) as src, open(tmp, "wb") as out:
            while True:
                bloco = src.read(1 << 20)
                if not bloco:
                    break
                out.write(bloco)
        with open(tmp, "rb") as f:
            if f.read(2) == b"\x1f\x8b" and not destino.endswith(".gz"):
                destino += ".gz"  # o pandas reconhece a compressão pela extensão
        os.replace(tmp, destino)
        return destino

    def linhas_remotas(self, hub: str, dataset: str) -> Iterator[str]:
        """Lê um arquivo remoto linha a linha sem salvar em disco (gz ou texto)."""
        url = self._primeira_url(hub, dataset)
        raw = io.BufferedReader(LeitorRetomavel(self.s, url, self.log), buffer_size=1 << 20)
        try:
            gz = raw.peek(2)[:2] == b"\x1f\x8b"
            fh = gzip.GzipFile(fileobj=raw) if gz else raw
            for linha in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                yield linha
        finally:
            raw.close()

    def cabecalho_remoto(self, hub: str, dataset: str) -> list[str]:
        it = self.linhas_remotas(hub, dataset)
        try:
            return next(it).rstrip("\n").split("\t")
        finally:
            it.close()

    def matriz_remota(self, hub: str, dataset: str, manter: Callable[[str], bool]) -> pd.DataFrame:
        """Lê uma matriz (linhas = features, colunas = amostras) mantendo só as colunas pedidas."""
        it = self.linhas_remotas(hub, dataset)
        cab = next(it).rstrip("\n").split("\t")
        idx = [i for i, c in enumerate(cab) if i > 0 and manter(c)]
        nomes, linhas = [], []
        for n, linha in enumerate(it):
            p = linha.rstrip("\n").split("\t")
            nomes.append(p[0])
            linhas.append([p[i] if i < len(p) else "" for i in idx])
            if n and n % 5000 == 0:
                self.log(f"    {dataset[:40]}: {n:,} linhas lidas")
        df = pd.DataFrame(linhas, index=nomes, columns=[cab[i] for i in idx])
        return df.apply(pd.to_numeric, errors="coerce").astype("float32")

    @staticmethod
    def abrir(caminho: str):
        with open(caminho, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        return gzip.open(caminho, "rt", encoding="utf-8", errors="replace") if gz else open(
            caminho, "rt", encoding="utf-8", errors="replace")

    # --------------------------------------------------------------- GDC API
    def gdc_post(self, endpoint: str, corpo: dict, stream=False) -> requests.Response:
        r = self._requisitar("POST", f"{config.GDC_API}/{endpoint}", json=corpo, stream=stream,
                             timeout=(30, 1800))
        if not r.ok:
            raise ErroFonte(f"GDC {endpoint} respondeu {r.status_code}: {r.text[:300]}")
        return r

    def gdc_arquivos(self, projeto: str, data_type: str, workflow: str | None) -> pd.DataFrame:
        filtros = [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [projeto]}},
            {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ]
        if workflow:
            filtros.append({"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow]}})
        corpo = {"filters": {"op": "and", "content": filtros}, "format": "JSON", "size": 10000,
                 "fields": "file_id,file_name,file_size,cases.submitter_id,cases.samples.submitter_id,"
                           "cases.samples.sample_type"}
        hits = self.gdc_post("files", corpo).json()["data"]["hits"]
        linhas = []
        for h in hits:
            caso = (h.get("cases") or [{}])[0]
            am = (caso.get("samples") or [{}])[0]
            linhas.append({"file_id": h["file_id"], "file_name": h["file_name"],
                           "file_size": h.get("file_size", 0), "paciente": caso.get("submitter_id"),
                           "amostra_gdc": am.get("submitter_id"), "tipo_gdc": am.get("sample_type")})
        return pd.DataFrame(linhas)

    def gdc_baixar_tabelas(self, ids: list[str], ler: Callable[[io.BytesIO], object],
                           lote: int = 40) -> dict[str, object]:
        """Baixa arquivos do GDC em lotes e aplica `ler` a cada um, sem gravar em disco."""
        out = {}
        for i in range(0, len(ids), lote):
            grupo = ids[i:i + lote]
            for tentativa in range(4):
                try:
                    out.update(self._gdc_lote(grupo, ler))
                    break
                except Exception as e:
                    if tentativa == 3:
                        raise
                    self.log(f"    lote GDC falhou ({e}); tentando de novo")
                    time.sleep(10 * (tentativa + 1))
            self.log(f"    GDC: {min(i + lote, len(ids))}/{len(ids)} arquivos")
        return out

    def _gdc_lote(self, grupo, ler):
        out = {}
        r = self.gdc_post("data", {"ids": grupo}, stream=True)
        if len(grupo) == 1:
            out[grupo[0]] = ler(io.BytesIO(r.content))
            return out
        conteudo = io.BytesIO(r.content)  # lote pequeno cabe na memória (~40 x 4 MB)
        with tarfile.open(fileobj=conteudo, mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.name.endswith("MANIFEST.txt"):
                    continue
                fid = m.name.split("/")[0]
                out[fid] = ler(io.BytesIO(tf.extractfile(m).read()))
        return out

    def gdc_casos(self, projeto: str) -> list[dict]:
        corpo = {
            "filters": {"op": "in", "content": {"field": "project.project_id", "value": [projeto]}},
            "format": "JSON", "size": 5000,
            "expand": "demographic,diagnoses,diagnoses.treatments,exposures,follow_ups",
        }
        return self.gdc_post("cases", corpo).json()["data"]["hits"]

    def gdc_ids_para_barcode(self, uuids: list[str]) -> dict[str, str]:
        """Mapeia UUIDs (caso, amostra ou alíquota) para o barcode do paciente."""
        if not uuids:
            return {}
        campos = ["case_id", "samples.sample_id", "samples.portions.analytes.aliquots.aliquot_id"]
        corpo = {"filters": {"op": "or", "content": [
            {"op": "in", "content": {"field": c, "value": uuids}} for c in campos]},
            "format": "JSON", "size": 5000,
            "fields": "submitter_id,case_id,samples.sample_id,samples.submitter_id,"
                      "samples.portions.analytes.aliquots.aliquot_id"}
        out = {}
        for h in self.gdc_post("cases", corpo).json()["data"]["hits"]:
            out[h["case_id"]] = h["submitter_id"]
            for s in h.get("samples") or []:
                out[s.get("sample_id")] = s.get("submitter_id", h["submitter_id"])
                for p in s.get("portions") or []:
                    for a in p.get("analytes") or []:
                        for al in a.get("aliquots") or []:
                            out[al.get("aliquot_id")] = s.get("submitter_id", h["submitter_id"])
        return out

    # ----------------------------------------------------------- cBioPortal
    def cbio_get(self, caminho: str, **params):
        r = self._requisitar("GET", f"{config.CBIO_API}/{caminho}", params=params,
                             headers={"Accept": "application/json"})
        if not r.ok:
            raise ErroFonte(f"cBioPortal {caminho}: {r.status_code}")
        return r.json()

    def cbio_post(self, caminho: str, corpo, **params):
        r = self._requisitar("POST", f"{config.CBIO_API}/{caminho}", json=corpo, params=params,
                             headers={"Accept": "application/json"})
        if not r.ok:
            raise ErroFonte(f"cBioPortal {caminho}: {r.status_code} {r.text[:200]}")
        return r.json()
