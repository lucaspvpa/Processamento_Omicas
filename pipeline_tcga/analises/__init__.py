"""Análises por ômica. Cada módulo gera figuras, tabelas e um relatório HTML na pasta da ômica."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .. import config
from ..dados import codigo_tipo, eh_normal, eh_tumor, paciente
from ..graficos import estadio_grupo


@dataclass
class Contexto:
    cancer: str
    pasta: str                      # resultados/<CANCER>
    clin: pd.DataFrame              # TCGA-CDR, uma linha por paciente
    log: Callable
    resumo: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)   # métricas por amostra que vão para o pan-câncer

    def pasta_omica(self, chave: str) -> str:
        p = os.path.join(self.pasta, config.PASTA.get(chave, chave))
        os.makedirs(p, exist_ok=True)
        return p

    def nav(self, nivel: int = 1) -> str:
        sobe = "../" * nivel
        links = [f'<a href="{sobe}index.html">← Panorama {self.cancer}</a>']
        return "<nav>" + "".join(links) + "</nav>"

    def anotar(self, amostras) -> pd.DataFrame:
        """Tabela de anotação por amostra: tipo, estádio, sexo, idade, estado vital."""
        linhas = []
        for a in amostras:
            p = paciente(a)
            tipo = "Tumor" if eh_tumor(a) else ("Normal" if eh_normal(a) else "Outro")
            c = self.clin.loc[p] if p in self.clin.index else None
            linhas.append({
                "amostra": a, "paciente": p, "tipo": tipo, "codigo": codigo_tipo(a),
                "estadio": estadio_grupo(c["ajcc_pathologic_tumor_stage"]) if c is not None else "NA",
                "sexo": (c["gender"] if c is not None and isinstance(c["gender"], str) else "NA"),
                "idade": pd.to_numeric(c["age_at_initial_pathologic_diagnosis"], errors="coerce") if c is not None else None,
                "obito": pd.to_numeric(c["OS"], errors="coerce") if c is not None else None,
            })
        return pd.DataFrame(linhas).set_index("amostra")
