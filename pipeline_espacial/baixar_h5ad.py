"""Baixa os arquivos h5ad listados em manifesto_h5ad.csv (gerado pelo pipeline_espacial).

Exemplos:
    python baixar_h5ad.py                                   # tudo, na pasta h5ad/ ao lado do script
    python baixar_h5ad.py --estudo HTAN --destino D:/espacial
    python baixar_h5ad.py --modalidade espacial
    python baixar_h5ad.py --doador HTAPP-213                # espacial + single-cell de um doador
    python baixar_h5ad.py --listar                          # só mostra o que seria baixado e o tamanho

Só precisa de Python 3 e do pacote requests (pip install requests). Arquivos já baixados são pulados.
"""
import argparse
import csv
import os
import sys

import requests


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifesto", default=os.path.join(aqui, "manifesto_h5ad.csv"))
    ap.add_argument("--destino", default=os.path.join(aqui, "h5ad"))
    ap.add_argument("--estudo", help="trecho do nome do estudo (sem diferenciar maiúsculas)")
    ap.add_argument("--modalidade", choices=["espacial", "single-cell"])
    ap.add_argument("--doador", help="ID do doador (como aparece na coluna doadores)")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    with open(a.manifesto, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if a.estudo:
        linhas = [r for r in linhas if a.estudo.lower() in r.get("estudo", a.estudo).lower()]
    if a.doador:
        linhas = [r for r in linhas if a.doador in [x.strip() for x in r["doadores"].split(";")]]
    if a.modalidade:
        linhas = [r for r in linhas if r["modalidade"] == a.modalidade]
    total = sum(float(r["bytes"] or 0) for r in linhas)
    print(f"{len(linhas)} arquivos, {total / 1e9:.1f} GB")
    if a.listar:
        for r in linhas:
            print(f"  {r['modalidade']:12s} {float(r['bytes'] or 0) / 1e6:8.0f} MB  {r['dataset_title']}")
        return 0
    for i, r in enumerate(linhas, 1):
        pasta = os.path.join(a.destino, r.get("estudo", "estudo")[:60].replace("/", "-").strip(), r["modalidade"])
        os.makedirs(pasta, exist_ok=True)
        dest = os.path.join(pasta, r["dataset_id"] + ".h5ad")
        if os.path.exists(dest) and os.path.getsize(dest) == int(float(r["bytes"] or -1)):
            continue
        print(f"[{i}/{len(linhas)}] {r['dataset_title']}", flush=True)
        with requests.get(r["url_h5ad"], stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(dest + ".parcial", "wb") as f:
                for bloco in resp.iter_content(1 << 20):
                    f.write(bloco)
        os.replace(dest + ".parcial", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
