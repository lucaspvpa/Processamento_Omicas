# Processamento_Omicas

Pipeline de download e exploração multiômica do TCGA e o site que mostra os resultados por câncer.

## Pipeline TCGA multiômico (PanCanAtlas 2018 + contagens do GDC + ATAC + CPTAC)

Baixa os dados de cada câncer do TCGA, gera a exploração completa (figuras, tabelas e um relatório HTML por
ômica), guarda **um paciente de exemplo por combinação de ômicas** e **apaga os dados brutos** no fim. O que
fica no disco são só os resultados e os exemplos.

### Instalação (Windows)

1. Instale o Python 3.10 ou mais novo (python.org, marque "Add python.exe to PATH").
2. Clone este repositório em `C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets\codigo\`
   (`git clone https://github.com/lucaspvpa/Processamento_Omicas codigo`).
3. Abra o PowerShell nessa pasta `codigo` e rode:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se o PowerShell recusar o `activate`, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de
novo.

### Como rodar

Sempre da pasta `codigo` (a que contém `pipeline_tcga`), com o ambiente ativado:

```powershell
# Teste com o CHOL (36 pacientes, ~5 a 15 minutos conforme a internet)
python -m pipeline_tcga --canceres CHOL --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets"

# Só a disponibilidade + UpSet + clínico dos 33 cânceres (~12 minutos, bom para o panorama).
# Com --pular-existentes, o CHOL completo que já rodou não é sobrescrito.
python -m pipeline_tcga --todos --so-disponibilidade --pular-existentes --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets"

# Todos os cânceres, retomando de onde parou se cair
python -m pipeline_tcga --todos --pular-existentes --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets"

# Só os pacotes para download (baixa de novo os dados, sem refazer as análises que já existem)
python -m pipeline_tcga --todos --so-pacotes --pular-existentes --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets"
```

Opções úteis:

| opção | o que faz |
|---|---|
| `--canceres BRCA LUAD` | escolhe os cânceres (siglas TCGA) |
| `--todos` | os 33 cânceres |
| `--omicas rna cnv` | roda só algumas ômicas |
| `--so-disponibilidade` | só disponibilidade, UpSet e clínico |
| `--pular-existentes` | pula cânceres que já têm resultado |
| `--max-exemplos 12` | quantas combinações de ômicas ganham paciente de exemplo |
| `--tmp D:\tcga_tmp` | onde ficam os dados brutos enquanto roda (padrão: pasta temporária do Windows, fora do OneDrive) |
| `--manter-dados` | não apaga os dados brutos |
| `--modo arquivo` | ignora as APIs e baixa arquivos inteiros (mais lento, só se a API falhar) |
| `--so-pacotes` | só grava os dados e monta os pacotes por grupo de ômicas (a página do câncer já deve existir) |
| `--sem-pacotes` | não grava os dados nem os pacotes |
| `--pacotes-metilacao completa` | põe a metilação completa (485 mil CpGs) em cada pacote; o padrão (`top`) põe os CpGs mais variáveis e deixa a completa num arquivo à parte |

### O que é gerado

```
Datasets/
  resultados/
    index.html                      lista de cânceres
    PANCAN/index.html               comparação entre cânceres
    CHOL/
      index.html                    panorama: UpSet, tabela por ômica, links
      resumo.json                   todos os números da análise
      metricas_por_amostra.csv      TMB, fração alterada, metilação global, biblioteca
      00_disponibilidade/           UpSet, combinações, funil, quantidade por ômica
      01_clinico/                   faltantes, estádio, metadados, sobrevida
      02_mutacao/ ... 11_cptac/     um relatório.html + figuras + tabelas por ômica
      20_comparativo/               relações entre ômicas e agrupamento integrado
      dados/<ômica>.csv.gz          cada ômica inteira, todos os pacientes do câncer
      dados/grupos/*.zip            um pacote por grupo de ômicas do funil, com todos os pacientes do grupo
  exemplos/
    CHOL/01_mut-cnv-rna-.../TCGA-XX-XXXX/   dados completos de um paciente por combinação
  logs/
```

### Fontes de cada ômica

| ômica | fonte | forma |
|---|---|---|
| Clínico | TCGA-CDR (Liu 2018) + GDC | tabela por paciente |
| Mutação | MC3 v0.2.8 | lista de mutações |
| CNV | SNP6 whitelisted | segmentos, convertidos para gene |
| RNA-seq PanCan | EB++ (RSEM, corrigido por lote) | log2(norm+1) |
| RNA-seq contagens | GDC STAR - Counts | contagem bruta |
| miRNA PanCan | EB (corrigido por lote) | log2(norm+1) |
| miRNA contagens | GDC BCGSC | contagem bruta |
| Metilação | HM450 PanCan (ou GDC SeSAMe se a API falhar) | beta |
| RPPA | TCGA-RPPA-pancan-clean | normalizado |
| ATAC-seq | Corces 2018 (~400 tumores, 23 cânceres) | log2 CPM |
| Proteoma CPTAC | cBioPortal (só BRCA, COAD/READ, OV) | log2 razão |

Quando uma API não responde, o pipeline usa a alternativa e anota no relatório. As alternativas são: o
arquivo inteiro do Xena lido em fluxo (RNA PanCan), a metilação do GDC hub e as contagens do GDC hub
(log2(x+1) revertido). ATAC-seq e CPTAC só existem pela API.

### Espaço em disco

Durante um câncer grande (BRCA), a pasta temporária pode chegar a alguns GB. Ela é apagada ao fim de cada
câncer, e a pasta compartilhada (~250 MB) é apagada no fim. Os resultados ocupam ~20 a 60 MB por câncer, e os
exemplos de 5 a 50 MB por câncer, conforme o número de combinações.

## Site dos resultados

Site estático com os resultados do `pipeline_tcga` por tipo de câncer: UpSet das combinações de ômicas,
quantidade de dados por ômica, dados clínicos, relatórios por ômica, comparativo entre ômicas e a página
pan-câncer. Cada câncer tem botões para baixar os **dados de exemplo** (um paciente por combinação de ômicas)
e as **tabelas** (CSVs das análises).

```
pipeline_tcga/        o pipeline (python -m pipeline_tcga)
site/                 o que a Vercel publica (HTML, PNG e downloads/)
  index.html          página inicial com a lista de cânceres
  PANCAN/             comparação entre cânceres
  <CANCER>/           panorama e relatórios de cada câncer
  downloads/          exemplos_<CANCER>.zip e tabelas_<CANCER>.zip
scripts/montar_site.py   remonta site/ a partir da saída do pipeline
vercel.json           diz à Vercel para servir site/ sem etapa de build
```

### Publicar na Vercel

Em vercel.com, clique em **Add New → Project**, importe `lucaspvpa/Processamento_Omicas` e clique em
**Deploy**. Não precisa mudar nada: o `vercel.json` já aponta para a pasta `site/` e não há etapa de build.
Cada `git push` na branch principal publica uma nova versão.

### Atualizar com novos resultados

Depois de rodar o pipeline (por exemplo com `--todos`), remonte o site e suba:

```powershell
python scripts/montar_site.py "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets"
git add -A
git commit -m "Atualiza resultados"
git push
```

O script só usa a biblioteca padrão do Python. Ele apaga e recria `site/` a cada execução.

### Tamanho

Com o CHOL completo e os outros 32 cânceres só com disponibilidade, o site tem ~75 MB. Com os 33 cânceres
completos deve passar de 1 GB, por causa das figuras e dos exemplos. Nesse caso, o GitHub recusa arquivos acima
de 100 MB e a Vercel tem limite de tamanho por deploy no plano gratuito; vale publicar só os cânceres que você
quer mostrar ou guardar os zips de exemplo no GitHub Releases ou no Zenodo e trocar os links.

## Levantamento de linhagens celulares (DepMap + Cell Model Passports)

Pacote `pipeline_linhagens`. Baixa a anotação dos modelos e um arquivo por ômica de cada banco, descobre sozinho o
formato de cada tabela (modelos nas linhas, nas colunas ou formato longo) e gera, como no TCGA, uma página por
ômica, o panorama de cada banco (UpSet, funil de interseção, linhagens por tecido e por projeto TCGA, completude das
anotações) e uma comparação DepMap × Passports (linhagens em comum, ômicas nos dois bancos e concordância do RNA-seq
da mesma linhagem). Reaproveita gráficos e relatórios do `pipeline_tcga`.

```powershell
# da pasta que contém pipeline_linhagens e pipeline_tcga, com o .venv ativado
python -m pipeline_linhagens --so-listar --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets\Linhagens"
python -m pipeline_linhagens --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets\Linhagens"
```

| opção | o que faz |
|---|---|
| `--bancos depmap cmp` | escolhe os bancos |
| `--omicas expressao crispr` | só algumas ômicas (chaves em `pipeline_linhagens/fontes.py`) |
| `--so-listar` | só grava `arquivos_disponiveis.csv` e `arquivos_escolhidos.csv` de cada banco |
| `--manter-dados` | não apaga os arquivos baixados |
| `--sem-pacotes` | não grava os dados processados (`<BANCO>/dados/`) nem os pacotes por grupo de ômicas |

Cada banco ganha `dados/<ômica>.csv.gz` (a tabela processada, com o ID do banco) e `dados/grupos/*.zip`: um pacote
por grupo de ômicas do funil, só com as linhagens que têm todas as ômicas do grupo. A página do banco tem os botões
e a seção "De onde vêm os dados", com o link de cada arquivo original e a página da versão. Em `tipos/` há uma
página por tipo de linhagem (tecido), como a página de cada câncer no TCGA: UpSet, funil e pacotes próprios.

Os nomes dos arquivos mudam entre versões do DepMap e do Passports. Cada ômica é escolhida por expressão regular
em `fontes.py` (`DEPMAP_ALVOS`, `CMP_ALVOS`), sempre a versão mais recente; se algum alvo ficar sem arquivo, a lista
completa está em `arquivos_disponiveis.csv`.

## Levantamento single-cell + transcriptômica espacial (CELLxGENE Census)

Pacote `pipeline_espacial`. Lê só os metadados do CELLxGENE Census (versões com `census_spatial_sequencing`):
datasets Visium e Slide-seqV2 por doença, tecido e tecnologia, pareamento single-cell + espacial do mesmo doador
por estudo, tipos celulares das coleções de câncer, anotações dos doadores e o single-cell de câncer do Census
inteiro, com mapeamento aproximado para os projetos do TCGA.

```powershell
pip install -r pipeline_espacial/requirements.txt
python -m pipeline_espacial --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets\Espacial"
```

### Site resultados_espacial (só estudos com single-cell e espacial)

```powershell
python -m pipeline_espacial.pagina --saida "C:\Users\lucas\OneDrive\Desktop\Doutorado\Datasets\Espacial"
```

Gera `resultados_espacial/index.html` com a origem dos dados e uma página por estudo (doadores, mapas espaciais,
qualidade das lâminas, composição celular, concordância single-cell × espacial do mesmo doador) e botões para
baixar cada h5ad. `downloads/baixar_h5ad.py` baixa os h5ad do manifesto (filtros `--estudo` e `--modalidade`).
