# Será feito um container por ômica, que incluirá as ferramentas necessárias para o préprocessamento dos dados.
# docker build -t process_rnaseq .
# Tem que rodar com docker - só depois vou fazer para executar com nexflow

## Roda o FastQC dentro do container Docker
docker run --rm -v "$(pwd)/../../data:/data" -v "$(pwd)/../../results:/results" process_rnaseq fastqc /data/demo.fastq -o /results/fastqc_results

## Rodar trimmomatic para trimar as reads com baixa qualidade
docker run --rm -v "$(pwd)/../../data:/data" -v "$(pwd)/../../results:/results" process_rnaseq trimmomatic SE -phred33 /data/demo.fastq /results/trimmed/demo_trimmed.fastq TRAILING:10