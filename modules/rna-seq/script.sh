# Será feito um container por ômica, que incluirá as ferramentas necessárias para o préprocessamento dos dados.
# docker build -t process_rnaseq .
# Tem que rodar com docker - só depois vou fazer para executar com nexflow

## Roda o FastQC dentro do container Docker
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq fastqc /data/demo.fastq -o /results/fastqc_results

## Rodar trimmomatic para trimar as reads com baixa qualidade
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq trimmomatic SE -phred33 /data/demo.fastq /results/trimmed/demo_trimmed.fastq TRAILING:10

## Rodar o FASQC novamente para verificar a qualidade das reads após o trimming
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq fastqc /results/trimmed/demo_trimmed.fastq -o /results/fastqc_results

## Rodar alinhamento com HISAT2 
# Podemos usar o parâmetro rna-strandness para indicar a orientação das reads, caso seja conhecido. F para forward (alinhada) e R para reverse
read -p "Deseja indicar a orientação das redes (F para forward, R para reverse, N para não indicar)? " strandness
if [["$strandness" == "F" || "$strandness" == "R" || "$strandness" == "N" ]]; then
    echo "Orientação das reads: $strandness"
else
    echo "Opção inválida. Usando padrão (não indicando orientação)."
    strandness="N"
fi

if [[ "$strandness" == "N" ]]; then
    docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq bash -c 'hisat2 -q -x /data/grch38/genome -U /results/trimmed/demo_trimmed.fastq | samtools sort -o /results/aligned/demo_trimmed.bam'
else
    docker run --rm -e strandness="$strandness" -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq bash -c 'hisat2 -q --rna-strandness "$strandness" -x /data/grch38/genome -U /results/trimmed/demo_trimmed.fastq | samtools sort -o /results/aligned/demo_trimmed.bam'
fi

if [[ "$strandness" == "N" ]]; then
    docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq featureCounts -a /data/grch38/Homo_sapiens.GRCh38.106.gtf -o /results/counts/demo_featurecounts.txt /results/aligned/demo_trimmed.bam
else
    if [[ "$strandness" == "F" ]]; then
        strandness="1"
    elif [[ "$strandness" == "R" ]]; then
        strandness="2"
    fi
    docker run --rm -e strandness="$strandness" -v "$(pwd)/data:/data" -v "$(pwd)/results:/results" process_rnaseq bash -c 'featureCounts -s "$strandness" -a /data/grch38/Homo_sapiens.GRCh38.106.gtf -o /results/counts/demo_featurecounts.txt /results/aligned/demo_trimmed.bam'
fi