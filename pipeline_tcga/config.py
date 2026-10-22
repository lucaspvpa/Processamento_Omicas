"""Constantes do pipeline: hubs, nomes dos datasets e definição das ômicas."""
from __future__ import annotations

# Hubs do UCSC Xena (API de consulta, permite baixar só as amostras de um câncer)
HOSTS = {
    "pancan": "https://pancanatlas.xenahubs.net",
    "gdc": "https://gdc.xenahubs.net",
    "atac": "https://atacseq.xenahubs.net",
}

# Endereços de download de arquivo inteiro (usados como alternativa quando a API não responde)
DOWNLOAD_BASES = {
    "pancan": [
        "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/",
        "https://pancanatlas.xenahubs.net/download/",
    ],
    "gdc": [
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
        "https://gdc.xenahubs.net/download/",
    ],
    "atac": ["https://atacseq.xenahubs.net/download/"],
}

GDC_API = "https://api.gdc.cancer.gov"
CBIO_API = "https://www.cbioportal.org/api"

# Datasets do PanCancer Atlas (Hoadley et al., Cell 2018) no hub pancanAtlasHub
DS = {
    "cdr": "Survival_SupplementalTable_S1_20171025_xena_sp",
    "rna": "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena",
    "mirna": "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena",
    "mutacao": "mc3.v0.2.8.PUBLIC.xena",
    "cnv_seg": "broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.xena",
    "metilacao": "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena",
    "rppa": "TCGA-RPPA-pancan-clean.xena",
    "genes_hg19": "probeMap/hugo_gencode_good_hg19_V24lift37_probemap",
    "met450_map": "probeMap/illuminaMethyl450_hg19_GPL16304_TCGAlegacy",
    # ATAC-seq (Corces et al., Science 2018) no hub atacseqHub
    "atac": "TCGA_ATAC_peak_Log2Counts_dedup_sample",
    "atac_promotor": "TCGA_ATAC_peak_Log2Counts_dedup_promoter",
    "atac_promotor_map": "TCGA_ATAC_peak.promoter.probeMap",
    # GDC hub (alternativas por câncer)
    "gdc_genes": "gencode.v36.annotation.gtf.gene.probemap",
}

# Estudos do cBioPortal com proteômica por espectrometria de massa do CPTAC em pacientes TCGA
CPTAC_ESTUDOS = {
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "COAD": "coadread_tcga_pan_can_atlas_2018",
    "READ": "coadread_tcga_pan_can_atlas_2018",
    "OV": "ov_tcga_pan_can_atlas_2018",
}

CANCERES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH",
    "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG",
    "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]

# Ordem e rótulos das ômicas. "pasta" é a subpasta de resultados de cada uma.
OMICAS = [
    {"chave": "clinico", "rotulo": "Clínico", "pasta": "01_clinico"},
    {"chave": "mutacao", "rotulo": "Mutação (MC3)", "pasta": "02_mutacao"},
    {"chave": "cnv", "rotulo": "CNV (SNP6)", "pasta": "03_cnv"},
    {"chave": "rna", "rotulo": "RNA-seq PanCan", "pasta": "04_rnaseq_pancan"},
    {"chave": "rna_contagem", "rotulo": "RNA-seq contagens GDC", "pasta": "05_rnaseq_contagens"},
    {"chave": "mirna", "rotulo": "miRNA PanCan", "pasta": "06_mirna_pancan"},
    {"chave": "mirna_contagem", "rotulo": "miRNA contagens GDC", "pasta": "07_mirna_contagens"},
    {"chave": "metilacao", "rotulo": "Metilação 450k", "pasta": "08_metilacao"},
    {"chave": "rppa", "rotulo": "RPPA", "pasta": "09_rppa"},
    {"chave": "atac", "rotulo": "ATAC-seq", "pasta": "10_atac"},
    {"chave": "cptac", "rotulo": "Proteoma CPTAC", "pasta": "11_cptac"},
]
ROTULO = {o["chave"]: o["rotulo"] for o in OMICAS}
PASTA = {o["chave"]: o["pasta"] for o in OMICAS}

# Códigos de tipo de amostra do barcode TCGA (posições 14-15)
TIPO_AMOSTRA = {
    "01": "Tumor primário", "02": "Tumor recorrente", "03": "Câncer hematológico primário",
    "04": "Câncer hematológico recorrente", "05": "Tumor primário adicional", "06": "Metástase",
    "07": "Metástase adicional", "08": "Células tumorais em cultura", "09": "Medula (tumor)",
    "10": "Sangue normal", "11": "Tecido normal", "12": "Bucal normal", "13": "EBV normal",
    "14": "Medula normal",
}

# Efeitos considerados não silenciosos (alteram proteína)
EFEITOS_NAO_SILENCIOSOS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
    "large deletion",
}

# Limiares de log2(tumor/normal) para chamar ganho e perda no CNV por gene
CNV_GANHO = 0.3
CNV_PERDA = -0.3
TAMANHO_EXOMA_MB = 38.0

# Quantas features mais variáveis manter das matrizes grandes (metilação, ATAC)
TOP_CPGS = 20000
TOP_PICOS_ATAC = 20000
