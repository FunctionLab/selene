#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

vcf_filepath="${1:-}"
hg_version="${2:-}"  # input hg19 or hg38
outdir="${3:-}"

mkdir -p $outdir

vcf_basename=$(basename $vcf_filepath)

if [ "${hg_version}" = "hg19" ]
then
    python vep_cli_expecto.py $vcf_filepath \
                      ./resources/hg19_UCSC.fa \
                      $outdir

elif [ "${hg_version}" = "hg38" ]
then
    python vep_cli_expecto.py $vcf_filepath \
                      ./resources/hg38_UCSC.fa \
                      $outdir
else
    echo "'$hg_version' currently not supported"
fi

