# Dependencies
Please install the following before running `process_data.sh`:
- [htslib](https://www.htslib.org) for `bgzip` and `tabix`
- docopt (Python package, `conda install -c anaconda docopt`)
- numpy (Python package, `conda install -c anaconda numpy`)
- pyfaidx (Python package, `conda install -c bioconda pyfaidx`)

You should have already run [`../download_data.sh`](https://github.com/FunctionLab/selene/blob/master/manuscript/case1/download_data.sh). If you have  not yet done so, please run it from the main `case1` directory (if you are currently in this directory: `cd .. ; sh download_data.sh`).

# Additional note
The file `hg38_TF_intervals.txt` is a file of regions where the combined ENCODE and Roadmap Epigenomics DeepSEA dataset contained at least 1 transcription factor. These are the regions that the DeepSEA model was trained on, which is why we used this as an example for Selene's intervals sampler here. The hg19 file can be generated from scripts provided for [case study 2](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/1_train_with_online_sampler/data). `hg38_TF_intervals.txt` is the lifted over file from hg19 to hg38. 
