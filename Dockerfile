FROM continuumio/anaconda3:2018.12
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      make \
      gcc \
      libz-dev \
      libgl1-mesa-glx \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
ENV CONDA_ENV selene
COPY selene-cpu.yml /tmp/$CONDA_ENV.yml
RUN conda env create -q -f /tmp/selene.yml -n $CONDA_ENV && \
    bash -c '\
    source activate $CONDA_ENV && \
    conda install -c conda-forge -q nose sphinx==4.0.1 sphinx_rtd_theme==0.5.2 && \
    pip install recommonmark'
