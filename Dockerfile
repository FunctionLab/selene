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
    conda install -q nose sphinx sphinx_rtd_theme && \
    pip install recommonmark'
