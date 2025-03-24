######################################
## CONTAINER FOR GPU based SpliceAI ##
######################################

# start from the cuda docker base
FROM nvidia/cuda:11.4.0-base-ubuntu20.04

LABEL version="1.3"
LABEL description="This container was tested with \
    -  V100 on AWS p3.2xlarge with nvidia drivers 510.47.03 and cuda v11.6 \
    -  K80 on AWS p2.xlarge  with nvidia drivers 470.141.03 and cuda v11.4 \
    -  Geforce RTX 2070 SUPER (local) with nvidia drivers 470.141.03 and cuda v11.4"

LABEL author="Geert Vandeweyer"
LABEL author.email="geert.vandeweyer@uza.be"

## needed apt packages
ARG BUILD_PACKAGES="wget git bzip2"
# needed conda packages 

ARG CONDA_PACKAGES="python=3.9.13 tensorflow-gpu=2.10.0 cuda-nvcc=11.8.89"

##  ENV SETTINGS during runtime
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND noninteractive

# For micromamba:
SHELL ["/bin/bash", "-l", "-c"]
ENV MAMBA_ROOT_PREFIX=/opt/conda/
ENV PATH=/opt/micromamba/bin:/opt/conda/bin:$PATH
ARG CONDA_CHANNEL="-c bioconda -c conda-forge -c nvidia"

## INSTALL 
RUN apt-get -y update  && \
    apt-get -y install $BUILD_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# conda packages
RUN mkdir /opt/conda && \
    mkdir /opt/micromamba && \
    wget -qO - https://micromamba.snakepit.net/api/micromamba/linux-64/0.23.0 | tar -xvj -C /opt/micromamba bin/micromamba && \
    # initialize bash
    micromamba shell init --shell=bash --prefix=/opt/conda && \
    # remove a statement from bashrc that prevents initialization
    grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/micromamba/bashrc && \
    mv /opt/micromamba/bashrc /root/.bashrc && \
    source ~/.bashrc && \
    # activate & install base conda packag
    micromamba activate && \
    micromamba install -y $CONDA_CHANNEL $CONDA_PACKAGES && \
    micromamba clean --all --yes

# Break cache for recloning git
ARG DATE_CACHE_BREAK=$(date)

# my fork of spliceai : has gpu optimizations
RUN cd /opt/ && \
    git clone https://github.com/geertvandeweyer/SpliceAI.git && \
    cd SpliceAI && \
    python setup.py install 

# no command given, print help.
CMD spliceai -h 

