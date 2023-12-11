FROM ubuntu:22.04
MAINTAINER Karl Leswing karl.leswing@schrodinger.com

RUN apt-get update
RUN apt-get install -y curl

RUN mkdir /devtools
COPY ./devtools/install.sh /devtools
COPY ./devtools/environment.yml /devtools
RUN bash devtools/install.sh
ENV PATH="/anaconda/bin:${PATH}"
ENV PATH="/anaconda/condabin:${PATH}"
ENV PATH="/anaconda/envs/local_transform/bin:$PATH"

COPY ./scripts /scripts
COPY ./Synthesis.py /Synthesis.py
COPY ./tmp.py /tmp.py
COPY ./run_reaction.sh /run_reaction.sh
COPY ./LocalTemplate /LocalTemplate
COPY ./data /data
COPY ./models /models

ENV PYTHONPATH='/backend'
ENV CPU_ONLY 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

