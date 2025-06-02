FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/conda.yaml .
RUN conda env create -f conda.yaml

COPY MLProject/ .

ENV PATH /opt/conda/envs/smsml-env/bin:$PATH

ENTRYPOINT ["mlflow", "run", "."]