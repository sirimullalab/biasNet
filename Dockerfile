FROM informaticsmatters/rdkit-python3-debian:Release_2019_03_1

MAINTAINER Govinda KC<gbkc@miners.utep.edu>

USER ${UID}:${GID}
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libpango1.0-0 \
    libcairo2 \
    libpq-dev \
    perl \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt upgrade -y
RUN apt install -y python3-pip

WORKDIR /app

COPY models ./models
COPY gpcr ./gpcr
COPY static ./static
COPY logp.bin logs.bin drug_central_drugs.csv drug_central_drugs-stand.csv lookup_table_smiles.json requirements.txt app.py features.py models.txt gpcr.txt run_biasnet.py ./
RUN pip3 install -r requirements.txt

#ENTRYPOINT ["python3", "run_biasnet.py"]
#---------For web application --------#
ENTRYPOINT ["python3", "app.py"]
EXPOSE 5000
