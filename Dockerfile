FROM informaticsmatters/rdkit-python3-debian:Release_2019_03_1
USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libpango1.0-0 \
    libcairo2 \
    libpq-dev \
    perl \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt upgrade -y
RUN apt install -y python3-pip

MAINTAINER Govinda KC<gbkc@miners.utep.edu>

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY models ./models
COPY static ./static
COPY app.py features.py models.txt ./

ENTRYPOINT ["python3", "app.py"]
EXPOSE 5000
