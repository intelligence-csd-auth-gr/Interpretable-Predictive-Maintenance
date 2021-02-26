FROM ubuntu:18.04
FROM python:3.7.8
    
RUN apt-get update
RUN apt-get dist-upgrade -y

WORKDIR /usr/src/app

COPY VisioRedDemo.zip .

RUN unzip VisioRedDemo.zip

RUN pip install --upgrade setuptools

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install voila

RUN cp lime_tabular.py /usr/local/lib/python3.7/site-packages/lime
RUN cp model.py /usr/local/lib/python3.7/site-packages/nbeats_keras

RUN python3 -m pip install --upgrade notebook

CMD ["voila", "VisioRedDemo.ipynb", "--port=8866", "--no-browser"]