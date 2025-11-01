FROM apache/spark:latest

USER root

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install pyspark pandas scikit-learn

WORKDIR /usr/src/app

COPY main.py ./

CMD ["python3", "main.py"]