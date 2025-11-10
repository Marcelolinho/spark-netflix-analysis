docker pull apache/spark:latest

docker run -it --name spark-container apache/spark /bin/bash

docker cp ./spark-netflix-analysis

docker cp ./spark-netflix-analysis spark-container:/opt/spark/work-dir

docker exec -it spark-container /bin/bash

cd spark-netflix-analysis

mkdir analysis
mkdir analysis-top

python -m venv -r requirements.txt

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

spark-submit main.py