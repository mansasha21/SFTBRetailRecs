FROM ubuntu:20.04
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

COPY src/ .
COPY data/ data/
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN chmod +x recs.sh
RUN mkdir cache export
ENTRYPOINT [ "./recs.sh" ]
