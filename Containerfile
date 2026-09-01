FROM registry.access.redhat.com/ubi9/python-39:latest

USER root

RUN dnf install -y gcc make && dnf clean all && \
    mkdir -p /opt/app-root/src/data/docs \
             /opt/app-root/src/vectorstore \
             /opt/app-root/src/static \
             /opt/app-root/src/templates \
             /opt/app-root/src/.tmp \
             /opt/app-root/src/.cache && \
    chown -R 1001:0 /opt/app-root/src && \
    rm -rf /var/cache/dnf

USER 1001

WORKDIR /opt/app-root/src

ENV TMPDIR=/opt/app-root/src/.tmp
ENV PIP_NO_CACHE_DIR=1
ENV FASTEMBED_CACHE_PATH=/opt/app-root/src/.cache

COPY --chown=1001:0 requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /opt/app-root/src/.tmp/*

COPY --chown=1001:0 config.py .
COPY --chown=1001:0 main.py .
COPY --chown=1001:0 ingest.py .
COPY --chown=1001:0 rag.py .
COPY --chown=1001:0 cleanup.py .
COPY --chown=1001:0 templates/ templates/
COPY --chown=1001:0 static/ static/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]