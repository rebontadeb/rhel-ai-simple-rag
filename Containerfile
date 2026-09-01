FROM registry.access.redhat.com/ubi9/python-39:latest

USER root

RUN dnf install -y gcc make && dnf clean all && \
    mkdir -p /opt/app-root/src/data/docs \
             /opt/app-root/src/vectorstore \
             /opt/app-root/src/static \
             /opt/app-root/src/templates \
             /tmp && \
    chmod 1777 /tmp && \
    chown -R 1001:0 /opt/app-root/src

USER 1001

WORKDIR /opt/app-root/src

COPY --chown=1001:0 requirements.txt .
COPY --chown=1001:0 config.py .
COPY --chown=1001:0 main.py .
COPY --chown=1001:0 ingest.py .
COPY --chown=1001:0 rag.py .
COPY --chown=1001:0 cleanup.py .
COPY --chown=1001:0 templates/ templates/
COPY --chown=1001:0 static/ static/

ENV TMPDIR=/tmp

RUN pip install --upgrade pip && \
    pip install pysqlite3-binary pdfminer.six && \
    pip install -r requirements.txt && \
    pip install "chromadb==1.0.0"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

