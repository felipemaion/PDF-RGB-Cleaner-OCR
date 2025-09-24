# Dockerfile (final robusto)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr tesseract-ocr-por tesseract-ocr-eng \
    fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Tessdata padrão do Debian/Ubuntu
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY pdf_batch_remove_with_previews_and_ocr.py /app/pdf_batch_remove_with_previews_and_ocr.py
COPY entrypoint.sh /app/entrypoint.sh

# Remove CRLF -> LF e garante executável
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

VOLUME ["/data"]

# Wrapper preserva defaults e ANEXA flags passadas no docker run
ENTRYPOINT ["/bin/sh","/app/entrypoint.sh"]
