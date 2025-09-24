# entrypoint.sh  (salve com LF, não CRLF)
#!/bin/sh
set -e

# Garante as pastas
mkdir -p /data/Input /data/Output /data/OutputTxt

# Defaults; "$@" anexa seus args
exec python /app/pdf_batch_remove_with_previews_and_ocr.py \
  -i /data/Input -o /data/Output --dpi 300 --text-dir /data/OutputTxt \
  "$@"
