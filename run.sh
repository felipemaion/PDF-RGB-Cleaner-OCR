#!/usr/bin/env bash
# run.sh — 1-clique para processar PDFs via Docker Compose
set -euo pipefail

# Checagens básicas
command -v docker >/dev/null 2>&1 || { echo "Erro: docker não encontrado."; exit 1; }
if ! docker compose version >/dev/null 2>&1; then
  echo "Erro: 'docker compose' (v2) não encontrado. Atualize o Docker Desktop/Engine."
  exit 1
fi

# Pastas de dados
mkdir -p data/Input data/Output data/OutputTxt

# Build + run
echo "➡️  Construindo imagem e executando (isso pode levar alguns minutos na 1ª vez)..."
docker compose up --build --remove-orphans

# Dica final
echo "✅ Concluído."
echo "📂 PDFs processados:   $(pwd)/data/Output"
echo "📝 Texto (OCR):        $(pwd)/data/OutputTxt"
