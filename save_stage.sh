#!/bin/bash

if [ -z "$1" ]; then
  echo "Uso: ./save_stage.sh nome_da_etapa"
  exit 1
fi

STAGE=$1
DEST="etapas/$STAGE"

mkdir -p "$DEST"

# Copiar apenas arquivos que não estão no .gitignore
rsync -av --exclude-from='.gitignore' full/ "$DEST/"

git add .
git commit -m "Snapshot of the stage: $STAGE"

# cria tag leve (sem mensagem)
git tag "$STAGE"