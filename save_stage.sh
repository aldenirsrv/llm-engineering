#!/bin/bash

if [ -z "$1" ]; then
  echo "Uso: ./save_stage.sh step_or_stage_name"
  exit 1
fi

STAGE=$1
DEST="etapas/$STAGE"

mkdir -p "$DEST"

# Copy just files that isn't in .gitignore
rsync -av --exclude-from='.gitignore' full/ "$DEST/"

git add .
git commit -m "Snapshot of the stage: $STAGE"
git tag "$STAGE"