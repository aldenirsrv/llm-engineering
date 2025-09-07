#!/bin/bash

# Usage: ./save_stage.sh stage_name
# Example: ./save_stage.sh step_1_tooling_installation

if [ -z "$1" ]; then
  echo "Usage: ./save_stage.sh stage_name"
  exit 1
fi

STAGE=$1
DEST="stages/$STAGE"

echo "🔄 Saving snapshot for stage: $STAGE"

# Create destination directory if it does not exist
mkdir -p "$DEST"

# Copy only tracked files (ignoring .gitignore if present)
if [ -f ".gitignore" ]; then
  rsync -av --exclude-from='.gitignore' full/ "$DEST/"
else
  cp -r full/* "$DEST/"
fi

# Add changes to Git
git add .

# Create commit
git commit -m "Snapshot of the stage: $STAGE"

# Create a lightweight tag
git tag "$STAGE"

# Push to GitHub (main branch + tags)
git push origin main
git push origin --tags

echo "✅ Stage '$STAGE' saved and pushed to GitHub!"