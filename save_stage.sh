#!/bin/bash
set -euo pipefail

# Usage: ./save_stage.sh stage_name [--force]
# Example: ./save_stage.sh step_1_tooling_installation
#          ./save_stage.sh step_1_tooling_installation --force

if [ -z "${1:-}" ]; then
  echo "Usage: ./save_stage.sh stage_name [--force]"
  exit 1
fi

STAGE=$1
FORCE=${2:-""}
DEST="stages/$STAGE"

echo "🔄 Saving snapshot for stage: $STAGE"

# Check if stage already exists
if [ -d "$DEST" ] && [ "$FORCE" != "--force" ]; then
  echo "❌ Stage '$STAGE' already exists. Use --force to overwrite."
  exit 1
fi

# Create destination directory (clean if --force)
if [ -d "$DEST" ] && [ "$FORCE" == "--force" ]; then
  echo "⚠️  Overwriting existing stage '$STAGE'"
  rm -rf "$DEST"
fi
mkdir -p "$DEST"

# Copy from full/ to stages/$STAGE, respecting .gitignore
if [ -f ".gitignore" ]; then
  rsync -av --exclude-from='.gitignore' full/ "$DEST/"
else
  cp -r full/* "$DEST/"
fi

# Add changes to Git
git add "stages/$STAGE"

# Create commit
git commit -m "Snapshot of the stage: $STAGE" || echo "⚠️  No changes to commit"

# Create or update tag
if git rev-parse "$STAGE" >/dev/null 2>&1; then
  git tag -d "$STAGE"
  git tag -a "$STAGE" -m "Snapshot of the stage: $STAGE"
else
  git tag -a "$STAGE" -m "Snapshot of the stage: $STAGE"
fi

# Push to GitHub (main branch + tags)
git push origin main
git push origin --tags --force

echo "✅ Stage '$STAGE' saved and pushed to GitHub!"