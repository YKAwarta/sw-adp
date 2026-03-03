#!/bin/bash

# Ensure the script exits if a command fails
set -e

# Default commit message if none is provided
COMMIT_MSG=${1:-"Update repository"}

echo "Status of the repository before adding:"
git status

echo ""
echo "Adding new files and updates to staging area..."
git add .

echo ""
echo "Committing with message: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG" || true # Ignores error if there's nothing to commit

echo ""
echo "Pushing changes to remote..."
git push origin HEAD

echo ""
echo "✅ Repository push complete."
