#!/bin/bash
# LLM-CFD GitHub Repository Sync Script
# Pushes all local files to https://github.com/LucienHuang2501/LLM-CFD
#
# Usage:
#   bash sync_to_github.sh
#
# You will be prompted for your GitHub username and Personal Access Token (PAT).
# Create a PAT at: https://github.com/settings/tokens (needs "repo" scope)

set -e

REPO_URL="https://github.com/LucienHuang2501/LLM-CFD.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

echo "============================================"
echo "LLM-CFD GitHub Repository Sync"
echo "============================================"
echo "Local directory: $(pwd)"
echo "Remote: $REPO_URL"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Ensure remote is set
if ! git remote get-url origin &>/dev/null; then
    git remote add origin "$REPO_URL"
else
    git remote set-url origin "$REPO_URL"
fi

# Ensure git identity is set
if ! git config user.email &>/dev/null; then
    git config user.email "hchiuwing@163.com"
    git config user.name "Huang Zhaoyong"
fi

# Add all files (respecting .gitignore)
echo "Staging files..."
git add -A

# Show what will be committed
CHANGED=$(git status --porcelain | wc -l)
if [ "$CHANGED" -gt 0 ]; then
    echo "Files to commit: $CHANGED"
    git commit -m "sync: update repo to match paper v33

- README.md: v33-consistent values (DeepSeek-V4-Flash, F1=0.737, AUPRC=0.495, 15 GT rules)
- 17 experiment .py scripts
- docs/SUPPLEMENTARY_THEORY.md (8 sections)
- experiments/PROMPT_TEMPLATES.md
- All result JSONs, cache files, figures (PNG+SVG)
- UCI Adult dataset (census+income/)
- .gitignore for clean structure"
else
    echo "No changes to commit."
fi

echo ""
echo "Pushing to GitHub..."
echo "You will be asked for your GitHub credentials."
echo "Username: your GitHub username"
echo "Password: your Personal Access Token (NOT your GitHub password)"
echo ""

# Force push to replace old repo content
git push -f origin main

echo ""
echo "============================================"
echo "Sync complete!"
echo "Repository: https://github.com/LucienHuang2501/LLM-CFD"
echo "============================================"
