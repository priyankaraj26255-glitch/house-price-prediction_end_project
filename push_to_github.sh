#!/usr/bin/env bash
# Usage: ./push_to_github.sh https://github.com/<you>/house-price-predictor.git
set -e
[ -z "$1" ] && { echo "Usage: $0 <github-repo-url>"; exit 1; }
git init -q && git add . && git commit -qm "feat: end-to-end house price predictor (EDA, modeling, Flask app)"
git branch -M main && git remote add origin "$1" && git push -u origin main
echo "Pushed. Enable Pages: Settings → Pages → Branch main, folder /docs"
