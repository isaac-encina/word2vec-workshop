#!/usr/bin/env bash
# setup.sh — One-command setup for workshop attendees
# Usage: bash setup.sh

set -e   # exit on first error

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Word2Vec Workshop — Setup Script       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "❌  Python not found. Please install Python 3.10+ from https://python.org"
  exit 1
fi

PYTHON=$(command -v python3 || command -v python)
VERSION=$($PYTHON --version 2>&1)
echo "✅  Found $VERSION"

# ── Install dependencies ──────────────────────────────────────
echo ""
echo "📦  Installing dependencies..."
$PYTHON -m pip install -r requirements.txt -q
echo "✅  Dependencies installed"

# ── Run tests ─────────────────────────────────────────────────
echo ""
echo "🧪  Running tests..."
$PYTHON test_word2vec.py
echo ""

# ── Launch Jupyter ────────────────────────────────────────────
echo "🚀  Launching Jupyter Notebook..."
echo "    → Open word2vec_workshop.ipynb"
echo "    → Run each cell with Shift+Enter"
echo ""
$PYTHON -m jupyter notebook
