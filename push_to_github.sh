#!/bin/bash
# ============================================================================
# GitHub Repository Push Script
# Time: 2026-09-10
# Purpose: Push prepared repository to GitHub and create Zenodo archive
# Usage: ./push_to_github.sh
# ============================================================================

set -e

echo "=========================================="
echo "GitHub Repository Push Script"
echo "=========================================="
echo ""

# Configuration
REPO_DIR="/home/rxp/dta2/repo_for_github"
GITHUB_ORG="rxp613-dev"  # Replace with your GitHub organization/username
REPO_NAME="contrastive-dta"
GITHUB_URL="git@github.com:${GITHUB_ORG}/${REPO_NAME}.git"

echo "Step 1: Initialize git repository"
echo "----------------------------------------"
cd "$REPO_DIR"
if [ ! -d .git ]; then
    git init
    echo "✅ Git repository initialized"
else
    echo "⚠️  Git repository already exists"
fi
echo ""

echo "Step 2: Add .gitignore"
echo "----------------------------------------"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db

# Data files (too large for GitHub)
data/raw/
data/processed/
results/controlled/*/
results/cold_start/*/
*.npy
*.pt
*.pth

# Logs
*.log
EOF
echo "✅ .gitignore created"
echo ""

echo "Step 3: Add all files and commit"
echo "----------------------------------------"
git add -A
git status
echo ""
read -p "Review the files above. Press Enter to commit or Ctrl+C to abort..."
git commit -m "Initial commit: Complete reproducibility package

- 300+ training runs across 5 dimensions
- 115-directory KIBA audit with config-level verification
- Holm-corrected statistical analysis (family size = 6)
- Authoritative n=10 KIBA scan: monotonic dose-response (+42.74% at alpha=0.1, Holm p=0.001)
- Cold-target vulnerability characterization (alpha-dependent)
- Full audit trail and verification scripts

Paper: Contrastive Learning for DTA - A Systematic Empirical Benchmark
Authors: Xiaoping Ren, Chaoya Sui, Xiaoyong Liu, Xiaoli Ren, Ling Tao, Yan Zhang
Contact: rxp613@gmail.com"
echo ""

echo "Step 4: Add GitHub remote"
echo "----------------------------------------"
git remote remove origin 2>/dev/null || true
git remote add origin "$GITHUB_URL"
echo "✅ Remote added: $GITHUB_URL"
echo ""

echo "Step 5: Push to GitHub"
echo "----------------------------------------"
echo "⚠️  IMPORTANT: Before pushing, ensure you have:"
echo "   1. Created the repository on GitHub: https://github.com/new"
echo "   2. Repository name: $REPO_NAME"
echo "   3. Organization/User: $GITHUB_ORG"
echo "   4. Set to PUBLIC (for Zenodo integration)"
echo ""
read -p "Press Enter when repository is ready, or Ctrl+C to abort..."

git branch -M main
git push -u origin main
echo ""

echo "=========================================="
echo "✅ Repository pushed successfully!"
echo "=========================================="
echo ""
echo "Repository URL: https://github.com/${GITHUB_ORG}/${REPO_NAME}"
echo ""
echo "Next steps:"
echo "1. Enable Zenodo integration:"
echo "   - Go to https://zenodo.org/account/settings/github/"
echo "   - Enable this repository"
echo "   - Zenodo will automatically archive new releases"
echo ""
echo "2. Create a release for DOI:"
echo "   - Go to https://github.com/${GITHUB_ORG}/${REPO_NAME}/releases/new"
echo "   - Tag version: v1.0.0"
echo "   - Title: 'v1.0.0 - Initial reproducibility package'"
echo "   - Publish release"
echo "   - Zenodo will generate a DOI (e.g., 10.5281/zenodo.XXXXXXX)"
echo ""
echo "3. Update paper with DOI:"
echo "   - Replace 'github.com/contrastive-dta/contrastive-dta' with actual URL"
echo "   - Add '(archived at doi:10.5281/zenodo.XXXXXXX)'"
echo ""
echo "4. Update CITATION.cff with actual DOI"
echo ""
