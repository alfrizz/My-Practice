#!/usr/bin/env bash
set -euo pipefail

# A) Unset bad NODE_OPTIONS (fixes openssl-legacy error)
unset NODE_OPTIONS

# B) Create a simple symlink “analysis” to your Stock Analysis folder
ln -sfn "/workspace/my models/Trading/0.Stock Analysis" "/workspace/analysis"

# C) Print a friendly, click-ready launch banner
cat <<EOF

======================
  JupyterLab is live!
  → http://localhost:8888/
    (opens in 'analysis')
  Your full project remains under /workspace
======================

EOF

# D) Launch JupyterLab:
#    - serve /workspace as root
#    - default‐land in /lab/tree/analysis
#    - disable extension‐manager checks
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.allow_origin='*' \
  --NotebookApp.default_url='/lab/tree/analysis' \
  --ExtensionManagerLabApp.extension_manager_enabled=False \
  --LabApp.open_browser=False
