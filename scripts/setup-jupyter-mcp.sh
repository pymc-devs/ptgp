#!/usr/bin/env bash
#
# setup-jupyter-mcp.sh
#
# Reproduce the JupyterLab + Jupyter MCP server setup on a new machine.
# Sets up:
#   1. The shared token env file (~/.config/jupyter-mcp.env)
#   2. The `jlab` launcher function in ~/.zshrc
#   3. The Claude Code `jupyter` MCP server (user scope)
#
# Prerequisites (installed automatically where possible):
#   - uv / uvx        : runs the MCP server package
#   - jupyterlab      : install into your ptgp conda env
#   - jupyter-collaboration : required by the MCP server for live editing
#
# Usage:
#   bash setup-jupyter-mcp.sh
#
# Idempotent: safe to re-run. Existing config is left in place unless missing.

set -euo pipefail

# --- Shared tokens (keep identical across machines so JUPYTER_TOKEN matches) ---
JUPYTER_TOKEN="J7NWWjS5VKrGMi5Zv14-EkVoTIO3aejOxkGjHVnFj50"
MCP_TOKEN="QNcgURDRbx17PyT7wqG3pxFwB2bOIgndxO9jekrVDSU"

ENV_FILE="$HOME/.config/jupyter-mcp.env"
ZSHRC="$HOME/.zshrc"

echo "==> 1/4  Installing uv (provides uvx) if missing"
if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "    uvx already installed: $(command -v uvx)"
fi

echo "==> 2/4  Writing shared token file: $ENV_FILE"
mkdir -p "$(dirname "$ENV_FILE")"
if [ -f "$ENV_FILE" ]; then
  echo "    $ENV_FILE already exists, leaving it in place."
else
  cat > "$ENV_FILE" <<EOF
JUPYTER_TOKEN=$JUPYTER_TOKEN
MCP_TOKEN=$MCP_TOKEN
EOF
  chmod 600 "$ENV_FILE"
  echo "    wrote $ENV_FILE (mode 600)"
fi

echo "==> 3/4  Adding jlab() launcher to $ZSHRC"
if grep -q "^jlab()" "$ZSHRC" 2>/dev/null; then
  echo "    jlab() already defined in $ZSHRC, skipping."
else
  cat >> "$ZSHRC" <<'EOF'

# Launch JupyterLab wired to the Jupyter MCP server.
# Sources the shared token file so JUPYTER_TOKEN matches the MCP config.
jlab() {
  # Activate the project env so notebooks run against ptgp.
  conda activate ptgp 2>/dev/null || true
  set -a; source ~/.config/jupyter-mcp.env; set +a
  jupyter lab \
    --port 8888 \
    --ip 127.0.0.1 \
    --IdentityProvider.token "$JUPYTER_TOKEN"
}
EOF
  echo "    appended jlab() to $ZSHRC"
fi

echo "==> 4/4  Registering the jupyter MCP server with Claude Code (user scope)"
if claude mcp get jupyter >/dev/null 2>&1; then
  echo "    'jupyter' MCP server already registered, skipping."
  echo "    (to reset: claude mcp remove jupyter -s user, then re-run)"
else
  claude mcp add jupyter -s user \
    -e JUPYTER_URL=http://localhost:8888 \
    -e "JUPYTER_TOKEN=$JUPYTER_TOKEN" \
    -e "MCP_TOKEN=$MCP_TOKEN" \
    -e ALLOW_IMG_OUTPUT=true \
    -- uvx jupyter-mcp-server@latest
  echo "    registered."
fi

cat <<'EOF'

Done. Next steps:

  1. Install JupyterLab into your ptgp env (once):
       conda activate ptgp
       pip install jupyterlab jupyter-collaboration

  2. Reload your shell:
       source ~/.zshrc

  3. Start the lab server:
       jlab

  4. In another terminal, confirm the MCP server connects:
       claude mcp get jupyter        # should show: Connected

The order matters: jlab (the Jupyter server) must be running for the
MCP server to connect.
EOF
