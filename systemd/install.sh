#!/usr/bin/env bash
# Install the DS4 server as a systemd *user* service.
#
# Usage:
#   systemd/install.sh                        # defaults below
#   DS4_CTX=65536 DS4_BATCH=2 systemd/install.sh
#   DS4_MODEL=/path/to/model.gguf systemd/install.sh
#   INSTALL_ONLY=1 systemd/install.sh         # copy + reload, do not enable/start
#
# Defaults:
#   DS4_MODEL  ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
#   DS4_CTX    131072
#   DS4_BATCH  4
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="$repo_dir/ds4-server"
if [[ ! -x "$bin" ]]; then
    echo "error: $bin not built -- run 'make strix-halo' first" >&2
    exit 1
fi

model="${DS4_MODEL:-$HOME/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
ctx="${DS4_CTX:-131072}"
batch="${DS4_BATCH:-4}"

if [[ ! -f "$model" ]]; then
    echo "error: model not found: $model (set DS4_MODEL=...)" >&2
    exit 1
fi

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
unit="$unit_dir/ds4-server.service"

sed -e "s|__DS4_DIR__|$repo_dir|g" \
    -e "s|__MODEL__|$model|g" \
    -e "s|__CTX__|$ctx|g" \
    -e "s|__BATCH__|$batch|g" \
    "$repo_dir/systemd/ds4-server.service" > "$unit"

systemctl --user daemon-reload

if [[ "${INSTALL_ONLY:-0}" == "1" ]]; then
    echo "installed $unit (not enabled/started; INSTALL_ONLY=1)"
    exit 0
fi

systemctl --user enable --now ds4-server.service

if ! loginctl enable-linger "$USER" 2>/dev/null; then
    echo "note: could not enable lingering; the service stops when you log out."
    echo "      run 'loginctl enable-linger $USER' to keep it running."
fi

systemctl --user --no-pager status ds4-server.service
