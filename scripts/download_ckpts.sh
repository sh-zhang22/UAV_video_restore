#!/usr/bin/env bash
# 从 HuggingFace 拉取 SeedVR/SeedVR2 ckpts 到 third_party/SeedVR/ckpts/。
# 仅在没有可共享 NAS 的情况下使用；本组服务器请优先 symlink /nas/datasets/zsh/seedvr_ckpts/。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/third_party/SeedVR/ckpts"
mkdir -p "$CKPT_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] 未找到 huggingface-cli，请先 pip install -U huggingface_hub" >&2
  exit 1
fi

# VAE 与四个 DiT ckpt 各自的仓库；SeedVR 系列 repo 文件命名是 ema_vae.pth + seedvr*_ema_*.pth
hf_dl() {
  local repo="$1"
  local file="$2"
  echo "[*] $repo :: $file"
  huggingface-cli download "$repo" "$file" --local-dir "$CKPT_DIR" --local-dir-use-symlinks False
}

# 这五个文件是 4 个 method 共用的，VAE 仅一份。
hf_dl ByteDance-Seed/SeedVR2-3B  ema_vae.pth
hf_dl ByteDance-Seed/SeedVR2-3B  seedvr2_ema_3b.pth
hf_dl ByteDance-Seed/SeedVR2-7B  seedvr2_ema_7b.pth
hf_dl ByteDance-Seed/SeedVR-3B   seedvr_ema_3b.pth
hf_dl ByteDance-Seed/SeedVR-7B   seedvr_ema_7b.pth

echo "Done. ckpts at $CKPT_DIR"
ls -lh "$CKPT_DIR"
