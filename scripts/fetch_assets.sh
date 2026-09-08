#!/usr/bin/env bash
# Sparse-clone the three asset repos at the commits pinned in configs/g1_reconcile.yaml.
set -euo pipefail
DEST="${O2S_THIRD_PARTY:-$HOME/o2s_third_party}"
mkdir -p "$DEST"

fetch () {
  local repo="$1" subdir="$2" commit="$3" name="${1##*/}"
  if [ ! -d "$DEST/$name/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "https://github.com/$repo.git" "$DEST/$name"
  fi
  (
    cd "$DEST/$name"
    git sparse-checkout init --cone
    git sparse-checkout set "$subdir"
    git fetch -q --depth 1 origin "$commit"
    git checkout -q "$commit"
    echo "$name @ $(git rev-parse HEAD)"
  )
}

fetch google-deepmind/mujoco_menagerie unitree_g1 8161bba264d7fa7c99ca301e91e7fb44737676ad
fetch google-deepmind/mujoco_playground mujoco_playground/_src/locomotion/g1 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad
fetch unitreerobotics/unitree_ros robots/g1_description 7d6075f7f58588b189b940130e3edab3c839b2df
