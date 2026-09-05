#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the BlastFoam GUI Manager.
#
# Installs the system libraries required to run PyQt5 / VTK / PyVista headless,
# then creates a project virtualenv and installs pinned Python dependencies.
# Safe to run repeatedly: apt and pip both no-op when everything is present.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- System libraries (headless Qt5 + VTK/OpenGL + Xvfb for GUI capture) ----
SYS_PACKAGES=(
  python3-venv python3-dev build-essential
  libgl1 libglu1-mesa libegl1 libopengl0
  libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6
  libx11-6 libxext6 libxrender1 libxkbcommon0 libxkbcommon-x11-0
  libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0
  libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0
  libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-util1 libxcb-cursor0
  libsm6 libice6 libgomp1 xvfb x11-utils
)

if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends "${SYS_PACKAGES[@]}"
fi

# --- Python virtualenv + pinned dependencies -------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -q -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Run tests:  QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true python -m pytest -q"
echo "Run GUI:    xvfb-run -a python main_new.py"
