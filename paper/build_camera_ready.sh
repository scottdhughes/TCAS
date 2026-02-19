#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

SHA="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
printf '\\renewcommand{\\CameraReadyCommit}{%s}\n' "${SHA}" > camera_ready_commit.tex

pdflatex -interaction=nonstopmode main.tex >/tmp/tcas_pdflatex_1.log
bibtex main >/tmp/tcas_bibtex.log
pdflatex -interaction=nonstopmode main.tex >/tmp/tcas_pdflatex_2.log
pdflatex -interaction=nonstopmode main.tex >/tmp/tcas_pdflatex_3.log

cp main.pdf TCAS_AAAI2026_CAMERA_READY.pdf

if command -v pdftotext >/dev/null 2>&1; then
  if ! pdftotext TCAS_AAAI2026_CAMERA_READY.pdf - | rg -q "${SHA}"; then
    echo "Camera-ready PDF does not contain commit hash ${SHA}" >&2
    exit 1
  fi
else
  if ! strings TCAS_AAAI2026_CAMERA_READY.pdf | rg -q "${SHA}"; then
    echo "Camera-ready PDF does not contain commit hash ${SHA}" >&2
    exit 1
  fi
fi

echo "Built ${SCRIPT_DIR}/TCAS_AAAI2026_CAMERA_READY.pdf with commit ${SHA}"
