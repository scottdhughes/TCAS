#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

SHA="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
printf '\\renewcommand{\\CameraReadyCommit}{%s}\n' "${SHA}" > camera_ready_commit.tex
RUN_SHA="$(python3 - <<'PY'
import json
from pathlib import Path
path = Path('../supplementary/run_manifest.json')
if path.exists():
    data = json.loads(path.read_text())
    print(str(data.get('git_sha', 'UNSET'))[:12] or 'UNSET')
else:
    print('UNSET')
PY
)"
printf '\\renewcommand{\\EmpiricalRunCommit}{%s}\n' "${RUN_SHA}" > camera_ready_run_commit.tex

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
  if [ "${RUN_SHA}" != "UNSET" ]; then
    if ! pdftotext TCAS_AAAI2026_CAMERA_READY.pdf - | rg -q "${RUN_SHA}"; then
      echo "Camera-ready PDF does not contain empirical run commit ${RUN_SHA}" >&2
      exit 1
    fi
  fi
else
  if ! strings TCAS_AAAI2026_CAMERA_READY.pdf | rg -q "${SHA}"; then
    echo "Camera-ready PDF does not contain commit hash ${SHA}" >&2
    exit 1
  fi
  if [ "${RUN_SHA}" != "UNSET" ]; then
    if ! strings TCAS_AAAI2026_CAMERA_READY.pdf | rg -q "${RUN_SHA}"; then
      echo "Camera-ready PDF does not contain empirical run commit ${RUN_SHA}" >&2
      exit 1
    fi
  fi
fi

echo "Built ${SCRIPT_DIR}/TCAS_AAAI2026_CAMERA_READY.pdf with build commit ${SHA} and empirical run commit ${RUN_SHA}"
