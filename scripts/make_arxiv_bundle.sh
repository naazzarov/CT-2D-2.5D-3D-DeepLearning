#!/usr/bin/env bash
# Build the paper and package the source tarball arXiv expects
# (main.tex, main.bbl, figures). arXiv does not run bibtex, so main.bbl is required.
set -euo pipefail

cd "$(dirname "$0")/../paper"

pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null

if grep -q "undefined" main.log; then
  grep -n "undefined" main.log >&2
  echo "Unresolved references or citations; fix before submitting." >&2
  exit 1
fi

out=arxiv_submission.tar.gz
tar -czf "$out" main.tex main.bbl refs.bib figures/*.pdf
echo "Wrote paper/$out"
tar -tzf "$out"
