#!/usr/bin/env bash
# Fast grep-based detector for the GitHub-KaTeX render bugs the Python
# validator enforces. Sub-second feedback for iterative editing.
#
# Usage:
#   scripts/grep_math_bugs.sh                  # scan whole repo
#   scripts/grep_math_bugs.sh path/to/file.md  # scan one file
#   scripts/grep_math_bugs.sh some/folder/     # scan a directory
#
# Skips _legacy/, docs/qc-reports/, .git/, and the script itself.
# Exits 1 if any pattern hits, 0 if clean.

set -euo pipefail

target="${1:-.}"

if [[ ! -e "$target" ]]; then
  echo "error: path not found: $target" >&2
  exit 2
fi

# Gather .md files under the target, excluding the usual non-catalog paths.
# Portable: avoid `mapfile` (bash 4+) so this works on macOS default bash 3.2.
files=()
if [[ -f "$target" ]]; then
  files+=("$target")
else
  while IFS= read -r -d '' f; do
    files+=("$f")
  done < <(find "$target" -type f -name '*.md' \
    -not -path '*/_legacy/*' \
    -not -path '*/docs/qc-reports/*' \
    -not -path '*/docs/superpowers/*' \
    -not -path '*/.git/*' \
    -not -path '*/.serena/*' \
    -not -name 'bullshit-detector_*.md' \
    -not -name 'CLAUDE.md' \
    -not -name 'handoff.md' \
    -print0)
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no markdown files under $target"
  exit 0
fi

total=0

# Helper: report pattern findings. Args: name, regex.
scan() {
  local name="$1" regex="$2"
  local hits
  hits=$(grep -nE "$regex" "${files[@]}" 2>/dev/null || true)
  if [[ -n "$hits" ]]; then
    local n
    n=$(printf '%s\n' "$hits" | wc -l | tr -d ' ')
    total=$((total + n))
    printf '\n[%s] %d hit(s)\n' "$name" "$n"
    printf '%s\n' "$hits" | sed 's/^/  /'
  fi
}

# Bare-dollar math is forbidden everywhere; the catalog uses code-fence
# math syntax (`$`expr`$` inline, ```math fence for display). This is
# the only authoritative check; the others below are KaTeX-genuine
# pattern hits that still apply inside fenced math.
scan "bare \$...\$ or \$\$...\$\$ math"  '(^|[^\\\`])\$[^$\`]+\$|^\$\$|\$\$$'
scan "unbraced math-font macro"     '\\(mathbb|mathbf|mathcal|mathrm|mathfrak|mathit|mathsf|mathtt)[[:space:]]+[A-Za-z0-9]'

echo
if (( total == 0 )); then
  echo "clean: no patterns hit across ${#files[@]} file(s)"
  exit 0
else
  echo "total findings: $total across ${#files[@]} file(s)"
  echo "run \`python scripts/validate_catalog.py\` for the authoritative check"
  exit 1
fi
