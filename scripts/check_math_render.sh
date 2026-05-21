#!/usr/bin/env bash
# Render a README through GitHub's actual markdown API and open the HTML.
# This is ground-truth: same engine, same sanitizer, same KaTeX that
# github.com uses when displaying a README.
#
# Usage:
#   scripts/check_math_render.sh path/to/README.md
#   scripts/check_math_render.sh path/to/README.md --diff
#
# --extract: after rendering, list the LaTeX inside every
# <math-renderer> placeholder. This is what GitHub's markdown sanitizer
# produced before client-side KaTeX runs. Compare each entry against the
# source: if the API mangled an underscore, brace, or backslash, the
# sanitizer is the cause and KaTeX cannot recover.
#
# Requires `gh` CLI authenticated against any GitHub account.

set -euo pipefail

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

file="$1"
extract_mode=false
[[ "${2:-}" == "--extract" || "${2:-}" == "--diff" ]] && extract_mode=true

if [[ ! -f "$file" ]]; then
  echo "error: file not found: $file" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: \`gh\` CLI not installed. Install with: brew install gh" >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: \`gh\` not authenticated. Run: gh auth login" >&2
  exit 2
fi

# For README.md files, name the output after the parent folder so
# /tmp doesn't fill up with `render-README.html` collisions.
fname=$(basename "$file" .md)
if [[ "$fname" == "README" ]]; then
  parent=$(basename "$(dirname "$file")")
  base="${parent}"
else
  base="$fname"
fi
out="/tmp/render-${base}.html"

# gh api /markdown accepts a JSON body. Use -F text=@FILE to load file
# contents and -f mode=gfm for GitHub-flavored Markdown (math enabled).
# Wrap in a minimal HTML shell with the GitHub stylesheets so KaTeX
# renders the way it does on github.com.
body=$(gh api /markdown -f mode=gfm -F text=@"$file" 2>&1) || {
  echo "error: gh api /markdown failed:" >&2
  echo "$body" >&2
  exit 1
}

cat > "$out" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>Render preview: ${file}</title>
<link rel="stylesheet" href="https://github.githubassets.com/assets/light-0eace2597ca3.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<style>
  body { max-width: 980px; margin: 2em auto; padding: 0 1em; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  .source { color: #57606a; font-size: 0.85em; margin-bottom: 1em; }
</style>
<div class="source">Rendered via gh api /markdown from <code>${file}</code></div>
<article class="markdown-body">
${body}
</article>
HTML

echo "wrote: $out"

if $extract_mode; then
  echo ""
  echo "== LaTeX inside <math-renderer> placeholders =="
  echo "(this is what GitHub's sanitizer produced; KaTeX runs client-side)"
  echo ""
  # Pull out each <math-renderer>...</math-renderer> body. KaTeX gets
  # exactly this string. If it doesn't match the source, the sanitizer
  # corrupted it before KaTeX ran.
  perl -0ne 'while (/<math-renderer[^>]*>(.*?)<\/math-renderer>/gs) { print "$1\n" }' "$out" \
    | nl -ba
fi

if command -v open >/dev/null 2>&1; then
  open "$out"
fi
