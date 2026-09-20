#!/usr/bin/env bash
# Build an orphan gh-pages branch from site/ without touching HEAD, the index,
# or the working tree.
#
# Uses plumbing (write-tree / commit-tree) against a throwaway index, so it is
# safe to run with uncommitted changes present. The branch is rewritten each
# time rather than appended to, so repeated publishes never grow the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d site ] || { echo "site/ not built - run: python src/build_site.py" >&2; exit 1; }

BRANCH="${1:-gh-pages}"
TMPIDX=$(mktemp -t ghpages-idx.XXXXXX)
rm -f "$TMPIDX"
trap 'rm -f "$TMPIDX"' EXIT

touch site/.nojekyll          # Pages otherwise drops files beginning with _

TREE=$(GIT_INDEX_FILE="$TMPIDX" git --work-tree=site add -A -f . \
       && GIT_INDEX_FILE="$TMPIDX" git write-tree)
COMMIT=$(git commit-tree "$TREE" -m "site: $(date -u '+%Y-%m-%d %H:%M UTC')")
git branch -f "$BRANCH" "$COMMIT"

echo "branch '$BRANCH' -> $(git rev-parse --short "$COMMIT")  ($(git ls-tree -r --name-only "$BRANCH" | wc -l | tr -d ' ') files, $(git cat-file -s "$TREE") B tree)"
echo "publish with:  git push -f origin $BRANCH"
