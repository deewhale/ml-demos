#!/bin/sh
# 新 clone 后跑一次：把 scripts/git-hooks/pre-commit symlink 到 .git/hooks/
set -e
cd "$(git rev-parse --show-toplevel)"
ln -sf ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit
echo "git hooks installed: .git/hooks/pre-commit -> ../../scripts/git-hooks/pre-commit"
