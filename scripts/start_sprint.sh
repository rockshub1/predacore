#!/bin/bash
# PredaCore Development Sprint Initiator
set -euo pipefail

# Configuration
SPRINT_GOALS=(
    "Intelligence Engine: Implement meta-reasoning capabilities"
    "World Interaction: Develop adaptive API layer"
    "Knowledge System: Add cross-domain synthesis"
)
SPRINT_DURATION=7 # days
CODE_REVIEWERS=("lead-engineer" "qa-specialist")

# Initialize sprint
echo "🚀 Starting PredaCore Development Sprint"
echo "📅 Duration: $SPRINT_DURATION days"
echo "🎯 Goals:"
for goal in "${SPRINT_GOALS[@]}"; do
    echo " - $goal"
done

# Refuse to mutate git state if working tree is dirty — previously this script
# unconditionally `git checkout main && git pull && git checkout -b sprint/...`,
# which would either fail (and dump the user mid-checkout) or carry uncommitted
# changes onto main.
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "❌ Working tree has uncommitted changes."
    echo "   Commit, stash, or pass --force (not implemented) to override."
    git status --short
    exit 1
fi
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "❌ Untracked files present. Commit/stash/clean before starting a sprint."
    git status --short
    exit 1
fi

# Setup environment
git checkout main
git pull
git checkout -b "sprint/$(date +%Y-%m-%d)"

# Assign tasks
echo "👥 Assigning code reviewers: ${CODE_REVIEWERS[*]}"
echo "🔧 Configuring CI pipeline with:"
echo "   - Automated testing (90% coverage required)"
echo "   - Performance benchmarking"
echo "   - Security scanning"

# Pre-sprint verification — we used to chain
#   ./scripts/run_tests.sh && ./scripts/check_coverage.sh 90 && ./scripts/security_scan.sh
# but those helpers were aspirational stubs that never landed. The H27 fix
# replaces the false-pass chain with explicit "run these manually" guidance
# so the script doesn't silently report success.
echo "🔍 Pre-sprint verification (run these manually before starting):"
echo "   - tests:    .venv/bin/python -m pytest"
echo "   - coverage: .venv/bin/python -m pytest --cov=src/predacore --cov-report=term-missing"
echo "   - security: .venv/bin/python -m bandit -r src/predacore || pipx run pip-audit"

echo
echo "✅ Sprint branch created: sprint/$(date +%Y-%m-%d)"
echo "📌 Daily standup: ./scripts/daily_standup.sh"