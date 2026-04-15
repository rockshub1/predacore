#!/bin/bash
# Project Prometheus Daily Standup Manager

echo "🔄 Generating standup report for $(date +%Y-%m-%d)"

# Get sprint progress
PROGRESS=$(./scripts/track_progress.sh | grep -E "code_commits|features_completed")

# Get recent changes
echo "📝 Recent Changes:"
git log --since="1 day ago" --pretty=format:" - %s (%an)"

# Display metrics
echo -e "\n📊 Current Progress:"
echo "$PROGRESS"

# Generate recommendations
echo -e "\n🔍 Analysis:"
COVERAGE=$(./scripts/track_progress.sh | grep test_coverage)
if [[ $COVERAGE =~ 31m ]]; then
    echo " - Need to improve test coverage"
fi

# Action items
echo -e "\n✅ Action Items:"
echo "1. Review pull requests"
echo "2. Address high-priority bugs"
echo "3. Update documentation"

# Schedule next check
echo -e "\n⏰ Next check scheduled for tomorrow at 9:00 AM"