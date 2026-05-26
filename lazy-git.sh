#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "🔍 Checking status..."
git status -s

# 1. Pull latest changes to prevent push rejections
echo "📥 Fetching latest changes from GitHub..."
if ! git pull origin main; then
    echo "❌ Error: Could not pull updates. You might have a conflict. Ask for help!"
    exit 1
fi

# 2. Stage EVERYTHING first (including brand new files)
echo "🚀 Staging files..."
git add .

# 3. NOW check if there are any staged changes to commit
if git diff --cached --quiet; then
    echo "✅ Everything is already up to date. Nothing new to push!"
    exit 0
fi

# 4. Prompt for a commit message
echo "✏️  Enter a quick note on what you changed (e.g., 'fixed login button'):"
read -r commit_message

if [ -z "$commit_message" ]; then
    commit_message="Update made on $(date +'%Y-%m-%d %H:%M')"
fi

# 5. Commit and Push
echo "💾 Committing changes..."
git commit -m "$commit_message"

echo "📤 Pushing to GitHub..."
if git push origin main; then
    echo "🎉 Success! Your changes are live on GitHub."
else
    echo "❌ Push failed. Try running the script again."
fi
