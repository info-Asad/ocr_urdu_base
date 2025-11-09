#!/bin/bash
# GitHub Upload Script for Urdu OCR Project

echo "🚀 Urdu OCR - GitHub Upload Helper"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files except those in .gitignore
echo "📁 Adding files to Git..."
git add .

# Create commit
echo "💾 Creating commit..."
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: Complete Urdu OCR Neural Network System"
fi
git commit -m "$commit_msg"

# Add remote repository
echo "🔗 Setting up GitHub remote..."
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/urdu-ocr.git): " repo_url

if [ ! -z "$repo_url" ]; then
    git remote add origin "$repo_url"
    echo "✅ Remote repository added"
    
    # Push to GitHub
    echo "⬆️ Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    
    echo "🎉 Successfully uploaded to GitHub!"
    echo "🌐 Your repository: $repo_url"
else
    echo "❌ No repository URL provided. Please add remote manually:"
    echo "   git remote add origin <your-repo-url>"
    echo "   git push -u origin main"
fi

echo ""
echo "📋 Next steps:"
echo "1. Go to your GitHub repository"
echo "2. Add a description and topics"
echo "3. Enable GitHub Pages (if needed)"
echo "4. Add collaborators (if needed)"
echo ""
echo "🎊 Your Urdu OCR project is now on GitHub!"